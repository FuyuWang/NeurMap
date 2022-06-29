'''
Two action at a time discrete
'''

import torch.nn as nn

from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.LSTMCell:
        nn.init.orthogonal_(m.weight_hh)
        nn.init.orthogonal_(m.weight_ih)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, model_def, slevel_min, slevel_max, num_pe=64, par_RS=False, h_size=128, hidden_dim=10, device=None):
        super(Actor, self).__init__()

        self.model_def = model_def
        self.h_size = h_size
        self.slevel_max = slevel_max
        self.slevel_min = slevel_min
        dim_length = slevel_max * 8
        self.dim_encoder = nn.Sequential(
            nn.Linear(dim_length, dim_length*hidden_dim),
            nn.ReLU(),
            nn.Linear(dim_length*hidden_dim, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
        )

        # tile_length = slevel_max * 8
        # self.tile_encoder = nn.Sequential(
        #     nn.Linear(tile_length, tile_length * hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(tile_length * hidden_dim, h_size),
        #     nn.ReLU(),
        #     nn.Linear(h_size, h_size),
        #     nn.ReLU(),
        #     nn.Linear(h_size, h_size),
        #     nn.ReLU(),
        # )

        self.cluster_space = 6 if par_RS else 4
        self.parallel_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, self.cluster_space),
        )

        self.model_def_temp = np.array(np.array([162, 161, 224, 224, 7, 7]))
        # self.model_def_temp = [7, 3, 7, 7, 7, 7]
        # self.model_def_temp = np.minimum(model_def, num_pe)
        # self.tile_size = np.max(self.model_def_temp)
        # TODO
        self.tile_size = 512
        self.tile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, self.tile_size),
            # nn.Softmax()
        )

        self.stop_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1),
            nn.Sigmoid()
        )

        self.lstm = torch.nn.LSTMCell(h_size, h_size)
        # self.kc_mask = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48,
        #                          64, 96, 128, 192, 256, 384, 512])
        # self.k_mask = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
        # self.yx_mask = np.array([1, 7, 14, 21, 28, 42, 56, 84, 112, 168, 224])

        self.parallel_temperature = 1.
        self.order_temperature = 1.
        self.tile_temperature = 1.
        self.lstm_value = None

        self.init_weight()
        self.device = device

    def reset(self):
        self.lstm_value = self.init_hidden()

    def init_weight(self):
        self.apply(init_weights)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(1, self.h_size),
                weight.new_zeros(1, self.h_size))

    def set_tile_temperature(self, temp):
        self.tile_temperature = temp

    def set_order_temperature(self, temp):
        self.order_temperature = temp

    def set_parallel_temperature(self, temp):
        self.parallel_temperature = temp

    def forward(self, state_info):
        '''
        :param state dim_info if instruction == 0, origin tile state if instruction in [1,6], origin tile and order state if instruction in [7, 12]
        :param instruction int parallel action, order action or tile action
        :param last_level_tiles  next level tile <= last level tile
        :return: parallel dim action
        '''
        # print("action: ", state_info)
        state = torch.from_numpy(state_info['state']).type(torch.FloatTensor).to(self.device)
        parallel_mask = torch.from_numpy(state_info['parallel_mask']).type(torch.FloatTensor).to(self.device)
        instruction = state_info['instruction']
        last_level_tiles = state_info['last_level_tiles']
        state = state.unsqueeze(0)
        dim_feat = self.dim_encoder(state)
        h, x = self.lstm(dim_feat, self.lstm_value)
        self.lstm_value = (h, x)
        if instruction == 0:
            parallel_mask = parallel_mask.unsqueeze(0)
            parallel_score = self.parallel_decoder(h) + parallel_mask
            parallel_prob = F.softmax(parallel_score / self.parallel_temperature, dim=1)
            # print("parallel_prob: ", parallel_prob)
            parallel_density = Categorical(parallel_prob)
            parallel_action = parallel_density.sample()
            parallel_log_prob = parallel_density.log_prob(parallel_action)
            if state_info['cur_levels'] > self.slevel_min:
                stop_score = self.stop_decoder(h).contiguous().view(1)
                stop_density = Bernoulli(stop_score)
                stop_action = stop_density.sample()
                stop_log_prob = stop_density.log_prob(stop_action)
                return [stop_action, parallel_action], [stop_log_prob, parallel_log_prob], False
            else:
                return parallel_action, parallel_log_prob, False
        else:
            last_level_tile = last_level_tiles[instruction - 2]
            # print("tile_feat:", state, instruction)

            # print(start, end)
            tile_mask = torch.zeros(1, self.tile_size).float().fill_(float("-Inf")).to(self.device)
            # if instruction == 2:
            #     tile_mask[:, self.k_mask - 1] = 0
            # elif instruction == 4 or instruction == 5:
            #     tile_mask[:, self.yx_mask - 1] = 0
            # else:
            #     tile_mask[:, :self.model_def_temp[instruction-2]] = 0
            #
            # tile_mask[:, last_level_tile:] = float("-Inf")
            # print (torch.nonzero(tile_mask.squeeze() == 0))
            tile_mask[:, :last_level_tile] = 0.
            tile_score = self.tile_decoder(h)
            tile_score = tile_score + tile_mask
            tile_prob = F.softmax(tile_score / self.tile_temperature, dim=1)
            # print(end, start, tile_prob.size(), self.tile_decoder(h).size())
            tile_density = Categorical(tile_prob)
            # print("tile_prob: ", tile_prob)
            # print("tile_feat: ", start, last_level_tile, end, tile_mask, tile_prob)
            tile_action = tile_density.sample()
            # print("tile_feat: ", start, last_level_tile, end, tile_action)
            tile_log_prob = tile_density.log_prob(tile_action)
            return tile_action, tile_log_prob, last_level_tile == 1
        # else:
        #     order_feat = self.order_encoder(state[0]).unsqueeze(0)
        #     # print("tile_feat:", state, instruction)
        #     tile_feat = self.tile_encoder(state[1]).unsqueeze(0)
        #     h, x = self.lstm(order_feat + tile_feat, self.lstm_value)
        #     self.lstm_value = (h, x)
        #     order_score = self.order_decoder(h)
        #     order_prob = F.softmax(order_score / self.order_temperature, dim=1)
        #     order_density = Categorical(order_prob)
        #     order_action = order_density.sample()
        #     order_log_prob = order_density.log_prob(order_action)
        #
        #     return order_action, order_log_prob
