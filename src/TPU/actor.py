from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np

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
    def __init__(self, model_def, slevel_max, num_pe=64, par_RS=False, h_size=128, hidden_dim=10, device=None):
        super(Actor, self).__init__()

        self.model_def = model_def
        self.h_size = h_size
        self.slevel_max = slevel_max
        self.slevel_min = 1
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

        self.cluster_space = 6 if par_RS else 4
        self.parallel_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, self.cluster_space),
        )

        self.model_def_temp = np.array(np.array([162, 161, 224, 224, 7, 7]))

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
            tile_mask = torch.zeros(1, self.tile_size).float().fill_(float("-Inf")).to(self.device)
            tile_mask[:, :last_level_tile] = 0.
            tile_score = self.tile_decoder(h)
            tile_score = tile_score + tile_mask
            tile_prob = F.softmax(tile_score / self.tile_temperature, dim=1)
            tile_density = Categorical(tile_prob)
            tile_action = tile_density.sample()
            tile_log_prob = tile_density.log_prob(tile_action)
            return tile_action, tile_log_prob, last_level_tile == 1
