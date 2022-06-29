import numpy as np
import copy, random
import os
from subprocess import Popen, PIPE
import pandas as pd
import math
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
m_type_dicts = {1:"CONV", 2:"DSCONV", 3:"CONV", 4:"TRCONV"}


class MaestroEnvironment(object):
    def __init__(self, dimension, fitness="latency", par_RS=False, num_pe=64, l1_size=512, l2_size=108000, NocBW=81920000, slevel_min=2,slevel_max=2, fixedCluster=0, log_level=2):
        super(MaestroEnvironment,self).__init__()
        self.dimension = dimension
        self.dim_max = np.max(dimension)
        self.dimension_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        self.lastcluster_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        self.out_repr = set(["K", "C", "R", "S"])
        self.num_pe = num_pe
        self.fitness = fitness
        self.cluster_space = ["K", "C", "Y","X","R","S"] if par_RS else ["K", "C", "Y","X"]
        self.dim2id = {"K":1, "C":2, "Y":3, "X":4, "R":5, "S":6}
        self.id2dim = {1:"K", 2:"C", 3:"Y", 4:"X", 5:"R", 6:"S"}
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.NocBW = NocBW
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.fixedCluster = fixedCluster
        self.log_level = log_level
        self.scale = 2

        self.level = 1
        self.total_levels = self.slevel_min
        self.state = None
        self.best_reward = float('-inf')
        self.min_reward = float('inf')
        self.mode = 0
        self.mode_sequence = [2, 3, 4, 5, 6, 7]
        self.total_eps_reward = None
        self.last_reward = 0.
        self.level_steps = 8
        self.last_level_tiles = None
        self.parallel_mask = np.array([0, 0, 0, 0])

    def reset_dimension(self, dimension, fitness):
        self.dimension = dimension
        self.fitness = fitness

    def epoch_reset(self, dimension, fitness):
        self.level = 1
        self.total_levels = self.slevel_min
        self.state = np.ones(self.slevel_max*(7+1), dtype=np.int32)
        self.state[0] = self.dim2id['K']
        self.state[self.level_steps] = self.dim2id['C']
        # self.state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1])
        if self.slevel_min == 1:
            self.state = np.array([2, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        else:
            self.state = np.array([2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        # self.state[1] = min(self.dimension[self.state[0] - 1], self.num_pe)
        # self.state[2] = min(self.dimension[self.state[0] - 1], self.num_pe)
        # state: slevel_max*(para_dim, cluster_size, tile_size*6)

        self.mode = 0
        # self.mode_sequence = [0, 2, 3, 4, 5, 6, 7]
        self.mode_sequence = [2, 3, 4, 5, 6, 7]
        random.shuffle(self.mode_sequence)
        # for i in range(len(self.mode_sequence)):
        #     if self.mode_sequence[i] == 0:
        #         self.mode_sequence.insert(i+1, 1)
        self.mode_sequence.insert(0, 1)
        self.mode_sequence.insert(0, 0)
        # print(self.mode_sequence)
        # _, self.total_eps_reward, _ = self.get_reward()
        _, self.last_reward, _, _, _ = self.get_reward()
        # print(self.last_reward)
        self.best_reward = float('-inf')
        self.min_reward = self.last_reward.copy()
        self.dimension = dimension
        self.fitness = fitness
        self.dim_max = np.max(dimension)

        self.parallel_mask = np.array([0., 0., float("-inf"), 0.])
        last_level_max_tiles = np.array([64, 64, 56, 56, 7, 7])
        self.last_level_tiles = np.zeros(6, dtype=np.int32)
        if self.dimension[0] == 1:
            self.last_level_tiles[0] = 1
            self.parallel_mask = np.array([float("-inf"), 0., float("-inf"), 0.])
        else:
            self.last_level_tiles[0] = min(self.dimension[0], 64) // self.scale + 1
        if self.dimension[1] == 3:
            self.last_level_tiles[1] = 3
        else:
            self.last_level_tiles[1] = min(self.dimension[1], 64) // self.scale + 1
        self.last_level_tiles[2] = min(self.dimension[2], 56) // self.scale + 1
        self.last_level_tiles[3] = min(self.dimension[3], 56) // self.scale + 1
        # self.last_level_tiles[3] = self.dimension[3] // 7 + 1
        for i in range(4, 6):
            self.last_level_tiles[i] = min(last_level_max_tiles[i], self.dimension[i])
        print(self.last_level_tiles)

        next_state = np.zeros(self.total_levels * self.level_steps)
        next_state[0] = 1. / (self.slevel_max * self.level_steps)
        next_state[1] = self.state[0] / 4.
        next_state[2:] = self.state[2:] / self.dim_max
        if self.total_levels > 1:
            next_state[8] = self.state[8] / 4.
        if self.total_levels > 2:
            next_state[16] = self.state[16] / 4.
        next_state = np.concatenate((next_state, np.zeros((self.slevel_max - self.total_levels) * self.level_steps)))
        state_info = {}
        state_info['state'] = next_state
        state_info['parallel_mask'] = self.parallel_mask
        state_info['instruction'] = self.mode_sequence[self.mode]
        state_info['last_level_tiles'] = self.last_level_tiles
        state_info['cur_levels'] = self.level
        return state_info

    # def episode_reset(self):
    #     self.level = self.slevel_min
    #     self.mode = 0
    #     # _, self.last_reward, _ = self.get_reward()
    #
    #     self.parallel_mask = np.array([0., 0., 0., 0.])
    #     # self.last_level_tiles = np.array([10, 3, 11, 11, 7, 7])
    #     self.last_level_tiles = np.array([10, 10, 15, 15, 7, 7])
    #     if self.dimension[1] == 3:
    #         self.last_level_tiles[1] = 2
    #     for i in range(4, 6):
    #         self.last_level_tiles[i] = min(self.last_level_tiles[i], self.dimension[i])
    #     if self.dimension[2] == 224:
    #         self.last_level_tiles[2] = 15
    #         self.last_level_tiles[3] = 15
    #     elif self.dimension[2] == 112:
    #         self.last_level_tiles[2] = 13
    #         self.last_level_tiles[3] = 13
    #     elif self.dimension[2] == 56:
    #         self.last_level_tiles[2] = 11
    #         self.last_level_tiles[3] = 11
    #     elif self.dimension[2] == 28:
    #         self.last_level_tiles[2] = 9
    #         self.last_level_tiles[3] = 9
    #     elif self.dimension[2] == 14:
    #         self.last_level_tiles[2] = 8
    #         self.last_level_tiles[3] = 8
    #     elif self.dimension[2] == 7:
    #         self.last_level_tiles[2] = 7
    #         self.last_level_tiles[3] = 7
    #
    #     next_state = np.zeros(self.slevel_max * self.level_steps)
    #     next_state[0] = 1. / 16.
    #     next_state[1] = self.state[0] / 4.
    #     next_state[2:] = self.state[2:] / self.dim_max
    #
    #     state_info = {}
    #     state_info['state'] = next_state
    #     state_info['parallel_mask'] = self.parallel_mask
    #     state_info['instruction'] = self.mode
    #     state_info['last_level_tiles'] = self.last_level_tiles
    #     return state_info

    def scan_indv(self,indv):
        last_cluster_dict=self.lastcluster_dict
        for i in range(len(indv)-6,len(indv), 1):
            d, d_sz = indv[i]
            last_cluster_dict[d] = d_sz
        return  last_cluster_dict

    def get_out_repr(self, x):
        if x in self.out_repr:
            return x
        else:
            return x + "'"

    def get_reward(self):
        sol = []
        # for i in range(self.level):
        for i in range(self.total_levels):
            sol.append([self.id2dim[self.state[i*self.level_steps]], self.state[i*self.level_steps+1]])
            for j in range(1, 7):
                sol.append([self.id2dim[j], self.state[i*self.level_steps+j+1]])

        # print(self.state, sol)
        reward, latency, energy, l1_size, l2_size = self.oberserve_maestro(sol)
        constraint = (l1_size, l2_size)
        # print(sol, reward, constraint)
        # print(reward, l1_size, l2_size)

        # if self.min_reward is None:
        #     self.min_reward = reward
        # reward_saved = reward.copy()
        # self.min_reward = min(self.min_reward, reward_saved)
        # reward -= self.min_reward
        # self.total_eps_reward += reward
        return sol, reward, latency, energy, constraint

    def step(self, action):
        # print (self.last_reward)
        state_info = {}
        # print("step_start:", self.mode, self.mode_sequence[self.mode], action, self.state)
        done = 0

        if self.mode_sequence[self.mode] == 0:
            if len(action) == 2:
                stop_action, parallel_action = action
            else:
                parallel_action = action
            if len(action) == 2 and stop_action.cpu().numpy()[0] > 0:
                self.level -= 1
                done = 1
            else:
                parallel_action = parallel_action.cpu().numpy()[0]

                if self.level == 1:
                    self.state[(self.level - 1) * self.level_steps] = parallel_action + 1
                    cluster_size = self.dimension[self.state[(self.level-1)*self.level_steps]-1]
                    self.last_level_tiles[parallel_action] = min(self.last_level_tiles[parallel_action],
                                                                 math.ceil(
                                                                     cluster_size / self.num_pe) // self.scale + 1)
                    cluster_size = min(cluster_size, self.num_pe)
                    if self.dimension[0] == 1:
                        self.parallel_mask = np.array([float("-inf"), 0, 0., 0.])
                    else:
                        self.parallel_mask = np.array([0., 0, 0., 0.])
                else:
                    init_par_dim = 1
                    for i in range(4):
                        if self.parallel_mask[i] == 0:
                            init_par_dim = i + 1
                            break

                    if self.level > self.slevel_min:
                        self.total_levels += 1
                        self.state = np.concatenate((self.state, np.array([init_par_dim, 1, 1, 1, 1, 1, 1, 1])))

                    self.state[(self.level - 1) * self.level_steps] = parallel_action + 1
                    par_dim = self.state[(self.level - 1) * self.level_steps]
                    cluster_size = self.state[(self.level-2)*self.level_steps + par_dim + 1]
                    self.last_level_tiles[parallel_action] = min(self.last_level_tiles[parallel_action],
                                                                 math.ceil(
                                                                     cluster_size / self.num_pe) // self.scale + 1)
                    cluster_size = min(cluster_size, self.num_pe)
                    # self.last_level_tiles[0] = 1
                    # self.last_level_tiles[1] = 1
                    if self.state[(self.level - 2) * 8 + 2] == 1:
                        self.parallel_mask[0] = float("-inf")
                    # self.mode += 2
                # print("cluster_size: ", cluster_size)

                self.state[(self.level - 1) * self.level_steps + 1] = cluster_size

                self.parallel_mask[parallel_action] = float('-inf')
                self.mode += 1
        else:
            tile_action = action.cpu().numpy()[0]
            if self.mode_sequence[self.mode] == 2:
                if tile_action == 0:
                    tile_size = 1
                else:
                    tile_size = self.scale * tile_action
            elif self.mode_sequence[self.mode] == 3:
                if tile_action == 0:
                    tile_size = 1
                else:
                    if self.dimension[1] == 3:
                        tile_size = tile_action + 1
                    else:
                        tile_size = self.scale * tile_action
            elif self.mode_sequence[self.mode] == 4 or self.mode_sequence[self.mode] == 5:
                if tile_action == 0:
                    tile_size = 1
                else:
                    tile_size = self.scale * tile_action
            else:
                tile_size = tile_action + 1
            self.state[(self.level-1)*self.level_steps+self.mode_sequence[self.mode]] = tile_size
            self.last_level_tiles[self.mode_sequence[self.mode] - 2] = tile_action + 1

        next_state = np.zeros(self.total_levels * self.level_steps)
        next_state[0] = (self.mode_sequence[self.mode] + 1) / (self.slevel_max * self.level_steps)
        next_state[1] = self.state[0] / 4.
        next_state[2:] = self.state[2:] / self.dim_max
        if self.total_levels > 1:
            next_state[8] = self.state[8] / 4.
        if self.total_levels > 2:
            next_state[16] = self.state[16] / 4.
        next_state = np.concatenate((next_state, np.zeros((self.slevel_max - self.total_levels) * self.level_steps)))

        sol, reward, latency, energy, constraint = self.get_reward()
        l1_size, l2_size = constraint
        info = 'success'
        if reward is None:
            # reward = -2.
            # print("fail: ", self.min_reward, l1_size, l2_size)
            # reward = self.min_reward * (l1_size / self.l1_size + l2_size / self.l2_size)
            reward = self.min_reward * 10
            info = "fail"
            done = 1
            reward_saved = float('-inf')
            self.last_reward = self.min_reward * 10
        else:
            reward_saved = reward.copy()
            # if self.last_reward is None or reward_saved > self.last_reward:
            #     reward = 2.
            # else:
            #     reward = -1.
            reward = reward_saved - self.last_reward
            self.last_reward = reward_saved
            self.min_reward = min(self.min_reward, reward_saved)
            # print ("success:", reward)

        self.mode += 1

        if self.mode == self.level_steps:
            if self.level == self.slevel_max and not done:
                done = 1
            self.level += 1
            self.mode = 0
        # print('end_step:', self.state, next_state, self.last_level_tiles, reward, constraint)
        state_info['state'] = next_state
        state_info['parallel_mask'] = self.parallel_mask
        state_info['instruction'] = self.mode_sequence[self.mode]
        state_info['last_level_tiles'] = self.last_level_tiles
        state_info['cur_levels'] = self.level
        return state_info, sol, reward, reward_saved, latency, energy, constraint, done, info

    def write_maestro(self, indv, layer_id=0, m_file=None):
        m_type = m_type_dicts[int(self.dimension[-1])]
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            fo.write("Layer {} {{\n".format(m_type))
            fo.write("Type: {}\n".format(m_type))
            fo.write("Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*self.dimension))
            fo.write("Dataflow {\n")
            for k in range(0, len(indv), 7):
                for i in range(k, k+7):
                    d, d_sz = indv[i]
                    if i%7==0:
                        if k != 0:
                            fo.write("Cluster({},P);\n".format(d_sz))
                    else:
                        sp = "SpatialMap" if d == indv[k][0] else "TemporalMap"
                        if not (m_type =="DSCONV" and self.get_out_repr(d) =="K"):
                            fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
            fo.write("}\n")
            fo.write("}\n")
            fo.write("}")
        # with open("{}.m".format(m_file), "r") as fo:
        #     lines = fo.readlines()
        #     for line in lines:
        #         print(line)

    def oberserve_maestro(self, indv):
        m_file = "{}".format(random.randint(0, 2**32))
        self.write_maestro(indv,m_file=m_file)
        # print(num_pe, bw, l1_size)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        # command = [self._executable,
        #            "--Mapping_file={}.m".format(m_file),
        #            "--full_buffer=false", "--noc_bw={}".format(self.NocBW),
        #            "--noc_hops=1", "--noc_hop_latency=1",
        #            "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
        #            "--num_simd_lanes=1", "--l1_size={}".format(self.l1_size),
        #            "--l2_size={}".format(self.l2_size), "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw=81920000",
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
                   "--num_simd_lanes=1", "--l1_size=81920000",
                   "--l2_size=81920000", "--print_res=false", "--print_res_csv_file=true",
                   "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]


        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        os.remove("./{}.m".format(m_file)) if os.path.exists("./{}.m".format(m_file)) else None
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            # print(runtime, throughput, energy, area, l1_size, l2_size, mac, power)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]

            def catch_exception():
                if l1_size > self.l1_size or l2_size > self.l2_size or runtime < 1 or l1_size < 0 or l2_size < 0:
                    return True
                else:
                    return False

            if len(str(stdout)) > 3 or catch_exception():
                # print(stdout, catch_exception())
                return None, None, None, np.mean(l1_size), np.mean(l2_size)
            return self.judge()
        except Exception as e:
            print(e, indv)
            return None, None, None, -1, -1

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        for term in [self.fitness]:
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "EDP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            else:
                raise NameError('Undefined fitness type')
            values.append(reward)
            # print("values: ", values)
        values.append(l1_size)
        values.append(l2_size)
        # return values
        # return -energy * runtime, -runtime, -energy, l1_size, l2_size
        return reward, -runtime, -energy, l1_size, l2_size

    def print_indv(self, indv,fd=False):
        for k in range(0, len(indv), 7):
            if fd:
                fd.write("\n{}".format(indv[k:k+7]))
            else:
                print(indv[k:k+7])