"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
# import functions
matplotlib.use('Agg')

def get_filelist(dir, Filelist):
    '''
    for filename in os.listdir(dir):
        print(filename)
        Filelist.append(filename)
    return Filelist
    '''
    newDir = dir

    if os.path.isfile(dir):
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # #if s == "xxx":
            #   #continue
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist

class VehicleRoutingDataset(Dataset):
    def __init__(self, max_load, opt_df=None, baselines_df=None, folder_path=None):
        super(VehicleRoutingDataset, self).__init__()

        my_demands, self.baselines_list, self.opt_list = list(), list(), list()
        instances_path_list = get_filelist(dir=folder_path, Filelist=[])
        self.instances_num = len(instances_path_list)
        self.instances_name = list()
        for instance_path in instances_path_list:
            name = os.path.basename(instance_path)

            baseline = -1;
            opt = -1
            if baselines_df is not None:
                base_result = baselines_df[baselines_df['name'] == name]['pure_obj']
                baseline = base_result.iloc[0]
                opt_result = opt_df[opt_df['Name'] == name]['Best LB']
                opt = opt_result.iloc[0]
            self.baselines_list.append(baseline)
            self.opt_list.append(opt)

            instance = [[0]]
            with open(instance_path, 'r', encoding='utf-8') as f:
                item_number = int(f.readline())
                capacity = int(f.readline())
                for id in range(item_number):
                    order = f.readline().split()
                    for count in range(np.int(order[1])):
                        instance[0].append(np.float(order[0]) / capacity)
            my_demands.append(instance)
            self.instances_name.append(name)

        for id in range(len(my_demands)):
            for _ in range(max_load-len(my_demands[id][0])+1):
                my_demands[id][0].append(0)

        self.isempty = False
        if 0 == len(my_demands):
            self.isempty = True
            return

        demands = torch.Tensor(my_demands)

        seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = len(demands)
        self.max_load = max_load

        # Depot location will be the first node in each
        # locations = torch.rand((num_samples, 2, input_size + 1))
        # self.static = locations
        self.static = torch.Tensor(my_demands)

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (self.num_samples, 1, max_load + 1)
        loads = torch.full(dynamic_shape, 1.)

        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        # demands = torch.randint(1, max_demand + 1, dynamic_shape)
        # demands = demands / float(max_load)
        # demands[:, 0, 0] = 0  # depot starts with a demand of 0

        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1], self.instances_name[idx], self.baselines_list[idx], self.opt_list[idx])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0).int() * demands.lt(loads).int()

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0).int()

        # print(new_mask.int())

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (1 - repeat_home).any():
            new_mask[(1 - repeat_home).nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0).int()
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():

            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.tensor(tensor.data, device=dynamic.device)


def reward(static, tour_indices):
    """
    bins number of solutions
    """

    rewards = list()
    for tour in tour_indices:
        reward = 0
        for idid in range(len(tour)):
            if 0 == idid:
                continue
            elif (0 == tour[idid]):
                if (0 == tour[idid-1]):
                    break
                else:
                    reward += 1
        rewards.append(reward)

    rewards = torch.Tensor(rewards)

    return rewards


def render(rewards, save_path):
    """save the rewards."""

    plt.plot(rewards)
    plt.savefig(save_path + '/training_rewards/train_rewards.png')
    np.savetxt(save_path + '/training_rewards/train_rewards.data', np.array(rewards))

'''
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    path = 'C:/Users/Matt/Documents/ffmpeg-3.4.2-win64-static/bin/ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path'] = path

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                             sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    all_lines = []
    all_tours = []
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        cur_tour = np.vstack((x, y))

        all_tours.append(cur_tour)
        all_lines.append(ax.plot([], [])[0])

        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

    from matplotlib.animation import FuncAnimation

    tours = all_tours

    def update(idx):

        for i, line in enumerate(all_lines):

            if idx >= tours[i].shape[1]:
                continue

            data = tours[i][:, idx]

            xy_data = line.get_xydata()
            xy_data = np.vstack((xy_data, np.atleast_2d(data)))

            line.set_data(xy_data[:, 0], xy_data[:, 1])
            line.set_linewidth(0.75)

        return all_lines

    anim = FuncAnimation(fig, update, init_func=None,
                         frames=100, interval=200, blit=False,
                         repeat=False)

    anim.save('line.mp4', dpi=160)
    plt.show()

    import sys
    sys.exit(1)
'''
