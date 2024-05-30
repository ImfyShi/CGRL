import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpn import GPN
import copy
import os
import parameters.read_instance as read_instance
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import shutil

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description="GPN with RL")
    parser.add_argument('--size', default=50, help="size of TSP")
    parser.add_argument('--epoch', default=1000000000, help="number of epochs")
    parser.add_argument('--batch_size', default=1, help='')
    parser.add_argument('--train_size', default=2500, help='')
    parser.add_argument('--val_size', default=1000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    args = vars(parser.parse_args())

    size = int(args['size'])
    learn_rate = args['lr']  # learning rate
    B = int(args['batch_size'])  # batch_size
    B_val = int(args['val_size'])  # validation size
    steps = int(args['train_size'])  # training steps
    n_epoch = int(args['epoch'])  # epochs
    save_root = './model/gpn_csp.pt'

    print('=========================')
    print('prepare to train')
    print('=========================')
    print('Hyperparameters:')
    print('size', size)
    print('learning rate', learn_rate)
    print('batch size', B)
    print('validation size', B_val)
    print('steps', steps)
    print('epoch', n_epoch)
    print('save root:', save_root)
    print('=========================')

    if os.path.exists('./data/generated_results'):
        shutil.rmtree('./data/generated_results')
    os.mkdir('./data/generated_results')

    base_folder = '/Users/sfy/Store/Benchmark/CuttingStockProblem/original_instances/instances/'
    instance_folder_list = [
        'Scholl_CSP/',
        '',
        'Scholl_CSP/',
        'Schwerin_CSP/',
        'Schwerin_CSP/',
        '',
        '',
        'ANI_AI_CSP/',
        'ANI_AI_CSP/',
        'Scholl_CSP/',
        'Falkenauer_CSP/',
        'Falkenauer_CSP/',
        '',
    ]
    instance_set_name_list = [
        'Scholl_3',
        'Waescher_CSP',
        'Scholl_2',
        'Schwerin_1',
        'Schwerin_2',
        'Randomly_Generated_CSP',
        'Hard28_CSP',
        'AI',
        'ANI',
        'Scholl_1',
        'Falkenauer_T',
        'Falkenauer_U',
        'Irnich_CSP',
        ]

    load_root = './model/gpn_csp.pt'

    model = torch.load(load_root)

    opt_baseline_table = pd.read_excel(r'./data/opt_baselines.xlsx')

    for folder, set_name in zip(instance_folder_list, instance_set_name_list):
        instance_set_path = base_folder + folder + set_name

        result_table = list()

        instance_path_list = read_instance.get_filelist(dir=instance_set_path, Filelist=[])
        instance_path_list.sort()
        B = 1

        pbar = tqdm(range(len(instance_path_list)))
        for instance_path_idx in pbar:
            instance_path = instance_path_list[instance_path_idx]
            if '.DS_Store' == os.path.basename(instance_path):
                continue

            instance_name = os.path.basename(instance_path)
            instance = read_instance.read_instance_file(instance_path=instance_path)

            opt_baseline_data = opt_baseline_table.loc[opt_baseline_table['Name'] == instance_name]
            opt_obj = opt_baseline_data['Best LB'].iloc[0]

            train_data = read_instance.obtain_instance_data(instance_list=[instance])
            size = len(train_data[0])

            data = copy.deepcopy(train_data)
            data = torch.Tensor(data)
            X = copy.deepcopy(data)

            mask = torch.where(X > 0.0, torch.tensor(0.0), torch.tensor(-np.inf)).squeeze()

            Y = copy.deepcopy(X)
            x = Y[:, 0, :]
            h = None
            c = None

            width_sequence = None
            for k in range(size):

                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)

                sampler = torch.distributions.Categorical(output)
                # idx = sampler.sample()  # now the idx has B elements
                idx = torch.argmax(output, dim=1)    # greedy baseline

                x = Y[[i for i in range(B)], idx.data].clone()
                if width_sequence is None:
                    width_sequence = x
                else:
                    width_sequence = np.c_[width_sequence, x]

                mask[idx.data] += -np.inf
                if not torch.any(mask == 0): mask[-1] = 0

            # TODO
            # reward
            R = 0
            for row_idx in range(len(width_sequence)):
                sum_len = 0
                ele_idx = 0
                while ele_idx < width_sequence[row_idx].shape[0]:
                    if sum_len + width_sequence[row_idx, ele_idx] > 1:
                        R += 1
                        sum_len = 0
                    else:
                        sum_len += width_sequence[row_idx, ele_idx]
                        ele_idx += 1
            gap = (R - opt_obj) / opt_obj * (100)
            result_table.append([instance_name, R, gap])
            pbar.set_description('Evaluating {}: {}'.format(set_name, instance_name))

        columns = ['name', 'obj', 'gap']

        df = pd.DataFrame(list(result_table), columns=columns)
        excel_name = "./data/generated_results/{}.xlsx".format(set_name)
        df.to_excel(excel_name, index=False)
