import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from model import DRL4TSP

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

from tasks import vrp
from tasks.vrp import VehicleRoutingDataset

def validate(data_loader, actor, reward_fn):
    """Used to monitor progress on a validation set & optionally plot solution."""

    # actor.eval()

    names, rewards, baselines, opts = [], [], [], []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0, name, baselines_result, opts_result = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
            # tour_indices, _ = actor(static, dynamic, x0)

        # reward = reward_fn(static, tour_indices).mean().item()
        reward = reward_fn(static, tour_indices)
        names += name; rewards += reward; baselines += baselines_result; opts += opts_result;
    rewards=np.array(rewards); baselines=np.array(baselines); opts=np.array(opts)
    alg_gaps = (rewards-opts)/opts*100; baseline_gap_gaps = (baselines-opts)/opts*100; improved_gaps = (baselines-rewards)/opts*100;
    return names, rewards, baselines, opts, alg_gaps, baseline_gap_gaps, improved_gaps


parser = argparse.ArgumentParser(description='Combinatorial Optimization')
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--task', default='vrp')
parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
parser.add_argument('--actor_lr', default=5e-4, type=float)
parser.add_argument('--critic_lr', default=5e-4, type=float)
parser.add_argument('--max_grad_norm', default=2., type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--layers', dest='num_layers', default=1, type=int)
parser.add_argument('--train-size', default=1000000, type=int)
parser.add_argument('--valid-size', default=1000, type=int)

args = parser.parse_args()

STATIC_SIZE = 1  # (x, y)
DYNAMIC_SIZE = 2  # (load, demand)

base_path = '/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/'

instance_folder_list = [
    '',
    '',
    'ANI_AI_CSP/',
    'ANI_AI_CSP/',
    'Scholl_CSP/',
    '',
    'Falkenauer_CSP/',
    'Falkenauer_CSP/',
    '',
    'Scholl_CSP/',
    'Scholl_CSP/',
    'Schwerin_CSP/',
    'Schwerin_CSP/',
    ]
instance_set_name_list = [
    'Randomly_Generated_CSP',
    'Hard28_CSP',
    'AI',
    'ANI',
    'Scholl_3',
    'Waescher_CSP',
    'Falkenauer_T',
    'Falkenauer_U',
    'Irnich_CSP',
    'Scholl_1',
    'Scholl_2',
    'Schwerin_1',
    'Schwerin_2',
    ]

# baselines_result = pd.read_excel('./data/baselines_on_all_dataset.xlsx')
baselines_result = None
opt_result = pd.read_excel('./data/opt_solutions.xlsx')
max_nodes_num = 2000

for instance_folder, instance_set_name in zip(instance_folder_list, instance_set_name_list):
    start_time = time.time()

    train_data = VehicleRoutingDataset(max_load=max_nodes_num, opt_df=opt_result, baselines_df=baselines_result, folder_path=base_path+instance_folder+instance_set_name)
    if train_data.isempty:
        continue
    train_loader = DataLoader(train_data, 1, True, num_workers=0)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    path = 'vrp/checkpoints/actor.pt'
    actor.load_state_dict(torch.load(path, device))

    data_table = list()

    names, rewards, baselines, opts, alg_gaps, baseline_gap_gaps, improved_gaps = validate(train_loader, actor, vrp.reward)

    for name, reward, alg_gap in zip(names, rewards, alg_gaps):
        data_table.append([name, reward, alg_gap])

    df = pd.DataFrame(data_table, columns=['name', 'ptrnet', 'alg_gap'])
    excel_name = "./data/deterministic/drl4vrp_on_{}.xlsx".format(instance_set_name)
    df.to_excel(excel_name, index=False)
