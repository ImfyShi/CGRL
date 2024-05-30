"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

import shutil

from model import DRL4TSP, Encoder

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn):
    """Used to monitor progress on a validation set & optionally plot solution."""

    # actor.eval()

    rewards, baselines, opts = [], [], []
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
        rewards += reward; baselines += baselines_result; opts += opts_result;
    rewards=np.array(rewards); baselines=np.array(baselines); opts=np.array(opts)
    alg_gap = (rewards-opts)/opts*100; baseline_gap_gap = (baselines-opts)/opts*100; improved_gap = (baselines-rewards)/opts*100;
    return alg_gap.mean(), baseline_gap_gap.mean(), improved_gap.mean()


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    # save_dir = os.path.join(task, '%d' % num_nodes, now)
    save_dir = task

    if os.path.exists(save_dir + '/training_rewards/'):
        shutil.rmtree(save_dir + '/training_rewards/')
    os.makedirs(save_dir + '/training_rewards/')
    if os.path.exists('train_log'):
        shutil.rmtree('train_log')
    writer_train = SummaryWriter('train_log/train')
    writer_test = SummaryWriter('train_log/test/')


    actor.eval()

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf
    step = -1
    train_rewards_list, test_rewards_list = list(), list()

    for epoch in range(2000000000000):
        instance_id = -1
        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        # pbar = tqdm(range(train_data.instances_num))

        for batch_idx, batch in enumerate(train_loader):
            # pbar.set_description('Epoch: {}'.format(epoch))
            step += 1
            instance_id += 1
            print('Epoch: {}, instance id: {}'.format(epoch,instance_id))
            static, dynamic, x0, name, baselines_result, opts_result = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            train_rewards_list.append(reward[0].data.item()/baselines_result[0].data.item())

            save_path = os.path.join(checkpoint_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)
            save_path = os.path.join(checkpoint_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            writer_train.add_scalar('train/improved gap (%)', torch.div(baselines_result-reward, opts_result).mean()*100, step)
            writer_train.add_scalars('train/mean gap (%)', {"alg": torch.div(reward-opts_result, opts_result).mean()*100,
                                                          "baseline": torch.div(baselines_result-opts_result, opts_result).mean()*100}, step)
            if 0 == step % 10:
                mean_alg_gap, mean_baseline_gap_gap, mean_improved_gap = validate(valid_loader, actor, reward_fn)
                writer_test.add_scalar('test/improved gap (%)', mean_improved_gap, step)
                writer_test.add_scalars('test/mean gap (%)', {"alg": mean_alg_gap,
                                                                "baseline": mean_baseline_gap_gap},step)

def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    baselines_result = pd.read_excel('./baselines_on_all_dataset.xlsx')
    opt_result = pd.read_excel('./opt_solutions.xlsx')

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 1 # demand
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]
    max_nodes_num = 2000

    train_data = VehicleRoutingDataset(max_load=max_nodes_num, opt_df=opt_result, baselines_df=baselines_result, folder_path='/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/train_set/')

    valid_data = VehicleRoutingDataset(max_load=max_nodes_num, opt_df=opt_result, baselines_df=baselines_result, folder_path='/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/test_set/')

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render
    kwargs['batch_size'] = train_data.instances_num
    kwargs['batch_size'] = 2

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.train_size,
                          max_nodes_num,
                          max_nodes_num,
                          MAX_DEMAND,
                          args.seed + 2,
                          'B:/BinPackingProblem/debug_set/')

    # test_dir = 'test'
    # test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    # out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    # print('Average tour length: ', out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    # parser.add_argument('--checkpoint', default="vrp/checkpoints/")
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
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    args = parser.parse_args()

    if args.task == 'vrp':
        train_vrp(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)
