import os
import torch_geometric
import torch
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from random import shuffle
import math
import yaml

import utilities
import agent.ppo as agent

EP_MAX = 1000000000

if __name__ == '__main__':
    utilities.fix_seed()
    with open('train_config.yaml', encoding='utf-8') as f:
        parameters = yaml.load(f.read(), Loader=yaml.FullLoader)
    gpu_id = parameters['gpu_id']
    REUSME = parameters['REUSME']
    batch_size = parameters['batch_size']
    is_stochastic = parameters['is_stochastic']
    # batch_size = train_env_num

    current_state_path, best_state_path = '../model/current_model_state.pth', '../model/best_model_state.pth'
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    original_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/instances/'
    tobetrained_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/tobetrained_instances/instances/root/'
    base_folder = original_folder + parameters['base_folder']
    write_folder = tobetrained_folder + parameters['write_folder']

    train_instance_folder_list = [
        ''
    ]
    train_instance_set_name_list = [
        ''
    ]
    test_instance_folder_list = [
        ''
    ]
    test_instance_set_name_list = [
        ''
    ]

    # instance_set_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/debug/'
    initialed_log_path = '../data/generated_results/initialed.log'
    file_obj = open(initialed_log_path, 'w+')
    file_obj.close()
    initialing_log_path = '../data/generated_results/initialing.log'
    file_obj = open(initialing_log_path, 'w+')
    file_obj.close()

    baselines_result = pd.read_excel('../data/baselines/FFD_baselines_on_all_dataset.xlsx')
    # baselines_result = None
    opt_result = pd.read_excel('../data/baselines/opt_baselines.xlsx')
    train_instances_set_folder = train_instance_folder_list[0] + train_instance_set_name_list[0]
    train_set_instances_folder = base_folder+train_instances_set_folder
    train_set_root_data_folder = write_folder+train_instances_set_folder
    train_env_list, train_env_num = utilities.load_env_list(instance_folder=train_set_instances_folder, write_instance_data_folder=train_set_root_data_folder, is_collect_root_info=False, initialed_log_path=initialed_log_path, initialing_log_path=initialing_log_path, baselines_result=baselines_result, opt_results=opt_result)

    test_instances_set_folder = test_instance_folder_list[0] + test_instance_set_name_list[0]
    test_set_instances_folder = base_folder+test_instances_set_folder
    test_set_root_data_folder = write_folder+test_instances_set_folder
    test_env_list, test_env_num = utilities.load_env_list(instance_folder=test_set_instances_folder, write_instance_data_folder=test_set_root_data_folder, is_collect_root_info=False, initialed_log_path=initialed_log_path, initialing_log_path=initialing_log_path, baselines_result=baselines_result, opt_results=opt_result)

    model_state = None
    if REUSME:
        model_state = torch.load(current_state_path)
        ppo = agent.PPO(device=device, model_state=model_state)
    else:
        if os.path.exists('../training_log/train'):
            shutil.rmtree('../training_log/train')
        if os.path.exists('../training_log/test'):
            shutil.rmtree('../training_log/test')
        if os.path.exists('../model'):
            shutil.rmtree('../model')
        os.mkdir('../training_log/train')
        os.mkdir('../training_log/test')
        os.mkdir('../model')
        ppo = agent.PPO(device=device, model_state=None)

    writer_train = SummaryWriter('../training_log/train')
    writer_test = SummaryWriter('../training_log/test/')

    shuffle(train_env_list)
    if batch_size is None:
        batch_size = len(train_env_list)
    group_num = math.ceil(float(train_env_num)/batch_size)
    group_gaps = np.ones(group_num)*1e10 if model_state is None else model_state['group_gaps']

    iter_num = -1 if model_state is None else model_state['iter_num'] - 1
    ep_start = 0 if model_state is None else model_state['ep']
    group_idx_start = 0 if model_state is None else model_state['group_idx']
    for ep in range(ep_start, EP_MAX):
        for group_idx in range(group_idx_start, group_num):
            iter_num += 1
            rwr_gap_list_for_each_epoh, baselines_gap_list = [], []
            buffer_item, buffer_edge, buffer_edge_indices, buffer_column, buffer_a, buffer_r, buffer_column_num, buffer_item_num, buffer_ep_r, buffer_ep_rwr = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
            pbar = tqdm(range(group_idx*batch_size, (group_idx+1)*batch_size if (group_idx+1)*batch_size < train_env_num else train_env_num, 1))
            for env_id in pbar:
                env = train_env_list[env_id]
                print('instance name: {}'.format(env.instance_name))
                env.load_minknapdll()
                _buffer_item, _buffer_edge, _buffer_edge_indices, _buffer_column, _buffer_a, _buffer_r, _buffer_column_num, _buffer_item_num, ep_r, ep_rwr, mip_select_obj = utilities.Rollout(env=env, agent=ppo, device=device, is_stochastic=is_stochastic)
                buffer_item+=_buffer_item; buffer_edge+=_buffer_edge; buffer_edge_indices+=_buffer_edge_indices; buffer_column+=_buffer_column; buffer_a+=_buffer_a; buffer_r+=_buffer_r; buffer_column_num+=_buffer_column_num; buffer_item_num+=_buffer_item_num; buffer_ep_r.append(ep_r); buffer_ep_rwr.append(ep_rwr);
                rwr_gap_list_for_each_epoh.append(100*(ep_rwr-env.opt)/env.opt)
                baselines_gap_list.append(100*(ep_rwr-env.baseline)/env.opt)
                pbar.set_description('train EP: {} | group idx: {} | mean gap: {} | name:f {}%'.format(ep, group_idx, np.mean(rwr_gap_list_for_each_epoh), env.instance_name))
            train_data = utilities.GraphDataset(buffer_item, buffer_edge, buffer_edge_indices,
                                                buffer_column, buffer_a, buffer_r,
                                                buffer_column_num, buffer_item_num)
            train_loader = torch_geometric.loader.DataLoader(train_data, train_data.len(), shuffle=False)

            current_gap = np.mean(rwr_gap_list_for_each_epoh)
            writer_train.add_scalar('train/mean gap (%)', current_gap, iter_num)
            writer_train.add_scalar('train/mip select and ffd gap (%)', float(env.baseline-mip_select_obj)/env.opt, iter_num)
            if 0==ep:
                writer_train.add_scalar('train/group improved gap (%)', 0, iter_num)
            else:
                writer_train.add_scalar('train/group improved gap (%)', group_gaps[group_idx] - current_gap, iter_num)
            if baselines_result is not None:
                writer_train.add_scalar('train/baselines gap (%)', np.mean(baselines_gap_list), iter_num)
            changed_flag = (group_gaps[group_idx]==current_gap)
            print('group_gaps[{}]={} | current_gap={} | {}'.format(group_idx,group_gaps[group_idx], current_gap, 'matained group gap has changed' if group_gaps[group_idx] > current_gap else ''))

            current_model_state = {
                "criticNet": ppo.criticNet.state_dict(),
                "actorNet": ppo.actorNet.state_dict(),
                'c_optimizer': ppo.c_optimizer.state_dict(),
                'a_optimizer': ppo.a_optimizer.state_dict(),
                'group_gaps': group_gaps,
                "iter_num": iter_num,
                'ep': ep,
                'group_idx': group_idx,
            }
            if group_gaps[group_idx] > current_gap:
                group_gaps[group_idx] = current_gap
                current_model_state['group_gaps'] = group_gaps
                torch.save(current_model_state, best_state_path)
            torch.save(current_model_state, current_state_path)

            for batch in train_loader:
                # print('update function')
                # batch = batch.cuda()
                batch = batch.to(device)
                closs, aloss = ppo.update(item_features=batch.item_features, edge_features=batch.edge_features, edge_indices=batch.edge_index, column_features=batch.column_features, actions=batch.actions, rewards=batch.rewards, column_nums=train_data.column_nums, item_nums=train_data.item_nums)
                writer_train.add_scalar('train/closs', closs, iter_num)
                writer_train.add_scalar('train/aloss', aloss, iter_num)

            '''
            if 0 == ep % 1:
                is_better_count = 0.0
                rwr_gap_list_for_each_epoh, baseline_gap_list_for_each_epoh = [], []
                improved_list = []
                pbar = tqdm(range(test_env_num))
                for env_id in pbar:
                    env = test_env_list[env_id]
                    buffer_item, buffer_edge, buffer_edge_indices, buffer_column, buffer_a, buffer_r, buffer_column_num, buffer_item_num, ep_r, ep_rwr = utilities.Rollout(
                        env=env, agent=ppo, device=device)
                    improved_list.append(100*float((env.baseline)-(ep_rwr))/env.opt)
                    if (ep_rwr < env.baseline) or (ep_rwr==env.opt):
                        is_better_count += 1.0
                    rwr_gap_list_for_each_epoh.append(100*(ep_rwr-env.opt)/env.opt)
                    baseline_gap_list_for_each_epoh.append(100*(env.baseline-env.opt)/env.opt)
                    pbar.set_description('test EP: {} | mean improved gap {}%'.format(ep, np.mean(improved_list)))
                writer_test.add_scalar('test/better instance rate (%)', 100*is_better_count / test_env_num, ep)
                writer_train.add_scalars('test/mean gap(%)', {"objective": np.mean(rwr_gap_list_for_each_epoh),
                                                               "baseline": np.mean(baseline_gap_list_for_each_epoh)}, ep)
            # torch.cuda.empty_cache()
            '''
