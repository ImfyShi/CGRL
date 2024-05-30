import os
import torch_geometric
import torch
from tqdm import tqdm
import shutil
import pandas as pd
import numpy as np
import multiprocessing
import yaml
import argparse

import utilities
import agent.ppo as agent
from eval_our_method.eval_our_method import eval_pure_results as eval_pure_gnn_policy
from baselines.eval_policy import Evaluate as eval_policy

STOCHASTIC = 'stochastic'
DETERMINISTIC = 'deterministic'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Collect Evalution Results')
    parser.add_argument('--instance_set_group')
    parser.add_argument('--policy_mode')
    args = parser.parse_args()
    assert (args.policy_mode==STOCHASTIC) or (args.policy_mode==DETERMINISTIC)
    print(args.policy_mode)
    is_stochastic = True if args.policy_mode==STOCHASTIC else False

    utilities.fix_seed()

    with open('eval_config.yaml', encoding='utf-8') as f:
        parameters = yaml.load(f.read(), Loader=yaml.FullLoader)

    device = torch.device("cpu")
    best_state_path = '../model/best_model_state.pth'
    model_state = torch.load(best_state_path,map_location=device)
    ppo = agent.PPO(device=device, model_state=model_state)
    # ppo = torch.load('../model/current.model', map_location=torch.device('cpu'))

    dataset_log_path = '../data/generated_results/dataset.log'
    file_obj = open(dataset_log_path, 'w+')
    file_obj.close()

    policy_name = parameters['policy_name']

    base_folder = parameters['base_folder']
    write_folder = parameters['write_folder']

    instance_folder_list = parameters['eval_instance_sets_path'][args.instance_set_group]['instance_folder_list']
    instance_set_name_list = parameters['eval_instance_sets_path'][args.instance_set_group]['instance_set_name_list']

    opt_result = pd.read_excel('../data/baselines/opt_baselines.xlsx')
    ffd_result = pd.read_excel('../data/baselines/FFD_baselines_on_all_dataset.xlsx')

    data = np.ones((1,13)) * 10000
    meangapdf = pd.DataFrame(data, columns=[
        'AI',
        'ANI',
        'Falkenauer_T',
        'Falkenauer_U',
        'Hard28_CSP',
        'Irnich_CSP',
        'Randomly_Generated_CSP',
        'Scholl_1',
        'Scholl_2',
        'Scholl_3',
        'Schwerin_1',
        'Schwerin_2',
        'Waescher_CSP',
    ], dtype=float)

    for instance_folder, instance_set_name in zip(instance_folder_list, instance_set_name_list):
        print('set {} results are collecting'.format(instance_set_name))
        collected_log_path = '../data/generated_results/collected.log'
        file_obj = open(collected_log_path, 'w+')
        file_obj.close()
        collecting_log_path = '../data/generated_results/collecting.log'
        file_obj = open(collecting_log_path, 'w+')
        file_obj.close()
        initialed_log_path = '../data/generated_results/initialed.log'
        file_obj = open(initialed_log_path, 'w+')
        file_obj.close()
        initialing_log_path = '../data/generated_results/initialing.log'
        file_obj = open(initialing_log_path, 'w+')
        file_obj.close()

        with open(dataset_log_path, 'a+') as file_object:
            file_object.writelines('{}'.format(instance_set_name) + '\n')
        original_instance_folder = base_folder + instance_folder + instance_set_name
        make_folder_path = write_folder + instance_folder + instance_set_name
        env_list, env_num = utilities.load_env_list(instance_folder=original_instance_folder,
                                                    write_instance_data_folder=make_folder_path,
                                                    is_collect_root_info=False, initialed_log_path=initialed_log_path,
                                                    initialing_log_path=initialing_log_path, baselines_result=ffd_result,
                                                    opt_results=opt_result)

        print('{} set {} is collecting'.format(env_num, instance_set_name))
        task_pool = multiprocessing.Pool(int(multiprocessing.cpu_count()))
        # task_pool = multiprocessing.Pool(2)
        data_table = multiprocessing.Manager().list()
        is_better_count = 0.0
        pbar = tqdm(range(len(env_list)))
        for env_idx in pbar:
            env = env_list[env_idx]
            env.load_minknapdll()
            pbar.set_description('{}: {} '.format(instance_set_name, env.instance_name))
            eval_pure_gnn_policy(data_table, env, ppo, utilities.Rollout, collected_log_path, collecting_log_path, device, is_stochastic)

        task_pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        task_pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        # env.instance_name, ep_rwr, float(ep_rwr-env.opt)/env.opt*100, mip_select_obj, float(mip_select_obj-env.baseline)/env.opt*100
        columns = ['name', 'obj', 'gap', 'sp_obj']

        df = pd.DataFrame(list(data_table), columns=columns)
        excel_name = "../data/generated_results/{}/{}_on_{}.xlsx".format(args.policy_mode, instance_set_name, policy_name)
        df.to_excel(excel_name, index=False)
        print('date set: {} finished'.format(instance_set_name))
