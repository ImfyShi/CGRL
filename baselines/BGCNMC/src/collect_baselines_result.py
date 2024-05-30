import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
import shutil
import numpy as np
import yaml

from baselines.eval_policy import Evaluate as eval_policy
from baselines import policy
import utilities
from utilities import construct_edge_feature

if __name__ == '__main__':
    utilities.fix_seed()

    with open('eval_config.yaml', encoding='utf-8') as f:
        parameters = yaml.load(f.read(), Loader=yaml.FullLoader)

    if not os.path.exists('../data'):
        os.mkdir('../data')
    if os.path.exists('../data/generated_results'):
        shutil.rmtree('../data/generated_results')
    os.mkdir('../data/generated_results')


    dataset_log_path = '../data/generated_results/dataset.log'
    file_obj = open(dataset_log_path, 'w+')
    file_obj.close()

    policy_name = parameters['policy_name']

    base_folder = parameters['base_folder']
    write_folder = parameters['write_folder']

    instance_folder_list = parameters['instance_sets_path'][parameters['instance_set_group']]['instance_folder_list']
    instance_set_name_list = parameters['instance_sets_path'][parameters['instance_set_group']]['instance_set_name_list']

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
            file_object.writelines('dealing instance set: {}'.format(instance_set_name) + '\n')

        instances_set_folder = base_folder + instance_folder + instance_set_name
        make_folder_path = write_folder + instance_folder + instance_set_name

        env_list, env_num = utilities.load_env_list(instance_folder=instances_set_folder, write_instance_data_folder=make_folder_path, is_collect_root_info=False, initialed_log_path=initialed_log_path, initialing_log_path=initialing_log_path, baselines_result=ffd_result, opt_results=opt_result)

        print('{} set {} is collecting'.format(env_num, instance_set_name))
        task_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        data_table = multiprocessing.Manager().list()

        pbar = tqdm(range(len(env_list)))
        for env_idx in pbar:
            env = env_list[env_idx]
            env.load_minknapdll()
            pbar.set_description('{}: {} '.format(instance_set_name, env.instance_name))
            # task_pool.apply_async(eval_policy, args=(data_table, env, policy.pure_diving, construct_edge_feature, collected_log_path, collecting_log_path,))
            # task_pool.apply_async(eval_policy, args=(data_table, env, policy.trim_loss_greedy, construct_edge_feature, collected_log_path, collecting_log_path,))

        task_pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        task_pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用


        columns = ['name', 'obj', 'gap', 'sp_obj', 'ffd_gap']

        df = pd.DataFrame(list(data_table), columns=columns)
        excel_name = "../data/generated_results/{}_on_{}.xlsx".format(instance_set_name, policy_name)
        df.to_excel(excel_name, index=False)
        print('date set: {} finished'.format(instance_set_name))
