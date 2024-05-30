import os
import utilities

base_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/instances/'
write_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/tobetrained_instances/instances/minknaproot/'

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
baselines_result = None
opt_result = None

for instance_folder, instance_set_name in zip(instance_folder_list, instance_set_name_list):
    print('set {} roots are collecting'.format(instance_set_name))
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

    original_instance_folder = base_folder + instance_folder + instance_set_name
    make_folder_path = write_folder + instance_folder + instance_set_name
    env_list, env_num = utilities.load_env_list(instance_folder=original_instance_folder,
                                                write_instance_data_folder=make_folder_path,
                                                is_collect_root_info=True, initialed_log_path=initialed_log_path,
                                                initialing_log_path=initialing_log_path, baselines_result=None,
                                                opt_results=None)
    print('set {} roots are collected finished'.format(instance_set_name))
