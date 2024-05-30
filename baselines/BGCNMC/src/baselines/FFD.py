
import sys
import time
import os
import pandas as pd
import multiprocessing
import shutil

import torch
# import yaml
import random
import argparse
import numpy as np

def FFD(capacity, lengths):
    bins_num = 0
    while 0 < np.sum(lengths):
        used_space, item_id = 0.0, 0
        while item_id < len(lengths):
            hold_space = used_space + lengths[item_id]
            if hold_space > capacity:
                item_id += 1
                continue
            else:
                used_space += lengths[item_id]
                lengths = np.delete(arr=lengths,obj=item_id,axis=0)
        bins_num += 1
    return bins_num

def get_filelist(dir, Filelist):
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

def get_instance(instance_path):
    lengths = []
    with open(instance_path, 'r', encoding='utf-8') as f:
        item_number = int(f.readline())
        capacity = int(f.readline())
        for id in range(item_number):
            order = f.readline().split()
            for count in range(int(order[1])):
                lengths.append(float(order[0]))
    return capacity, np.array(lengths)

def collect(data_table, file_path, instance_name, opt_obj):
    capacity, lengths = get_instance(instance_path=file_path)

    ffd_bins = FFD(capacity=capacity, lengths=lengths)
    results = [instance_name, ffd_bins, float(ffd_bins-opt_obj)/opt_obj*100]
    data_table.append(results)

if __name__ == '__main__':

    if not os.path.exists('../data'):
        os.mkdir('../data')
    if os.path.exists('../data/generated_results'):
        shutil.rmtree('../data/generated_results')
    os.mkdir('../data/generated_results')

    base_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/instances/'
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
    '''
    instance_folder_list = [
        '',
        '',
    ]
    instance_set_name_list = [
        'Waescher_CSP',
        'Hard28_CSP',
    ]
    '''
    opt_results = pd.read_excel('../../data/baselines/opt_baselines.xlsx')


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
        print('date set: {} is collecting'.format(instance_set_name))
        task_pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
        data_table = multiprocessing.Manager().list()

        instances_path = base_folder + instance_folder + instance_set_name

        instances_path_list = get_filelist(dir=instances_path, Filelist=[])

        instance_number = -1

        for instance_path in instances_path_list:
            instance_name = os.path.basename(instance_path)
            if '.DS_Store' == instance_name or 'list.txt' == instance_name:
                continue
            instance_number += 1

            opt_result = opt_results[opt_results['Name'] == instance_name]['Best LB']
            opt = opt_result.iloc[0]
            # collect(data_table, instance_path, instance_name, opt)
            task_pool.apply_async(collect, (data_table, instance_path, instance_name, opt))

        task_pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        task_pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        columns = ['Name', 'ffd_obj', 'gap']

        df = pd.DataFrame(list(data_table), columns=columns)
        excel_name = "../../data/deterministic/FFD_baselines_on_{}.xlsx".format(instance_set_name)
        df.to_excel(excel_name, index=False)
        meangapdf.loc[0,instance_set_name] = df['gap'].mean()
        print('date set: {} finished'.format(instance_set_name))
    meangapexcel_name = "../../data/deterministic/FFD_meangap.xlsx"
    meangapdf.to_excel(meangapexcel_name, index=False)
