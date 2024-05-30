from tqdm import tqdm
import os
import numpy as np


def read_instance_file(file_path=None):
    instance_path = file_path
    demand_width = list()
    with open(instance_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        count_line = int(line) + 1
        for id in range(count_line):
            if id == 0:
                roll_width = np.array(int(f.readline()))
            else:
                line = f.readline()
                line = line.rstrip("\n")
                a = line.split("\t")
                b = int(a[0])
                c = int(a[1])
                for _ in range(c):
                    demand_width.append(b / roll_width)
    return demand_width

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

if __name__ == '__main__':

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

    max_item_number, max_instance_name = 0, None
    for folder, set_name in zip(instance_folder_list, instance_set_name_list):
        instance_set_path = base_folder + folder + set_name

        instance_path_list = get_filelist(dir=instance_set_path, Filelist=[])
        instance_path_list.sort()
        B = 1
        dataset_max_item_number, dataset_max_instance_name = 0, None

        pbar = tqdm(range(len(instance_path_list)))
        for instance_path_idx in pbar:
            instance_path = instance_path_list[instance_path_idx]
            if '.DS_Store' == os.path.basename(instance_path):
                continue
            instance = read_instance_file(file_path=instance_path)
            if max_item_number < len(instance):
                max_item_number = len(instance)
                max_instance_name = os.path.basename(instance_path)
            if dataset_max_item_number < len(instance):
                dataset_max_item_number = len(instance)
                dataset_max_instance_name = os.path.basename(instance_path)
            pbar.set_description('{} | dataset max item number = {} | dataset max instance = {} | global max item number = {}'.format(set_name, dataset_max_item_number, dataset_max_instance_name, max_item_number))
    print('max item number = {} | max instance name = {}'.format(max_item_number, max_instance_name))
