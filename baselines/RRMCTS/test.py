import time
import numpy as np
from Coach import Coach
# from morpionsolitaire.MorpionSolitaireGame import MorpionSolitaireGame as Game
# from morpionsolitaire.pytorch.NNet import NNetWrapper as nn
from CSP.CSPGame import CSPGame as Game
from CSP.pytorch.NNet import NNetWrapper as nn
import CSP.data.instance.read_instance as read_instance
import CSP.CSPPlayers as CSPPlayers
import Arena
from utils import *
import os
import pandas as pd
from tqdm import tqdm
import shutil

def eval_on_test_instances():

    opt_baseline_table = pd.read_excel(r'./CSP/data/opt_baselines.xlsx')

    args = dotdict({
        'epochNum': 100,
        'numIters':1, #100, #1000,
        'numEps': 1, #10, #100,
        'tempThreshold': 41,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 1,
        'arenaCompare': 1, #40,
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': True,
        'load_folder_file': ('./CSP/data/model/','best.pth.tar'),
        'numItersForTrainExamplesHistory': 10,
        'maxmum items number' : 5486,
        'feature_dim' : 1
    })

    if os.path.exists('./CSP/data/generated_results'):
        shutil.rmtree('./CSP/data/generated_results')
    os.mkdir('./CSP/data/generated_results')

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

    total_start_time = time.time()

    for folder, set_name in zip(instance_folder_list, instance_set_name_list):
        instance_set_path = base_folder + folder + set_name

        instance_path_list = read_instance.get_filelist(dir=instance_set_path, Filelist=[])
        instance_path_list.sort()
        instance_list, instance_name_list = list(), list()

        for instance_path in instance_path_list:
            if '.DS_Store' == os.path.basename(instance_path):
                continue

            instance_name = os.path.basename(instance_path)
            instance = read_instance.read_instance_file(file_path=instance_path, maxmum_items_number=args['maxmum items number'])
            instance_list.append(instance)
            instance_name_list.append(instance_name)

        table_data = list()

        nnet = None
        set_start_time = time.time()
        pbar = tqdm(range(len(instance_list)))
        for instance_idx in pbar:

            instance_name = instance_name_list[instance_idx]
            opt_baseline_data = opt_baseline_table.loc[opt_baseline_table['Name'] == instance_name]
            opt_obj = opt_baseline_data['Best LB'].iloc[0]

            instance_start_time = time.time()
            g = Game(instance_data=instance_list[instance_idx], maxmum_items_number=args['maxmum items number'])
            if nnet is None:
                nnet = nn(g, args['feature_dim'])
                if args.load_model:
                    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
            player = CSPPlayers.NNtBasedMCTSPlayer(g, nnet, args).play
            arena = Arena.Arena(player, g)
            obj = -arena.playGame()

            pbar.set_description('Testing Set {} | Acomplished Instance Idx: {} | Total Testing Time: {}s | Set Time: {}s'.format(set_name, instance_idx, round(time.time()-total_start_time, 2), round(time.time()-set_start_time, 2)))

            table_data.append([instance_name, obj, (obj-opt_obj)/opt_obj*100])

            alg_result = pd.DataFrame(table_data, columns=['name', 'obj', 'gap'])
            alg_result.to_excel('./CSP/data/generated_results/RRMCTS_result_on_{}.xlsx'.format(set_name),merge_cells=True, float_format='%.3f', encoding='utf-8')
