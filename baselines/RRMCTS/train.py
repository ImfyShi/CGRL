import time

import numpy as np
from tqdm import tqdm
from Coach import Coach
# from morpionsolitaire.MorpionSolitaireGame import MorpionSolitaireGame as Game
# from morpionsolitaire.pytorch.NNet import NNetWrapper as nn
from CSP.CSPGame import CSPGame as Game
from CSP.pytorch.NNet import NNetWrapper as nn
import CSP.data.instance.read_instance as read_instance
from utils import *
import os

def train_on_training_set():
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
        'load_model': False,
        'load_folder_file': ('./CSP/data/model/','best.pth.tar'),
        'numItersForTrainExamplesHistory': 10,
        'maxmum items number' : 5486,
        'feature_dim' : 1
    })

    instance_path_list = read_instance.get_filelist(dir="/Users/sfy/Store/Benchmark/CuttingStockProblem/original_instances/train_set/", Filelist=[])
    instance_path_list.sort()
    instance_list, instance_name_list = list(), list()

    for instance_path in instance_path_list:
        if '.DS_Store' == os.path.basename(instance_path):
            continue

        instance_name = os.path.basename(instance_path)
        instance = read_instance.read_instance_file(file_path=instance_path, maxmum_items_number=args['maxmum items number'])
        instance_list.append(instance)
        instance_name_list.append(instance_name)

    start_time = time.time()

    for epoch in range(args.epochNum):
        epoch_start_time = time.time()
        nnet = None
        instance_bar = tqdm(range(len(instance_list)), desc='training epoches')
        for instance_idx in instance_bar:
            instance_bar.set_description('Training Epoches {} | Instance: {} | Training Time: {}s | Epoch Time: {}s'.format(epoch, instance_idx, round(time.time()-start_time, 2), round(time.time()-epoch_start_time, 2)))
            g = Game(instance_data=instance_list[instance_idx], maxmum_items_number=args['maxmum items number'])
            if nnet is None:
                nnet = nn(g, args['feature_dim'])

            c = Coach(g, nnet, args)
            c.learn()
