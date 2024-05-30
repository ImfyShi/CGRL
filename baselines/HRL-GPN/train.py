import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpn import GPN
import copy
import os
import parameters.read_instance as read_instance
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import shutil

if __name__ == "__main__":
    writer = SummaryWriter('./training_log/')

    # args
    parser = argparse.ArgumentParser(description="GPN with RL")
    parser.add_argument('--size', default=50, help="size of TSP")
    parser.add_argument('--epoch', default=1000000000, help="number of epochs")
    parser.add_argument('--batch_size', default=512, help='')
    parser.add_argument('--train_size', default=2500, help='')
    parser.add_argument('--val_size', default=1000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    args = vars(parser.parse_args())

    size = int(args['size'])
    learn_rate = args['lr']    # learning rate
    B = int(args['batch_size'])    # batch_size
    B_val = int(args['val_size'])    # validation size
    steps = int(args['train_size'])    # training steps
    n_epoch = int(args['epoch'])    # epochs
    save_root ='./model/gpn_csp.pt'
    
    print('=========================')
    print('prepare to train')
    print('=========================')
    print('Hyperparameters:')
    print('size', size)
    print('learning rate', learn_rate)
    print('batch size', B)
    print('validation size', B_val)
    print('steps', steps)
    print('epoch', n_epoch)
    print('save root:', save_root)
    print('=========================')

    if os.path.exists('./training_log'):
        shutil.rmtree('./training_log')
    os.mkdir('./training_log')
    
    model = GPN(n_feature=1, n_hidden=128)
    # load model
    # model = torch.load(save_root)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    lr_decay_step = 2500
    lr_decay_rate = 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                         lr_decay_step), gamma=lr_decay_rate)

    C = 0     # baseline
    R = 0     # reward

    instance_path_list = read_instance.get_filelist(dir="/Users/sfy/Store/Benchmark/CuttingStockProblem/original_instances/train_set/", Filelist=[])
    instance_path_list.sort()
    instance_list, instance_name_list, opt_baseline_list = list(), list(), list()
    B = len(instance_path_list)

    opt_baseline_table = pd.read_excel(r'./data/opt_baselines.xlsx')

    for instance_path in instance_path_list:
        if '.DS_Store' == os.path.basename(instance_path):
            continue

        instance_name = os.path.basename(instance_path)
        instance = read_instance.read_instance_file(instance_path=instance_path)
        instance_list.append(instance)
        instance_name_list.append(instance_name)

        opt_baseline_data = opt_baseline_table.loc[opt_baseline_table['Name'] == instance_name]
        opt_obj = opt_baseline_data['Best LB'].iloc[0]
        opt_baseline_list.append(opt_obj)
    all_opt_baseline_tensor = torch.tensor(opt_baseline_list)

    iter_num = -1

    group_num = int(len(instance_list) / B) if (0 == len(instance_list) % B) else len(instance_list) // B + 1
    pbar = tqdm(range(n_epoch))
    for epoch in pbar:
        for batch_idx in range(group_num):
            iter_num += 1
            if batch_idx < group_num - 1:
                instance_batch = instance_list[batch_idx * B: (batch_idx + 1) * B]
                train_instance_name_list = instance_name_list[batch_idx * B: (batch_idx + 1) * B]
                opt_baseline_tensor = all_opt_baseline_tensor[batch_idx * B: (batch_idx + 1) * B]
            else:
                instance_batch = instance_list[batch_idx * B:]
                train_instance_name_list = instance_name_list[batch_idx * B:]
                opt_baseline_tensor = all_opt_baseline_tensor[batch_idx * B:]

            train_data = read_instance.obtain_instance_data(
                instance_list=instance_batch)
            size = len(train_data[0])
            current_batch_size = len(train_data)

            optimizer.zero_grad()

            data = copy.deepcopy(train_data)
            data = torch.Tensor(data)
            X = copy.deepcopy(data)

            # mask = torch.zeros(current_batch_size, size)
            mask = torch.where(X > 0.0, torch.tensor(0.0), torch.tensor(-np.inf)).squeeze()

            logprobs = 0
            reward = 0
            
            Y = copy.deepcopy(X)
            x = Y[:,0,:]
            h = None
            c = None

            width_sequence = None
            for k in range(size):
                
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
                # idx = torch.argmax(output, dim=1)    # greedy baseline

                x = Y[[i for i in range(B)], idx.data].clone()
                if width_sequence is None:
                    width_sequence = x
                else:
                    width_sequence = np.c_[width_sequence,x]

                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY) 
                
                mask[[i for i in range(B)], idx.data] += -np.inf
                mask[torch.nonzero(torch.any(mask==0, dim=1)==0).squeeze(), -1] = 0

            # TODO
            # reward
            R = torch.zeros(width_sequence.shape[0])
            for row_idx in range(len(width_sequence)):
                sum_len = 0
                ele_idx = 0
                while ele_idx < width_sequence[row_idx].shape[0]:
                    if sum_len + width_sequence[row_idx, ele_idx] > 1:
                        R[row_idx] += 1
                        sum_len = 0
                    else:
                        sum_len += width_sequence[row_idx, ele_idx]
                        ele_idx += 1
            batch_obj_gap = (R.numpy()-opt_baseline_tensor.numpy())/opt_baseline_tensor.numpy()*(100)
            writer.add_scalar('train/batch gap (%)', batch_obj_gap.mean(), iter_num)
            pbar.set_description('EP: {} | batch idx: {}/{} | batch gap: {:.2f}'.format(epoch, batch_idx+1, group_num, np.mean(batch_obj_gap)))
            
            # self-critic base line
            # mask = torch.zeros(B,size)
            mask = torch.where(X > 0.0, torch.tensor(0.0), torch.tensor(-np.inf)).squeeze()
            
            C = 0
            baseline = 0
            
            Y = X.view(B,size,1)
            x = Y[:,0,:]
            h = None
            c = None
            
            for k in range(size):

                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)

                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
                # idx = torch.argmax(output, dim=1)    # greedy baseline

                x = Y[[i for i in range(B)], idx.data].clone()
                if width_sequence is None:
                    width_sequence = x[:,np.newaxis]
                else:
                    width_sequence = np.c_[width_sequence,x]

                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY)

                mask[[i for i in range(B)], idx.data] += -np.inf
                mask[torch.nonzero(torch.any(mask==0, dim=1)==0).squeeze(), -1] = 0


            # reward
            C = torch.zeros(width_sequence.shape[0])
            for row_idx in range(len(width_sequence)):
                sum_len = 0
                for ele in width_sequence[row_idx]:
                    if sum_len + ele > 1:
                        C[row_idx] += 1
                        sum_len = 0
                    else:
                        sum_len += ele

            gap = (R-C).mean()
            loss = ((R-C-gap)*logprobs).mean()

            writer.add_scalar('train/loss', loss, iter_num)
        
            loss.backward()
            
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()

            '''
            if i % 50 == 0:
                print("epoch:{}, batch:{}/{}, reward:{}"
                    .format(epoch, i, steps, R.mean().item()))
                # R_mean.append(R.mean().item())
                # R_std.append(R.std().item())
                
                # greedy validation
                
                tour_len = 0

                X = X_val
                X = torch.Tensor(X)
                
                mask = torch.zeros(B_val,size)
                
                R = 0
                logprobs = 0
                Idx = []
                reward = 0
                
                Y = X.view(B_val, size, 2)    # to the same batch size
                x = Y[:,0,:]
                h = None
                c = None
                
                for k in range(size):
                    
                    output, h, c, hidden_u = model(x=x, X_all=X, h=h, c=c, mask=mask)
                    
                    sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)
                    Idx.append(idx.data)
                
                    Y1 = Y[[i for i in range(B_val)], idx.data]
                    
                    if k == 0:
                        Y_ini = Y1.clone()
                    if k > 0:
                        reward = torch.norm(Y1-Y0, dim=1)
            
                    Y0 = Y1.clone()
                    x = Y[[i for i in range(B_val)], idx.data]
                    
                    R += reward
                    
                    mask[[i for i in range(B_val)], idx.data] += -np.inf
            
                R += torch.norm(Y1-Y_ini, dim=1)
                tour_len += R.mean().item()
                print('validation tour length:', tour_len)

            '''
        # print('save model to: ', save_root)
        torch.save(model, save_root)
