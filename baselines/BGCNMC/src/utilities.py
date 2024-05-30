import copy
import gzip
import pickle
import datetime
import numpy as np
import os
import random
import multiprocessing

import torch
import torch.nn.functional as F
import torch_geometric

import environment.Class_Diving as env

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = np.array(pad_sizes).max()
    output = input_.split(pad_sizes)
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, item_features, edge_indices, edge_features, column_features, rewards, actions):
        super().__init__()
        self.item_features, self.edge_index, self.edge_features, self.column_features, self.rewards, self.actions = item_features, edge_indices, edge_features, column_features, rewards, actions

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.item_features.size(0)], [self.column_features.size(0)]])
        # elif key == 'column_features':
            # return self.column_features.size(0)
        else:
            return super(BipartiteNodeData, self).__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, item_features, edge_features, edge_indeices, column_features, actions, rewards, column_nums, buffer_item_num):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.item_features, self.edge_features, self.edge_indeices, self.column_features, self.actions, self.rewards, self.column_nums, self.item_nums = item_features, edge_features, edge_indeices, column_features, actions, rewards, column_nums, buffer_item_num

    def len(self):
        return len(self.actions)

    def get(self, index):
        item_features = torch.FloatTensor(self.item_features[index])
        edge_indices = torch.LongTensor(np.array(self.edge_indeices[index], dtype=np.int32))
        edge_features = torch.FloatTensor(self.edge_features[index])
        column_features = torch.FloatTensor(self.column_features[index])
        rewards = torch.FloatTensor([self.rewards[index]])
        actions = torch.LongTensor([self.actions[index]])

        graph = BipartiteNodeData(item_features, edge_indices, edge_features, column_features, rewards, actions)
        graph.num_nodes = item_features.shape[0]+column_features.shape[0]
        return graph

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

def fix_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_env(env_list, instance_path, baseline, opt, initialed_log_path=None, initialing_log_path=None, write_instance_data_folder=None, is_collect_root_info=False):

    if initialed_log_path is not None:
        with open(initialing_log_path, 'a+') as file_object:
            file_object.writelines('{}'.format(os.path.basename(instance_path)) + '\n')
    instance_name = os.path.basename(instance_path)
    instance_name = instance_name.split('.txt', 1 )[0]
    write_instance_data_folder = write_instance_data_folder+'/'+instance_name
    if is_collect_root_info:
        os.makedirs(write_instance_data_folder,exist_ok=True)
    env_list.append(env.Diving(instance_path=instance_path, baseline=baseline, opt=opt, write_instance_data_folder=write_instance_data_folder, is_collect_root_info=is_collect_root_info))
    # env_list.append(env.Diving(instance_path=instance_path, baseline=baseline, opt=opt, write_instance_data_folder=write_instance_data_folder, is_collect_root_info=is_collect_root_info))

    if initialed_log_path is not None:
        with open(initialed_log_path, 'a+') as file_object:
            file_object.writelines('{}'.format(os.path.basename(instance_path)) + '\n')


def load_env_list(instance_folder, initialed_log_path, initialing_log_path, baselines_result=None, opt_results=None, write_instance_data_folder=None, is_collect_root_info=False):
    load_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    env_list = multiprocessing.Manager().list()
    instances_path_list = get_filelist(dir=instance_folder, Filelist=[])
    # instances_path_list = ['/home/sfy/Benchmark/CuttingStockProblem/paper_used_instances/AI/201_2500_DI_0.txt']
    for instance_path in instances_path_list:
        instance_name = os.path.basename(instance_path)
        if '.DS_Store' == instance_name or 'list.txt' == instance_name:
            continue
        baseline=-1; opt=-1
        if baselines_result is not None:
            base_result = baselines_result[baselines_result['Name'] == instance_name]['obj']
            baseline = base_result.iloc[0]
        if opt_results is not None:
            opt_result = opt_results[opt_results['Name'] == instance_name]['Best LB']
            opt = opt_result.iloc[0]
        # load_pool.apply_async(load_env, (env_list, instance_path, baseline, opt, initialed_log_path, initialing_log_path, write_instance_data_folder, is_collect_root_info))
        load_env(env_list, instance_path, baseline, opt, initialed_log_path, initialing_log_path, write_instance_data_folder, is_collect_root_info)

    load_pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    load_pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
    return env_list, len(env_list)

def construct_edge_feature(items, edges, column_features, eval_baselines=False):
    original_edges, original_column_features = copy.deepcopy(edges), copy.deepcopy(column_features)
    # 去掉需求已经被满足的item并构造新的item features
    item_features_list = items
    if not eval_baselines:
        deleted_item_indices = list()
        itemnum = len(items)
        for item_id in range(len(items)):
            # print('iternum={}, itemid={}'.format(itemnum,item_id))
            if 0.5 > items[item_id][1]:
                deleted_item_indices.append(item_id)
                for col_id in range(edges.shape[1]):
                    # 计算列中对满足需求没有帮助的空间大小
                    column_features[col_id][1] += edges[item_id,col_id]*items[item_id][0]
                    edges[item_id, col_id] = 0
        item_features_list = [i for num,i in enumerate(items) if num not in deleted_item_indices]
    column_features[:, 1] = -column_features[:, 1]
    #
    edge_indices, edge_features, column_decision_list, column_features_list = list(), list(), list(), list()
    column_number = -1
    colnum = edges.shape[1]
    # 原始的列遍历一遍
    for col_id in range(edges.shape[1]):
        # print('colnum={}, colid={}'.format(colnum, col_id))
        if (0 >= column_features[col_id][0]):
            continue
        if (0 < column_features[col_id][0]) and (1 >= column_features[col_id][0]):
            # ceil
            column_number += 1
            column_features_list.append([column_features[col_id][0],column_features[col_id][1], column_features[col_id][2],column_features[col_id][3]])
            column_decision_list.append([original_edges[:,col_id],original_column_features[col_id,0],1])
            item_number = -1
            for item_id in range(edges.shape[0]):
                if 0 < edges[item_id,col_id]:
                    item_number += 1
                    edge_indices.append([item_number,column_number])
                    edge_features.append([edges[item_id,col_id]])
        elif 1 < column_features[col_id][0]:
            # floor
            if (column_features[col_id][0]-np.floor(column_features[col_id][0])) < 0.5:
                column_number += 1
                column_features_list.append([column_features[col_id][0],column_features[col_id][1], column_features[col_id][2],column_features[col_id][3]])
                column_decision_list.append([original_edges[:,col_id],original_column_features[col_id,0],0])
                item_number = -1
                for item_id in range(edges.shape[0]):
                    if 0 < edges[item_id,col_id]:
                        item_number += 1
                        edge_indices.append([item_number,column_number])
                        edge_features.append([edges[item_id,col_id]])
            # ceil
            if (column_features[col_id][0]-np.floor(column_features[col_id][0])) >= 0.5:
                column_number += 1
                column_features_list.append([column_features[col_id][0],column_features[col_id][1], column_features[col_id][2],column_features[col_id][3]])
                column_decision_list.append([original_edges[:,col_id],original_column_features[col_id,0],1])
                item_number = -1
                for item_id in range(edges.shape[0]):
                    if 0 < edges[item_id,col_id]:
                        item_number += 1
                        edge_indices.append([item_number,column_number])
                        edge_features.append([edges[item_id,col_id]])
    edge_indices = list(map(list, zip(*edge_indices)))
    return np.array(item_features_list), edge_indices, np.array(edge_features), np.array(column_features_list), column_decision_list

def cut_action_space(item_features, edges, column_features, history_action_pool, is_monotonicity_cut=False):
    frac_rule = column_features[:,0]>0
    # demand_rule = 0.5 < np.inner(edges.T, np.maximum(np.array(item_features), 0)[:, 1])
    # print('frac filter num = {} | demand filter num = {}'.format(np.sum(frac_rule), np.sum(demand_rule)))
    # extract_colunm_rule = frac_rule & demand_rule
    extract_colunm_rule = frac_rule
    extracted_column_features = column_features[extract_colunm_rule, :]
    extracted_edges = edges[:, extract_colunm_rule]

    if history_action_pool is not None:
        fracs = extracted_column_features[:, 0]
        fracs = np.where(fracs<=1, 1, fracs)
        fracs = np.around(fracs)
        cut_patterns = copy.deepcopy(extracted_edges).T
        action_space = np.insert(cut_patterns, 0, fracs, 1)
        diff = action_space[:, np.newaxis] - history_action_pool
        neg_diff = np.where(diff < 0, -1e10, diff)
        normed_diff = np.where(neg_diff > 0, 1, neg_diff)
        sum_normed_diff = np.sum(normed_diff, axis=2)
        ignore_neg_val_sum_normed_diff = np.where(sum_normed_diff < 0, 0, sum_normed_diff)
        sum_ignore_neg_val_sum_normed_diff = np.sum(ignore_neg_val_sum_normed_diff, axis=1)
        labeled_removed_column = sum_ignore_neg_val_sum_normed_diff > 0
        labeled_remained_column = ~labeled_removed_column
        if 0 != labeled_remained_column.sum():
            extracted_column_features = extracted_column_features[labeled_remained_column, :]
            extracted_edges = extracted_edges[:, labeled_remained_column]
    return item_features, extracted_edges, extracted_column_features


def parallel_rollout(env, agent, data_table):
    buffer_item, buffer_edge, buffer_edge_indices, buffer_column, buffer_a, buffer_r, buffer_column_num, buffer_item_num, ep_r, ep_rwr = Rollout(env=env, agent=agent)
    data_table.append([buffer_item, buffer_edge, buffer_edge_indices, buffer_column, buffer_a, buffer_r, buffer_column_num, buffer_item_num, ep_r, ep_rwr, env.baseline, env.opt])

def Rollout(env, agent, device, is_stochastic):
    item_features, edges, column_features = env.reset()
    buffer_item_features, buffer_edge_features, buffer_edge_indices, buffer_column_features, buffer_actions, buffer_reward, buffer_column_num, buffer_item_num = list(), list(), list(), list(), list(), list(), list(), list()
    done, ep_r, ep_rwr = False, 0, 0
    debug_count = -1
    history_action_pool = None
    while not done:  # in one episode
        debug_count += 1
        if 25==debug_count:
            asdf = 23423
        item_features, edges, column_features = cut_action_space(item_features=item_features, edges=edges, column_features=column_features, history_action_pool=history_action_pool)
        if 0 == len(column_features):
            adfasdf = 234234
        item_features, edge_indices, edge_features, column_features, column_decision_list = construct_edge_feature(items=item_features, edges=edges, column_features=column_features, eval_baselines=False)
        if 0 == len(edge_indices):
            adfasdf = 234234
        #print('column_decision_list num={}'.format(len(column_decision_list)))
        if 0 == len(edge_indices):
            adfasdf = 234234
        action = agent.choose_action(torch.FloatTensor(item_features).to(device), edge_features=torch.FloatTensor(edge_features).to(device), edge_indices=torch.LongTensor(edge_indices).to(device), column_features=torch.FloatTensor(column_features).to(device), is_stochastic=is_stochastic)
        implement_action = column_decision_list[action]
        if history_action_pool is None:
            history_action_pool = np.insert(implement_action[0], 0, implement_action[1])[np.newaxis, :]
        else:
            history_action_pool = np.insert(arr=history_action_pool, obj=0, values=np.insert(arr=implement_action[0], obj=0, values=implement_action[1]), axis=0)
        item_features_, edges_, column_features_, reward, done = env.step(cut_pattern_list=[implement_action[0]], fra_solu_list=[implement_action[1]], flag_list=[implement_action[2]])

        norm_reward = -reward 

        buffer_item_features.append(item_features); buffer_edge_features.append(edge_features); buffer_edge_indices.append(edge_indices); buffer_column_features.append(column_features); buffer_actions.append([action]); buffer_reward.append([norm_reward]); buffer_column_num.append(len(column_features)); buffer_item_num.append(len(item_features))

        ep_r += norm_reward; ep_rwr += reward

        item_features = item_features_; edges = edges_; column_features = column_features_;

    mip_select_obj = env.set_partitioning_problem(column=env.history_column_pool)

    v_s_ = 0
    discounted_r = []  # compute discounted reward
    for r in buffer_reward[::-1]:
        v_s_ = r[0] + v_s_
        discounted_r.append([v_s_])
    discounted_r.reverse()

    # buffer_item, buffer_edge, buffer_edge_index, buffer_column, buffer_a, buffer_r, buffer_column_num, ep_r, ep_rwr
    return buffer_item_features, buffer_edge_features, buffer_edge_indices, buffer_column_features, buffer_actions, discounted_r, buffer_column_num, buffer_item_num, ep_r, ep_rwr, mip_select_obj

if __name__ == '__main__':
    Rollout(env=0, agent=0)