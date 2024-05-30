# RRH (algorithm 2) of A residual recombination heuristic for one‑dimensional cutting stock problems, implemented by Fengyuan.Shi

import numpy as np
from numpy import *
import copy
import os
import ctypes
from tqdm import tqdm
import gurobipy as grb
from gurobipy import GRB
import math
import pandas as pd

class ColumnGeneration:

    def __init__(self, demand_width_array, demand_number_array, roll_width):
        self.demand_width_array = demand_width_array 
        self.demand_number_array = demand_number_array 
        self.roll_width = roll_width
        self.item_num = len(demand_width_array)
        self.trim_loss = None

        # self.minknap_dll_path = './environment/minknap.so'
        self.minknap_dll_path = './minknap.so'

        self.load_minknapdll()

    def load_minknapdll(self):
        self.minknap_dll = ctypes.CDLL(self.minknap_dll_path)
        self.minknap_dll.minknap.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.minknap_dll.minknap.restype = ctypes.c_double

    def master_problem(self, column, vtype):
        m = grb.Model()
        m.setParam('OutputFlag', 0)  # 不显示信息
        x = m.addMVar(shape=column.shape[1], lb=0, vtype=vtype)
        m.addConstr(lhs=column @ x >= self.demand_number_array)
        m.setObjective(x.sum(), GRB.MINIMIZE)
        m.optimize()
        # print(f'lowerbound: {m.objVal}')
        if vtype == GRB.CONTINUOUS:
            if m.status == GRB.INFEASIBLE:
                asdf = 234234
            return np.array(m.getAttr('Pi', m.getConstrs())), np.array(m.getAttr('X'))
        else:
            return m.objVal, np.array(m.getAttr('X'))

    def restricted_lp_master_problem(self, column):
        return self.master_problem(column, GRB.CONTINUOUS)

    def solve_r1dcsp_by_CG(self):
        self.buffer_item = [list(item) for item in
                                zip(self.demand_width_array, self.demand_number_array)]  # 生成item的特征向量返回给RL
        count = len(self.demand_width_array)  # item的数量
        self.feasible_solution = np.empty(shape=[0, count])  # Diving过程中最终可行解存放
        #        initial_cut_pattern = np.diag(np.floor(self.roll_width / self.demand_width_array))  # 获得初始可行解
        initial_cut_pattern = self.Heuristic()  # 启发式获得初始解
        self.cut_pattern = copy.deepcopy(initial_cut_pattern)  # 给列池赋值
        self.history_column_pool = copy.deepcopy(initial_cut_pattern)
        self.reduce_cost = np.zeros(len(self.cut_pattern[0]))  # 初始列消减费用都是0
        # 获得初始列的cut loss
        cut_pattern_count = len(self.cut_pattern[0])
        for i in range(cut_pattern_count):
            length = np.multiply(np.array(self.demand_width_array), np.array(self.cut_pattern[:, i]))
            element_in_trim_loss = self.roll_width - sum(length)
            self.trim_loss = np.array([element_in_trim_loss]) if self.trim_loss is None else np.append(self.trim_loss, element_in_trim_loss)
        self.fra_solu = self.C_G()

        return copy.deepcopy(self.cut_pattern), copy.deepcopy(self.fra_solu)

    def Heuristic(self):  # best_fit
        #        initial_cut_pattern = np.diag(np.floor(self.roll_width / self.demand_width_array))  # 获得初始可行解
        temp_demand = copy.copy(self.demand_number_array)  # 临时存储需求量
        bin_capacity = list()  # 记录每个bin的剩余容量
        bin_count = 0
        for i in range(0, self.item_num):
            while temp_demand[i] > 0:
                best_bin_index = -1
                best_left = self.roll_width  # 最小剩余
                for k in range(0, bin_count):
                    temp_left = bin_capacity[k] - self.demand_width_array[i]
                    if (temp_left < best_left) and (temp_left >= 0):
                        best_left = temp_left
                        best_bin_index = k
                if best_bin_index == -1:  # 如果找不到合适的，增加一个新的bin
                    bin_capacity.append(self.roll_width)  # 增加一个bin的剩余容量
                    if bin_count == 0:
                        initial_cut_pattern = np.zeros(self.item_num, dtype=np.int32)
                    else:
                        result = np.zeros(self.item_num, dtype=np.int32)  # 增加一个bin切割模式
                        initial_cut_pattern = np.column_stack((initial_cut_pattern, result))
                    bin_count = bin_count + 1
                    best_bin_index = bin_count - 1
                # 接下来把item放入到bin中
                bin_capacity[best_bin_index] = bin_capacity[best_bin_index] - self.demand_width_array[i]  # 更新bin的剩余容量
                if bin_count == 1:
                    initial_cut_pattern[i] = initial_cut_pattern[i] + 1
                else:
                    initial_cut_pattern[i][best_bin_index] = initial_cut_pattern[i][best_bin_index] + 1  # 更新bin方案
                temp_demand[i] = temp_demand[i] - 1  # 更新item的需求
        initial_cut_pattern = np.unique(initial_cut_pattern, axis=1)
        return initial_cut_pattern

    # 列生成主函数
    def C_G(self):
        demand_item_number = int(np.maximum(self.demand_number_array, 0).sum())
        binary_item_number = int(demand_item_number) + np.count_nonzero(0 >= self.demand_number_array)
        minknap_solution = (ctypes.c_int * demand_item_number)(*np.zeros(demand_item_number, dtype=np.int32))
        binary_cutting_stock_solution_para_matrix = np.zeros((len(self.demand_number_array), binary_item_number))
        minknap_solution_para_matrix = np.zeros((binary_item_number, demand_item_number))
        binary_cutting_stock_idx, demand_item_idx = -1, -1
        for demand_idx in range(len(self.demand_number_array)):
            demand_idx = int(demand_idx)
            if 0==self.demand_number_array[demand_idx]:
                binary_cutting_stock_idx += 1
                continue
            for demand_number_idx in range(int(self.demand_number_array[demand_idx])):
                binary_cutting_stock_idx += 1; demand_item_idx += 1
                binary_cutting_stock_solution_para_matrix[demand_idx,binary_cutting_stock_idx] = 1
                minknap_solution_para_matrix[binary_cutting_stock_idx, demand_item_idx] = 1
        np_demand_length_array = np.inner(binary_cutting_stock_solution_para_matrix.T, self.demand_width_array)
        np_demand_length_array = np.inner(minknap_solution_para_matrix.T, np_demand_length_array)
        np_demand_length_array = np_demand_length_array.astype(np.int32)
        if not np_demand_length_array.flags['C_CONTIGUOUS']:
            np_demand_length_array = np.ascontiguous(np_demand_length_array, dtype=np_demand_length_array.dtype)
        c_demand_length_array = np_demand_length_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        if np.any(1 > np_demand_length_array):
            asdfasdfasdf = 12341234
        flag_new_cut_pattern = True  # 产生新列标记
        new_cut_pattern = None  # 新列
        while flag_new_cut_pattern:  # 进入循环
            if new_cut_pattern is not None:  # 如果有新列，则加入列池
                deleted_idx = np.where((self.cut_pattern.T == new_cut_pattern).all(1))[0]
                assert(0==len(deleted_idx))
                self.cut_pattern = np.column_stack((self.cut_pattern, new_cut_pattern))
                self.history_column_pool = np.column_stack((self.history_column_pool, new_cut_pattern))
            kk, fra_solu = self.restricted_lp_master_problem(self.cut_pattern)  # 带着列池，求解松主弛问题
            pi = np.inner(binary_cutting_stock_solution_para_matrix.T, kk)
            np_demand_pi = np.inner(minknap_solution_para_matrix.T, pi)
            np_demand_pi = np_demand_pi.astype(double)
            # print('np_demand_pi={}'.format(np_demand_pi))
            c_demand_pi = np_demand_pi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            assert ((len(np_demand_pi)==demand_item_number) and (len(np_demand_length_array)==demand_item_number))
            # print(np_demand_pi)
            # print(np_demand_length_array)
            flag_new_cut_pattern, new_cut_pattern = self.knapsack_subproblem(demand_item_number=demand_item_number, demand_pi=c_demand_pi, demand_items_length=c_demand_length_array, minknap_solution=minknap_solution, binary_cutting_stock_solution_para_matrix=binary_cutting_stock_solution_para_matrix, minknap_solution_para_matrix=minknap_solution_para_matrix)  # 求解子问题
        # 如果求到下界，返回当前非零解，及对应的列
        return fra_solu

    def knapsack_subproblem(self, demand_item_number, demand_pi, demand_items_length, minknap_solution, minknap_solution_para_matrix, binary_cutting_stock_solution_para_matrix):
        demand_item_number = int(np.maximum(self.demand_number_array, 0).sum())
        binary_item_number = int(demand_item_number) + np.count_nonzero(0 >= self.demand_number_array)
        minknap_solution = (ctypes.c_int * demand_item_number)(*np.zeros(demand_item_number, dtype=np.int32))
        reduce_cost = 1 - self.minknap_dll.minknap(ctypes.c_int(demand_item_number), demand_pi, demand_items_length, minknap_solution, ctypes.c_int(self.roll_width))
        flag_new_column = reduce_cost < -0.00001
        if flag_new_column:
            new_column = np.frombuffer(minknap_solution, dtype=np.int32)
            new_column = np.inner(minknap_solution_para_matrix, new_column)
            new_column = np.inner(binary_cutting_stock_solution_para_matrix, new_column)
            new_column = new_column.astype(np.int32)

            self.reduce_cost = np.append(self.reduce_cost, reduce_cost)  # reduce_cost 增加值
            length = np.multiply(np.array(self.demand_width_array), new_column)
            element_in_trim_loss = self.roll_width - sum(length)
            self.trim_loss = np.append(self.trim_loss, element_in_trim_loss)
        else:
            new_column = None

        return flag_new_column, new_column

def greedy_heuristic(d, l, L, minknap):
    demand_item_number = int(np.maximum(d, 0).sum())
    binary_item_number = int(demand_item_number) + np.count_nonzero(0 >= d)
    minknap_solution = (ctypes.c_int * demand_item_number)(*np.zeros(demand_item_number, dtype=np.int32))
    binary_cutting_stock_solution_para_matrix = np.zeros((len(d), binary_item_number))
    minknap_solution_para_matrix = np.zeros((binary_item_number, demand_item_number))
    binary_cutting_stock_idx, demand_item_idx = -1, -1
    for demand_idx in range(len(d)):
        demand_idx = int(demand_idx)
        if 0==d[demand_idx]:
            binary_cutting_stock_idx += 1
            continue
        for demand_number_idx in range(int(d[demand_idx])):
            binary_cutting_stock_idx += 1; demand_item_idx += 1
            binary_cutting_stock_solution_para_matrix[demand_idx,binary_cutting_stock_idx] = 1
            minknap_solution_para_matrix[binary_cutting_stock_idx, demand_item_idx] = 1
    np_demand_length_array = np.inner(binary_cutting_stock_solution_para_matrix.T, l)
    np_demand_length_array = np.inner(minknap_solution_para_matrix.T, np_demand_length_array)
    np_demand_length_array = np_demand_length_array.astype(np.int32)
    if not np_demand_length_array.flags['C_CONTIGUOUS']:
        np_demand_length_array = np.ascontiguous(np_demand_length_array, dtype=np_demand_length_array.dtype)
    c_demand_length_array = np_demand_length_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    kk = np.ones_like(d)
    pi = np.inner(binary_cutting_stock_solution_para_matrix.T, kk)
    np_demand_pi = np.inner(minknap_solution_para_matrix.T, pi)
    np_demand_pi = np_demand_pi.astype(double)
    # print('np_demand_pi={}'.format(np_demand_pi))
    c_demand_pi = np_demand_pi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    assert ((len(np_demand_pi)==demand_item_number) and (len(np_demand_length_array)==demand_item_number))
    minknap(ctypes.c_int(demand_item_number), c_demand_pi, c_demand_length_array, minknap_solution, ctypes.c_int(L))
    new_column = np.frombuffer(minknap_solution, dtype=np.int32)
    new_column = np.inner(minknap_solution_para_matrix, new_column)
    new_column = np.inner(binary_cutting_stock_solution_para_matrix, new_column)
    new_column = new_column.astype(np.int32)

    return new_column

def solve_r1dcsp_by_greedy_exhaustive_repetition_heuristic(l, d, L):
    minknap_dll = ctypes.CDLL('./minknap.so')
    minknap_dll.minknap.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    minknap_dll.minknap.restype = ctypes.c_double
    pattern_count = 0
    debug_flag = 0
    while(d.sum() >= 0):
        pattern = greedy_heuristic(d=d, l=l, L=L, minknap=minknap_dll.minknap)
        _pattern_count = 1e10
        for item_idx in range(len(d)):
            if pattern[item_idx]!=0:
                if _pattern_count > int(d[item_idx]/pattern[item_idx]):
                    _pattern_count = int(d[item_idx]/pattern[item_idx])
        d = d - pattern * _pattern_count
        pattern_count = pattern_count + _pattern_count
        if np.dot(d, l) <= L:
            pattern_count = pattern_count + 1
            break

    return pattern_count

def split_by_x(A, x):
    inte_A_list, inte_x_list, frac_A_list, frac_x_list = list(), list(), list(), list()
    for x_idx in range(len(x)):
        if x[x_idx] == int(x[x_idx]):
            inte_A_list.append(A[:, x_idx])
            inte_x_list.append(x[x_idx])
        else:
            frac_A_list.append(A[:, x_idx])
            frac_x_list.append(x[x_idx])

    return np.array(inte_A_list).T, np.array(inte_x_list), np.array(frac_A_list).T, np.array(frac_x_list)

def algorithm_1(A):
    B = np.zeros_like(A)
    for i in range(len(B)):
        q = 0
        p = len(A[0])-1
        while(q < p):
            z = math.ceil(A[i,q]) - A[i,q]
            if z <= A[i,p]:
                A[i,q] = math.ceil(A[i,q])
                A[i,p] = A[i,p] - z
                q = q + 1
            else:
                A[i,q] = A[i,q] + A[i,p]
                A[i,p] = math.floor(A[i,p])
                p = p - 1
        B[i] = A[i,:]
    return B

def is_nmcp(L, l, b, min_l):
    return L - np.dot(l, b) > min_l

def sort_by_x(A, x):
    sorted_idx = np.argsort(x)[::-1]
    sort_A  =A[:,sorted_idx]
    sort_x = x[sorted_idx]
    return sort_A, sort_x

def equation_6(A, x):
    return A*x

def solve_r1dcsp_by_CG(l, d, L):
    CG = ColumnGeneration(demand_width_array=l, demand_number_array=d, roll_width=L)
    A, x = CG.solve_r1dcsp_by_CG()
    return A, x

def algorithm_2(d, l, L):
    patterns = None
    
    while(np.sum(np.where(d > 0.5, 1, 0)) > 0):
        A, x = solve_r1dcsp_by_CG(l=l, d=d, L=L)
        inte_A, inte_x, frac_A, frac_x = split_by_x(A, x)
        if 0 == np.sum(np.where(d - np.sum(inte_A * inte_x, axis=1) > 0.5, 1, 0)):
        # if 0 >= (d - np.sum(inte_A * inte_x, axis=1)).sum():
            return inte_x.sum()
        frac_A, frac_x = sort_by_x(x=frac_x, A=frac_A)
        if patterns is None:
            patterns = inte_A
            patterns_used = inte_x
        tilde_A = equation_6(A=frac_A, x=frac_x)
        B = algorithm_1(A=tilde_A)
        y = np.zeros_like(B[0])
        for patterns_idx in range(len(B[0])):
            if is_nmcp(L=L, l=l, b=B[:,patterns_idx], min_l=l.min()):
                y[patterns_idx] = 0
        if 0 == y.sum():
            return solve_r1dcsp_by_greedy_exhaustive_repetition_heuristic(l=l, d=d, L=L)
        elif (y.sum() == len(y)):
            d = np.zeros_like(d)
        else:
            patterns = np.append(patterns, B, axis = 1)
            patterns_used = np.append(patterns_used, y)
            _patterns = np.append(inte_A, B, axis = 1)
            _patterns_used = np.append(inte_x, y)
            d = d - np.sum(_patterns * _patterns_used, axis=1)

    return patterns_used.sum()

def load_instance(instance_path):
    demand_number = list()
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
                demand_width.append(b)
                demand_number.append(c)
    return np.array(demand_width), np.array(demand_number), roll_width

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

def debug():
    minknap_dll = ctypes.CDLL('./minknap.so')
    minknap_dll.minknap.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    minknap_dll.minknap.restype = ctypes.c_double
    new_column = greedy_heuristic(d=np.array([1,1,0,0,0]), l=np.array([1,2,3,4,5]), L=9, minknap=minknap_dll.minknap)
    print(new_column)

if __name__ == '__main__':
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
    opt_results = pd.read_excel('opt_baselines.xlsx')

    for folder, set_name in zip(instance_folder_list, instance_set_name_list):
        data_table = list()
        instance_set_path = base_folder + folder + set_name
        instance_path_list = get_filelist(dir=instance_set_path, Filelist=[])
        pbar = tqdm(range(len(instance_path_list)))
        for instance_path_idx in pbar:
            instance_path = instance_path_list[instance_path_idx]
            instance_name = os.path.basename(instance_path)
            '''
            set_name = 'Randomly_Generated_CSP'
            instance_name = 'BPP_50_50_0.2_0.8_0.txt'
            instance_path = base_folder + set_name + '/' + instance_name
            '''
            pbar.set_description('{}-{}'.format(set_name, instance_name))
            if '.DS_Store' == instance_name:
                continue
            l, d, L = load_instance(instance_path)
            pattern_count = algorithm_2(d=d, l=l, L=L)
            # pattern_count = solve_r1dcsp_by_greedy_exhaustive_repetition_heuristic(d=d, l=l, L=L)

            opt_result = opt_results[opt_results['Name'] == instance_name]['Best LB']
            opt = opt_result.iloc[0]
            data_table.append([os.path.basename(instance_path), pattern_count, float((pattern_count-opt)/opt*100)])

        columns = ['name', 'obj', 'gap']

        df = pd.DataFrame(list(data_table), columns=columns)
        excel_name = "./RRH_results/RRH_results_on_{}.xlsx".format(set_name)
        df.to_excel(excel_name, index=False)