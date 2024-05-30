# subproblem solved using minknap algorithm which comes from http://hjemmesider.diku.dk/~pisinger/minknap.c

import gurobipy as grb
from gurobipy import GRB
import numpy as np
from numpy import *
import copy
import os
import ctypes

class Diving:

    # cut_pattern = list() # 列池
    # reduce_cost = list() # 记录每一次加入列对应的reduced cost
    # trim_loss = list()  # 记录每一列trim_loss
    # buffer_item = list()  # 需要返回的item属性
    # feasible_solution = list()  # diving过程存放的可行解
    # demand_width_array = np.array(list())  # 全局宽度
    # demand_number_array =np.array(list())  # 全局个数
    # roll_width = 0 # 全局长度
    # fra_solu = list()
    # instance_path = ''

    def __init__(self, instance_path, baseline, opt, write_instance_data_folder=None, is_collect_root_info=False):
        self.instance_name = os.path.basename(instance_path)
        self.baseline = baseline
        self.opt = opt
        self.roll_width = 1
        self.used_bin = list()  # 记录当前用了几个bin
        self.instance_path = instance_path
        self.minknap_dll_path = './environment/minknap.so'
        # self.minknap_dll_path = './minknap.so'

        if is_collect_root_info:
            self.load_minknapdll()

            self.trim_loss = None
            self.initial_data()
            self.get_lowerbound_and_columns()
            self.release_minknapdll()

            np.savetxt(write_instance_data_folder + '/trim_loss.txt', np.array(self.trim_loss), delimiter=',')
            np.savetxt(write_instance_data_folder + '/cut_pattern.txt', self.cut_pattern, delimiter=',')
            np.savetxt(write_instance_data_folder + '/fra_solu.txt', self.fra_solu, delimiter=',')
            np.savetxt(write_instance_data_folder + '/reduce_cost.txt', self.reduce_cost, delimiter=',')

            self.cut_pattern_in_root = copy.deepcopy(self.cut_pattern)
            # self.trim_loss_in_root = copy.deepcopy(np.array(self.trim_loss))
            self.trim_loss_in_root = copy.deepcopy(np.array(self.trim_loss)).tolist()
            self.fra_solu_in_root = copy.deepcopy(self.fra_solu)
            self.reduce_cost_in_root = copy.deepcopy(self.reduce_cost)
            self.history_column_pool = copy.deepcopy(self.cut_pattern)


        else:
            self.initial_data()
            self.cut_pattern_in_root = np.loadtxt(write_instance_data_folder + '/cut_pattern.txt', delimiter=',')
            # self.trim_loss_in_root = np.loadtxt(write_instance_data_folder + '/trim_loss.txt', delimiter=',')
            self.trim_loss_in_root = np.loadtxt(write_instance_data_folder + '/trim_loss.txt', delimiter=',').tolist()
            self.fra_solu_in_root = np.loadtxt(write_instance_data_folder + '/fra_solu.txt', delimiter=',')
            self.reduce_cost_in_root = np.loadtxt(write_instance_data_folder + '/reduce_cost.txt', delimiter=',')

            if self.cut_pattern_in_root.shape[1] != np.unique(self.cut_pattern_in_root, axis=1).shape[1]:
                asdfasdf = 2435

    def load_minknapdll(self):
        self.minknap_dll = ctypes.CDLL(self.minknap_dll_path)
        self.minknap_dll.minknap.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.minknap_dll.minknap.restype = ctypes.c_double

    def release_minknapdll(self):
        self.minknap_dll = None

    def initial_data(self):
        instance_path = self.instance_path
        demand_number = list()
        demand_width = list()
        with open(instance_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            count_line = int(line) + 1
            for id in range(count_line):
                if id == 0:
                    self.roll_width = np.array(int(f.readline()))
                else:
                    line = f.readline()
                    line = line.rstrip("\n")
                    a = line.split("\t")
                    b = int(a[0])
                    c = int(a[1])
                    demand_width.append(b)
                    demand_number.append(c)
            self.demand_width_array = np.array(demand_width)  # / self.roll_width
            # self.roll_width = 1
            self.demand_number_array = np.array(demand_number)
            self.demand_number_array_initial = np.array(demand_number)  # 不随着改变
            self.item_num = len(self.demand_number_array_initial)

    # 获得首个下界
    def get_lowerbound_and_columns(self):
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

        self.cut_pattern_in_root = self.cut_pattern
        self.fra_solu_in_root = self.fra_solu
        self.reduce_cost_in_root = self.reduce_cost
        self.trim_loss_in_root = copy.deepcopy(self.trim_loss)

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

    def set_partitioning_problem(self, column):
        m = grb.Model()
        m.setParam('OutputFlag', 0)  # 不显示信息
        x = m.addMVar(shape=column.shape[1], lb=0, vtype=GRB.INTEGER)
        m.addConstr(lhs=column @ x >= self.demand_number_array_initial)
        m.setObjective(x.sum(), GRB.MINIMIZE)
        m.optimize()
        return m.objVal

    def restricted_lp_master_problem(self, column):
        return self.master_problem(column, GRB.CONTINUOUS)

    def restricted_ip_master_problem(self, column):
        return self.master_problem(column, GRB.INTEGER)

    def knapsack_subproblem(self, demand_item_number, demand_pi, demand_items_length, minknap_solution, minknap_solution_para_matrix, binary_cutting_stock_solution_para_matrix):
        '''
        print('')
        for ele in np_demand_length_array:
            print(' {}'.format(ele), end="")
        print('')
        self.minknap_dll = ctypes.CDLL('./minknap.so')
        # minknap(int n, int *p, int *w, int *x, int c)

        pyarray = np.asarray(range(demand_item_number), dtype=int)
        carray_w = pyarray.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        pyarray = pyarray.astype(double)
        carray_p = pyarray.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        self.minknap_dll.checkparas.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.minknap_dll.checkparas(demand_item_number, demand_pi, demand_items_length, minknap_solution, self.roll_width)

        '''

        demand_item_number = int(np.maximum(self.demand_number_array, 0).sum())
        binary_item_number = int(demand_item_number) + np.count_nonzero(0 >= self.demand_number_array)
        minknap_solution = (ctypes.c_int * demand_item_number)(*np.zeros(demand_item_number, dtype=np.int32))
        # self.minknap_dll.n4()
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
            # print(f'reduce: {m.objVal}')
            # for i in range(len(new_column)):
            #     if new_column[i] > 0:
            #         print(f'index, value: {i},{new_column[i]}')
        else:
            new_column = None

        return flag_new_column, new_column

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

    def is_cut_pattern_feasible(self, line):
        cut_demand = self.cut_pattern.sum(axis=1)
        for idx in range(self.item_num):
            cut, demand = cut_demand[idx], self.demand_number_array[idx]
            if (0==cut) and (0<demand):
                asdfasdf = 23423423

    # return column feature as: 1) frac solution, 2) reduce cost and 3) trim loss
    def step(self, cut_pattern_list, fra_solu_list, flag_list):
        bin = 0
        for cut_pattern, fra_solu, flag in zip(cut_pattern_list, fra_solu_list, flag_list):
            '''
            deleted_idx = np.where((self.cut_pattern.T==cut_pattern).all(1))[0]
            assert(1==len(deleted_idx))
            self.trim_loss = np.delete(self.trim_loss, deleted_idx)
            self.reduce_cost = np.delete(self.reduce_cost, deleted_idx)
            self.cut_pattern = np.delete(self.cut_pattern, deleted_idx, axis=1)
            '''
            self.feasible_solution = np.append(self.feasible_solution, [cut_pattern], axis=0)  # 固定解加入
            if (flag == 0):
                cur_bin = np.floor(fra_solu)
            else:
                cur_bin = np.ceil(fra_solu)
            bin += cur_bin
            self.demand_number_array = self.demand_number_array - np.array(cut_pattern) * cur_bin  # 更新欠量
            self.is_cut_pattern_feasible(301)
        # self.demand_number_array = np.maximum(self.demand_number_array, 0)
        self.demand_number_array = self.demand_number_array.astype(np.int32)
        count_bin = np.sum(self.used_bin)  # 计算当前使用的bin的数量:
        if(np.inner(np.maximum(self.demand_number_array, 0), self.demand_width_array) <= self.roll_width):
            bin += 1
            is_done = True
            self.used_bin.append(bin)  # 解值加入
            return None, None, None, copy.deepcopy(
            bin), copy.deepcopy(is_done)
        self.used_bin.append(bin)  # 解值加入
        if (np.any(self.demand_number_array > 0.5)):
            is_done = False
        else:
            is_done = True
        if is_done:
            return None, None, None, copy.deepcopy(
            bin), copy.deepcopy(is_done)

        self.fra_solu = self.C_G()

        buffer_item = [list(item) for item in
                       zip(self.demand_width_array / self.roll_width,
                           self.demand_number_array)]  # 返回 item，行为item，第一列为宽度，第二列为需求量
        buffer_edge = self.cut_pattern  # 返回模式，行为item，列为方案，元素为，方案内item使用的个数

        frac_feature = copy.deepcopy(self.fra_solu)
        frac_feature[frac_feature < 1] = frac_feature[frac_feature < 1] * (-1) + 1
        frac_feature[frac_feature >= 1] = 1 - np.minimum((frac_feature[frac_feature >= 1] - np.floor(
            frac_feature[frac_feature >= 1])), (np.ceil(
            frac_feature[frac_feature >= 1]) - frac_feature[frac_feature >= 1]))
        buffer_column = [list(item) for item in
                         zip(self.fra_solu, self.trim_loss / self.roll_width, frac_feature, 1-self.trim_loss / self.roll_width)]  # 返回 列特征，行为列，第一列为解值，第二列为reduce-cost，第三列为切损量

        return copy.deepcopy(np.array(buffer_item)), copy.deepcopy(buffer_edge), copy.deepcopy(np.array(buffer_column)), copy.deepcopy(bin), copy.deepcopy(is_done)

    def reset(self):  # 回归到diving之前的根节点状态
        self.item_num = len(self.demand_number_array_initial)
        self.demand_number_array = copy.deepcopy(self.demand_number_array_initial)
        self.history_column_pool = copy.deepcopy(self.cut_pattern_in_root)
        self.cut_pattern = copy.deepcopy(self.cut_pattern_in_root)
        self.fra_solu = copy.deepcopy(self.fra_solu_in_root)
        self.reduce_cost = copy.deepcopy(self.reduce_cost_in_root)
        self.trim_loss = copy.deepcopy(self.trim_loss_in_root)
        count = len(self.demand_width_array)  # item的数量
        self.feasible_solution = np.empty(shape=[0, count])  # Diving过程中最终可行解存放

        buffer_item = [list(item) for item in
                       zip(self.demand_width_array / self.roll_width,
                           self.demand_number_array)]  # 返回 item，行为item，第一列为宽度，第二列为需求量
        buffer_edge = self.cut_pattern  # 返回模式，行为item，列为方案，元素为，方案内item使用的个数

        frac_feature = copy.deepcopy(self.fra_solu)
        frac_feature[frac_feature < 1] = frac_feature[frac_feature < 1] * (-1) + 1
        frac_feature[frac_feature >= 1] = 1 - np.minimum((frac_feature[frac_feature >= 1] - np.floor(
            frac_feature[frac_feature >= 1])), (np.ceil(
            frac_feature[frac_feature >= 1]) - frac_feature[frac_feature >= 1]))

        buffer_column = [list(item) for item in
                         zip(self.fra_solu,
                             self.trim_loss / self.roll_width, frac_feature, 1-self.trim_loss / self.roll_width)]  # 返回 列特征，行为列，第一列为切损量，第二列为解值，第三列为reduce-cost
        return copy.deepcopy(np.array(buffer_item)), copy.deepcopy(buffer_edge), copy.deepcopy(np.array(buffer_column))

def main():
    ins_path = '/home/sfy/Store/Benchmark/CuttingStockProblem/paper_used_instances/ANI/201_2500_NR_0.txt'
    D = Diving(ins_path, -1, -1)
    #   D.initial_data()  # 初始化数据
    #   D.get_lowerbound_and_columns() #获得下界

    while np.any(D.demand_number_array > 0):  # 若还有需求，则迭代
        flag = True  # 判断是否存在整数解
        column_count = len(D.fra_solu)  # 列的个数
        for i in range(column_count):  # 找fixed
            if (D.fra_solu[i] - np.floor(D.fra_solu[i])) < 0.00001 and D.fra_solu[i] > 0:  # 找到整数固定
                D.step(i, 1)
                flag = False
                break
        if flag:  # 如果不存在整数解，则找分数
            p_fra_solu = D.fra_solu - D.fra_solu.astype(np.int)  # 获得分数部分
            max_solu_value = 0  # 记录最接近分数的值
            max_solu_index = -1  # 记录最接近分数的列序号
            for i in range(column_count):  # 寻找最接近分数的列
                if p_fra_solu[i] > max_solu_value and p_fra_solu[i] - np.floor(p_fra_solu[i]) > 0.0001:
                    max_solu_value = p_fra_solu[i]
                    max_solu_index = i
            if max_solu_index != -1:  # 如果找到了，更新
                D.step(max_solu_index, 1)

    print('************************************************')
    print('parameter:')
    print(f'roll_width: {D.roll_width}')
    print(f'demand_width_array: {D.demand_width_array}')
    print(f'demand_number_array: {D.demand_number_array}')
    print('result:')
    print(f'optimal_value: {np.sum(D.used_bin)}')
    print(f'solution: {D.feasible_solution}')

    buffer_item, buffer_edge, buffer_column = D.reset()
    print(f'buffer_item: {buffer_item}')
    print(f'buffer_edge: {buffer_edge}')
    print(f'buffer_column: {buffer_column}')

if __name__ == '__main__':

    '''
    main()
    '''

    instances_set_folder = '/home/sfy/Store/Benchmark/CuttingStockProblem/original_instances/instances/Scholl_CSP/Scholl_3/HARD1.txt'
    make_folder_path = '/home/sfy/Store/Benchmark/CuttingStockProblem/tobetrained_instances/instances/root/Scholl_CSP/Scholl_3/HARD1'

    # ins_path = '/home/sfy/Store/Benchmark/CuttingStockProblem/debug/Hard28_BPP13.txt'
    # ins_path = '/home/sfy/Store/Benchmark/CuttingStockProblem/paper_used_instances/ANI/201_2500_NR_0.txt'
    D = Diving(instance_path=instances_set_folder, baseline=-1, opt=-1, write_instance_data_folder=make_folder_path,
               is_collect_root_info=False)
    D.load_minknapdll()
    #   D.initial_data()  # 初始化数据
    #   D.get_lowerbound_and_columns() #获得下界

    item_features, edges, column_features = D.reset()

    done, bins_num = False, 0
    while not done:  # 若还有需求，则迭代

        '''
        selected_id, min_val, flag = 0, 1e10, 1
        for col_id in range(len(column_features)):
            frac = column_features[col_id][0]
            if 0>=frac: continue
            upper_gap = 1 - frac + np.floor(frac);
            lowwer_gap = frac - np.floor(frac)
            # frac solution is integer
            if (1e-10 > lowwer_gap) and (0 < frac):
                selected_id, flag = col_id, 1
                # item_features, edges, column_features, reward, done = D.step(selected_id, flag)
                # bins_num += reward
                # break
            if min_val > upper_gap:
                selected_id = col_id
                min_val = upper_gap
                flag = 1
            if (min_val > lowwer_gap) and (frac > 1):
                selected_id = col_id
                min_val = lowwer_gap
                flag = 0

        '''
        print(len(column_features))
        # selected_id, flag = pure_diving(column_features)
        selected_id, flag = trim_loss_greedy(item_features=item_features, edges=edges, column_features=column_features)

        print(selected_id, flag)
        if 20==bins_num:
            safaf = 24423
        item_features, edges, column_features, reward, done = D.step(selected_id, flag)
        bins_num += reward
        print('bings count: {}, rewar: {}'.format(bins_num, reward))
        print('remained items number = {}'.format(np.maximum(D.demand_number_array, 0).sum()))

    print('************************************************')
    # print(f'optimal_value: {np.sum(D.used_bin)}')
    print(f'optimal_value: {bins_num}')