import copy
import os, time
import pandas as pd
import numpy as np

def gap_sheet(gnn_result_path, gnnmc_result_path):
    decimals = 2
    opt_results = pd.read_excel('./opt_baselines.xlsx')
    ffd_results = pd.read_excel('./FFD_baselines_on_all_dataset.xlsx')

    columns = [
        'policy',
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
    ]

    instance_set_name_list = [
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
    ]

    trim_loss_greedy_gap_mean = ['RG']
    frac_greedy_gap_mean = ['FG']
    gnn_gap_mean = ['BGCN']
    gnnmc_gap_mean = ['BGCNMC']
    pt_gap_mean = ['PTR']
    ffd_gap_mean = ['FFD']
    hrlgpn_gap_mean = ['HRL-GPN']
    rrmcts_gap_mean = ['RRMCTS']
    rrh_gap_mean = ['RRH']

    trim_loss_greedy_gap_std = ['RG']
    frac_greedy_gap_std = ['FG']
    gnn_gap_std = ['BGCN']
    gnnmc_gap_std = ['BGCNMC']
    pt_gap_std = ['PTR']
    ffd_gap_std = ['FFD']
    hrlgpn_gap_std = ['HRL-GPN']
    rrmcts_gap_std = ['RRMCTS']
    rrh_gap_std = ['RRH']

    trim_loss_greedy_gap_mean_std = ['RG']
    frac_greedy_gap_mean_std = ['FG']
    gnn_gap_mean_std = ['BGCN']
    gnnmc_gap_mean_std = ['BGCNMC']
    pt_gap_mean_std = ['PTR']
    ffd_gap_mean_std = ['FFD']
    hrlgpn_gap_mean_std = ['HRL-GPN']
    rrmcts_gap_mean_std = ['RRMCTS']
    rrh_gap_mean_std = ['RRH']

    for instance_set_name in instance_set_name_list:
        trim_loss_greedy_df = pd.read_excel('./RG/trim_loss_greedy_baselines_on_{}.xlsx'.format(instance_set_name))
        frac_greedy_df = pd.read_excel('./FG/frac_greedy_baselines_on_{}.xlsx'.format(instance_set_name))
        # gnn_df = pd.read_excel('{}/{}_on_gnn_rl.xlsx'.format(gnn_result_path, instance_set_name))
        gnnmc_df = pd.read_excel('{}/gnn_rl_on_{}.xlsx'.format(gnnmc_result_path, instance_set_name))
        ffd_merged_gnnmc_df = copy.deepcopy(gnnmc_df)
        ffd_merged_gnnmc_df.set_index('name', inplace=True, verify_integrity=True)
        gnn_df = pd.read_excel('{}/gnn_rl_on_{}.xlsx'.format(gnn_result_path, instance_set_name))
        ffd_merged_gnn_df = copy.deepcopy(gnn_df)
        ffd_merged_gnn_df.set_index('name', inplace=True, verify_integrity=True)
        pt_df = pd.read_excel('./PTR/drl4vrp_on_{}.xlsx'.format(instance_set_name))
        ffd_df = pd.read_excel('./FFD/FFD_baselines_on_{}.xlsx'.format(instance_set_name))
        hrlgpn_df = pd.read_excel('./HRLGPN/{}.xlsx'.format(instance_set_name))
        rrmcts_df = pd.read_excel('./RRMCTS/RRMCTS_result_on_{}.xlsx'.format(instance_set_name))
        rrh_df = pd.read_excel('./RRH/RRH_results_on_{}.xlsx'.format(instance_set_name))

        trim_loss_greedy_gap_list = []
        frac_greedy_gap_list = []
        gnn_gap_list = []
        gnnmc_gap_list = []
        pt_gap_list = []
        ffd_gap_list = []
        hrlgpn_gap_list = []
        rrmcts_gap_list = []
        rrh_gap_list = []
        for index, row in trim_loss_greedy_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['name']]['Best LB'].iloc[0]
            ffd = ffd_results[ffd_results['Name'] == row['name']]['obj'].iloc[0]
            trim_loss_greedy_gap_list.append(float(min(row['tlg_obj'], ffd)-opt)/opt*100)
        for index, row in frac_greedy_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['name']]['Best LB'].iloc[0]
            ffd = ffd_results[ffd_results['Name'] == row['name']]['obj'].iloc[0]
            frac_greedy_gap_list.append(float(min(row['frac_obj'], ffd)-opt)/opt*100)
        for index, row in gnn_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['name']]['Best LB'].iloc[0]
            ffd = ffd_results[ffd_results['Name'] == row['name']]['obj'].iloc[0]
            ffd_merged_gnn_df.loc[row['name'],'obj'] = min(row['obj'], ffd)
            ffd_merged_gnn_df.loc[row['name'], 'gap'] = float(row['obj']-opt)/opt*100
            gnn_gap_list.append(float(min(row['obj'], ffd)-opt)/opt*100)
            '{}/gnn_rl_on_{}.xlsx'.format(gnnmc_result_path, instance_set_name)

        for index, row in gnnmc_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['name']]['Best LB'].iloc[0]
            ffd = ffd_results[ffd_results['Name'] == row['name']]['obj'].iloc[0]
            ffd_merged_gnnmc_df.loc[row['name'],'obj'] = min(row['obj'], ffd)
            ffd_merged_gnnmc_df.loc[row['name'], 'gap'] = float(row['obj']-opt)/opt*100
            sp_gap = float(min(row['obj'], ffd)-row['sp_obj'])/row['sp_obj']*100
            ffd_merged_gnnmc_df.loc[row['name'], 'sp_gap'] = sp_gap if sp_gap>=0 else 0
            gnnmc_gap_list.append(float(min(row['obj'], ffd)-opt)/opt*100)
        for index, row in pt_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['name']]['Best LB'].iloc[0]
            ffd = ffd_results[ffd_results['Name'] == row['name']]['obj'].iloc[0]
            pt_gap_list.append(float(row['ptrnet']-opt)/opt*100)
        for index, row in ffd_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['Name']]['Best LB'].iloc[0]
            ffd_gap_list.append(float(row['ffd_obj'] - opt) / opt * 100)
        for index, row in hrlgpn_df.iterrows():
            opt = opt_results[opt_results['Name'] == row['name']]['Best LB'].iloc[0]
            ffd = ffd_results[ffd_results['Name'] == row['name']]['obj'].iloc[0]
            hrlgpn_gap_list.append(float(row['obj']-opt)/opt*100)
        for index, row in rrmcts_df.iterrows():
            rrmcts_gap_list.append(row['gap'])
        for index, row in rrh_df.iterrows():
            rrh_gap_list.append(row['gap'])

        trim_loss_greedy_gap_mean_std.append(str(np.around(np.mean(trim_loss_greedy_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(trim_loss_greedy_gap_list), decimals=decimals)))
        frac_greedy_gap_mean_std.append(str(np.around(np.mean(frac_greedy_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(frac_greedy_gap_list), decimals=decimals)))
        gnn_gap_mean_std.append(str(np.around(np.mean(gnn_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(gnn_gap_list), decimals=decimals)))
        gnnmc_gap_mean_std.append(str(np.around(np.mean(gnnmc_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(gnnmc_gap_list), decimals=decimals)))
        pt_gap_mean_std.append(str(np.around(np.mean(pt_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(pt_gap_list), decimals=decimals)))
        ffd_gap_mean_std.append(str(np.around(np.mean(ffd_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(ffd_gap_list), decimals=decimals)))
        hrlgpn_gap_mean_std.append(str(np.around(np.mean(hrlgpn_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(hrlgpn_gap_list), decimals=decimals)))
        rrmcts_gap_mean_std.append(str(np.around(np.mean(rrmcts_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(rrmcts_gap_list), decimals=decimals)))
        rrh_gap_mean_std.append(str(np.around(np.mean(rrh_gap_list), decimals=decimals))+'+-'+str(np.around(np.std(rrh_gap_list), decimals=decimals)))

        trim_loss_greedy_gap_mean.append(np.mean(trim_loss_greedy_gap_list))
        frac_greedy_gap_mean.append(np.mean(frac_greedy_gap_list))
        gnn_gap_mean.append(np.mean(gnn_gap_list))
        gnnmc_gap_mean.append(np.mean(gnnmc_gap_list))
        pt_gap_mean.append(np.mean(pt_gap_list))
        ffd_gap_mean.append(np.mean(ffd_gap_list))
        hrlgpn_gap_mean.append(np.mean(hrlgpn_gap_list))
        rrmcts_gap_mean.append(np.mean(rrmcts_gap_list))
        rrh_gap_mean.append(np.mean(rrh_gap_list))

        trim_loss_greedy_gap_std.append(np.std(trim_loss_greedy_gap_list))
        frac_greedy_gap_std.append(np.std(frac_greedy_gap_list))
        gnn_gap_std.append(np.std(gnn_gap_list))
        gnnmc_gap_std.append(np.std(gnnmc_gap_list))
        pt_gap_std.append(np.std(pt_gap_list))
        ffd_gap_std.append(np.std(ffd_gap_list))
        hrlgpn_gap_std.append(np.std(hrlgpn_gap_list))
        rrmcts_gap_std.append(np.std(rrmcts_gap_list))
        rrh_gap_std.append(np.std(rrh_gap_list))

    std_result = [trim_loss_greedy_gap_std, frac_greedy_gap_std,ffd_gap_std, gnn_gap_std, gnnmc_gap_std, pt_gap_std, hrlgpn_gap_std, rrmcts_gap_std, rrh_gap_std]
    std_resultdf = pd.DataFrame(std_result, columns=columns)# .round(decimals)
    std_resultdf.set_index('policy', inplace=True, verify_integrity=True)
    std_resultdf = pd.DataFrame(std_resultdf.values.T, index=std_resultdf.columns, columns=std_resultdf.index)

    mean_result = [trim_loss_greedy_gap_mean, frac_greedy_gap_mean, ffd_gap_mean, gnn_gap_mean, gnnmc_gap_mean, pt_gap_mean, hrlgpn_gap_mean, rrmcts_gap_mean, rrh_gap_mean]
    mean_resultdf = pd.DataFrame(mean_result, columns=columns)# .round(decimals)
    mean_resultdf.set_index('policy', inplace=True, verify_integrity=True)
    mean_resultdf = pd.DataFrame(mean_resultdf.values.T, index=mean_resultdf.columns, columns=mean_resultdf.index)

    std_mean_result = [trim_loss_greedy_gap_mean_std, frac_greedy_gap_mean_std, ffd_gap_mean_std, gnn_gap_mean_std, gnnmc_gap_mean_std, pt_gap_mean_std, hrlgpn_gap_mean_std, rrmcts_gap_mean_std, rrh_gap_mean_std]
    std_mean_resultdf = pd.DataFrame(std_mean_result, columns=columns)# .round(decimals)
    std_mean_resultdf.set_index('policy', inplace=True, verify_integrity=True)
    std_mean_resultdf = pd.DataFrame(std_mean_resultdf.values.T, index=std_mean_resultdf.columns, columns=std_mean_resultdf.index)

    return mean_resultdf, std_resultdf, std_mean_resultdf

def compare_sheet(mean_gap_df, std_gap_df):
    policy_list = ['RG', 'FG', 'FFD', 'BGCN', 'BGCNMC', 'PTR', 'HRL-GPN', 'RRMCTS', 'RRH']
    zeros_list = np.zeros((len(policy_list), len(policy_list)))
    compared_df = pd.DataFrame(zeros_list, index=policy_list, columns=policy_list)
    dataset_indices = mean_gap_df.index
    for policy_index_1 in policy_list:
        for policy_index_2 in policy_list:
            for dataset_index in dataset_indices:
                if mean_gap_df.loc[dataset_index, policy_index_1] < mean_gap_df.loc[dataset_index, policy_index_2]:
                    compared_df.loc[policy_index_1, policy_index_2] += 1
                elif mean_gap_df.loc[dataset_index, policy_index_1] == mean_gap_df.loc[dataset_index, policy_index_2]:
                    if std_gap_df.loc[dataset_index, policy_index_1] < std_gap_df.loc[dataset_index, policy_index_2]:
                        compared_df.loc[policy_index_1, policy_index_2] += 1
    return compared_df

if __name__ == '__main__':
    excel_writer = pd.ExcelWriter('./result_in_paper/compared_results.xlsx')
    gnnmc_result_path = './BGCNMC'
    gnn_result_path = './BGCN'
    
    mean_resultdf, std_resultdf, std_mean_resultdf = gap_sheet(gnn_result_path=gnn_result_path, gnnmc_result_path=gnnmc_result_path)
    compared_df = compare_sheet(mean_gap_df=mean_resultdf, std_gap_df=std_resultdf)

    std_mean_resultdf.to_excel(excel_writer, sheet_name='results.xlsx')
    compared_df.to_excel(excel_writer, sheet_name='compared.xlsx')
    excel_writer.save()
    excel_writer.close()