import torch


def eval_pure_results(data_table, env, agent, Rollout, collected_log_path, collecting_log_path, device):

    with open(collecting_log_path, 'a+') as file_object:
        file_object.writelines('{}'.format(env.instance_name) + '\n')

    buffer_item, buffer_edge, buffer_edge_indices, buffer_column, buffer_a, buffer_r, buffer_column_num, buffer_item_num, ep_r, ep_rwr, mip_select_obj = Rollout(env=env, agent=agent, device=device)

    data_table.append([env.instance_name, ep_rwr, float(ep_rwr-env.opt)/env.opt*100, mip_select_obj])

    with open(collected_log_path, 'a+') as file_object:
        file_object.writelines('{}'.format(env.instance_name) + '\n')
