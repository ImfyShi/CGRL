
def Evaluate(data_table, env, policy, construct_edge_feature, collected_log_path, collecting_log_path):

    with open(collecting_log_path, 'a+') as file_object:
        file_object.writelines('{}'.format(env.instance_name) + '\n')

    item_features, edges, column_features = env.reset()
    done, bins_count = False, 0
    while not done:  # in one episode
        cut_pattern_list, fra_solu_list, flag_list  = policy(item_features=item_features, edges=edges, column_features=column_features)
        item_features, edges, column_features, reward, done = env.step(cut_pattern_list=cut_pattern_list, fra_solu_list=fra_solu_list, flag_list=flag_list)
        bins_count += reward
    setpartitioning_obj = env.set_partitioning_problem(column=env.history_column_pool)
    data_table.append([env.instance_name, bins_count, float(bins_count-env.opt)/env.opt*100, setpartitioning_obj, float(setpartitioning_obj-env.baseline)/env.opt*100])

    with open(collected_log_path, 'a+') as file_object:
        file_object.writelines('{}'.format(env.instance_name) + '\n')

def test_eval(data_table, env, policy, utilities):
    print(env.instance_name)