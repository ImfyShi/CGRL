import numpy as np
import copy

def pure_diving(item_features, edges, column_features):
    selected_id, min_val, flag = 0, 1e10, 1
    for col_id in range(len(column_features)):
        frac = column_features[col_id][0]
        if 0 >= frac: continue
        upper_gap = np.ceil(frac) - frac
        lowwer_gap = frac - np.floor(frac)
        # frac solution is integer
        if (1e-10 > lowwer_gap) and (1 > frac):
            selected_id, flag = col_id, 0
            break
        if (1e-10 > upper_gap):
            selected_id, flag = col_id, 1
            break
        if min_val > upper_gap:
            selected_id = col_id
            min_val = upper_gap
            flag = 1
        if (min_val > lowwer_gap) and (frac > 1):
            selected_id = col_id
            min_val = lowwer_gap
            flag = 0
    return [edges[:,selected_id]], [column_features[selected_id,0]], [flag]

def trim_loss_greedy(item_features, edges, column_features):
    selected_id, min_val, flag, min_trim_loss = 0, 1e10, 1, 1e10
    for col_id in range(len(column_features)):
        frac = column_features[col_id][0]
        if 0 >= frac: continue
        if (0.5 > np.inner(np.maximum(np.array(item_features), 0)[:, 1],edges[:, col_id])): continue
        trim_loss = column_features[col_id][2]
        if min_trim_loss >= trim_loss:
            min_trim_loss = trim_loss
            selected_id = col_id
            if (1 > frac):
                flag = 1
            elif 0.5>(frac-np.floor(frac)):
                flag = 0
            else:
                flag = 1
    return selected_id, flag
