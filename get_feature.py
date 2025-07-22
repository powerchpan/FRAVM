import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
path = 'E:\\ply\\code\\MS-GCN-main_tug\\MS-GCN-main\\data\\example\\features7\\example1_input.npy'
read_path = 'E:\\ply\\code\\MS-GCN-main_tug\\MS-GCN-main\\data\\sail\\keypoints\\1.npy'

def get_feature(path):
    data = np.load(path)
    Rhip = data[:, 9, :]
    Lhip = data[:, 12, :]
    # Mhip = (Rhip + Lhip) / 2 # - chunhua
    Mhip = data[:, 8, :]
    i = 0
    while i < 25:
        data[:, i, :] = data[:, i, :] - Mhip
        i += 1
    data_tensor = torch.tensor(data)
    data_tensor = data_tensor.permute(2, 0, 1)
    data_tensor = data_tensor.unsqueeze_(3)
    return data_tensor

def save_features(path, labelpath, savepath):
    i = 86
    label = pd.read_csv(labelpath)
    label = label.iloc[:, 2:3]
    label = label.iloc[1:,:]
    print(label)
    while i<101:
        print(i)
        if((i ==4)|(i==39)|(i==61)|(i==86)):
            i+=1
            continue
        index = int(label.iloc[i-3].item())
        print(index)
        end_path = str(i)+'.npy'
        read_path = path+end_path
        features = get_feature(read_path)
        features = features[:, index:, :]
        features = F.normalize(features)
        end_save_path = str(i)+'_input.npy'
        save_path = savepath+end_save_path
        np.save(save_path, features)
        i += 1
#save_features('E:\\ply\\code\\MS-GCN-main_tug\\MS-GCN-main\\data\\sail\\keypoints\\', 'E:\\ply\\data\\sail\\label\\label_100\\sail_frame.csv', 'E:\\ply\\code\\MS-GCN-#smain_tug\\MS-GCN-main\\data\\sail\\features7\\')



# input_fea = np.zeros((3, len(test_data), 25), dtype=float)
# i =0
# while i<len(test_data):
#     j = 0
#     while j<25:
#         keypoints = test_data[i][j]
#         input_fea[:, i, j] = keypoints
#         j += 1
#     i += 1
# print(input_fea)
# print(input_fea.shape)

# fea = test_data.view()

