
import joblib
import torch
import numpy as np
from get_individual_height import get_keypoints
from get_individual_height import get_individual_height

fps = 0

data_path = ''
net_path = 'D:\\2333\\code\\2d_keypoints_exp\\2d_keypoints_exp\\res\\net4.pkl'
svm_path = 'D:\\2333\\code\\2d_keypoints_exp\\2d_keypoints_exp\\res\\svm.m'


#载入神经网络模型
def predict_net(net_path):
    data = np.load(net_path)
    net = torch.load('net.pkl').cuda()
    prediction_net = net(data)
    return prediction_net

#载入svm模型
def predict_svm(svm_path):
    data = np.load(svm_path)
    clf = joblib.load("my_model.m")
    prediction_svm = clf.predict(data)
    return prediction_svm
#使用svm/神经网络预测每一帧的结果
def get_labels(data_path, net_path):
    net_model = predict_net(net_path)
    label_list = net_model(data_path)
    return label_list
#根据每一帧的结果计算起坐花费的总时间
def get_all_time(label_list):
    predict_arr = np.array(label_list)
    num_frames = len(predict_arr)
    time = num_frames  /  fps
    return time
#计算起坐次数
def get_sts_times(label_list):
    i = 0
    last_frame_list = []
    while i < len(label_list)-1:
        if ((label_list[i] == 3)&(label_list[i+1] == 0)):
            last_frame_list.append(i)
        i += 1
    #if (label_list[i-1] == 3):
    #    last_frame_list.append((len(label_list))-1)
    if((i==(len(label_list)-1))&(len(last_frame_list)==4)):
       if((label_list[i]== 3)&(label_list[i-1]==3)):
           last_frame_list.append(i)
    sts_times = len(last_frame_list)
    if(sts_times>5):
        i = 1
        while(i<sts_times):
            pre_last_frame = last_frame_list[i-1]
            now_last_frame = last_frame_list[i]
            if(now_last_frame-pre_last_frame<60):
                del last_frame_list[i]
                sts_times = sts_times-1
            i+=1
    return last_frame_list,sts_times

#计算每一次起坐中坐这一子任务花费时间
def get_each_sitting(label_list):

    print('video fps in get_feature_2.py:', fps)

    sitting_time_list = []
    i = 1
    j = 0
    last_frame_list,sts_times = get_sts_times(label_list)
    while i < sts_times:
        sitting_frame = 0
        if(i==0):
            j = 0
            while(j<last_frame_list[0]):
                if (label_list[j] == 0) & (label_list[j+1] != 1):
                    sitting_frame += 1
                    j += 1
                else:
                    j+=1
        
        else:
            j = last_frame_list[i-1]
            while(j<last_frame_list[i]):
               if (label_list[j] == 0) & (label_list[j+1] != 1):
        
                    sitting_frame += 1
                    j += 1
               else:
                    j+=1
        
       # while j < len(label_list)-1:
        #    if (label_list[j] == 0) & (label_list[j+1] != 1):
        #        sitting_frame += 1
        #        j += 1
        #    else:
        #        if (label_list[j] == 0) & (label_list[j+1] == 1):
        #            sitting_frame += 1
        #            j += 1
        #            break
        #        else:
        #            j += 1
        sitting_time = sitting_frame/fps
        sitting_time_list.append(sitting_time)
        i += 1
    return sitting_time_list

#计算每一次起坐中坐-站这一子任务花费时间
def get_each_sitstand(label_list):
    sitstand_time_list = []
    i = 0
    j = 0
    last_frame_list,sts_times = get_sts_times(label_list)
    while i < sts_times:
        sitstand_frame = 0
        if(i==0):
            j = 0
            while(j<last_frame_list[0]):
                if (label_list[j] == 1) & (label_list[j+1] != 2):
                    sitstand_frame += 1
                    j += 1
                else:
                    j+=1
        
        else:
            j = last_frame_list[i-1]
            while(j<last_frame_list[i]):
               if (label_list[j] == 1) & (label_list[j+1] != 2):
        
                    sitstand_frame += 1
                    j += 1
               else:
                    j+=1
        
    
    
    #while i < sts_times:
    #    sitstand_frame = 0
    #    while j < len(label_list)-1:
    #        if (label_list[j] == 1) & (label_list[j + 1] != 2):
    #            sitstand_frame += 1
    #            j += 1
    #        else:
    #            if (label_list[j] == 1) & (label_list[j + 1] == 2):
    #                sitstand_frame += 1
    #                j += 1
    #                break
    #            else:
    #                j += 1
        sitstand_time = sitstand_frame/fps
        sitstand_time_list.append(sitstand_time)
        i += 1
    return sitstand_time_list

#计算每一次起坐中站这一子任务花费时间
def get_each_standing(label_list):
    standing_time_list = []
    i = 0
    j = 0
    last_frame_list,sts_times = get_sts_times(label_list)
    while i < sts_times:
        standing_frame = 0
        if(i==0):
            j = 0
            while(j<last_frame_list[0]):
                if (label_list[j] == 2) & (label_list[j+1] != 3):
                    standing_frame += 1
                    j += 1
                else:
                    j+=1
        
        else:
            j = last_frame_list[i-1]
            while(j<last_frame_list[i]):
               if (label_list[j] == 1) & (label_list[j+1] != 2):
        
                    standing_frame += 1
                    j += 1
               else:
                    j+=1
    
    
    #while i < sts_times:
    #    standing_frame = 0
    #    while j < len(label_list):
    #        if (label_list[j] == '2') & (label_list[j + 1] != '3'):
    #            standing_frame += 1
    #            j += 1
    #        else:
    #            if (label_list[j] == '2') & (label_list[j + 1] == '3'):
    #                standing_frame += 1
    #                j += 1
    #                break
    #            else:
    #                j += 1
        standing_time = standing_frame/fps
        standing_time_list.append(standing_time)
        i += 1
    #print(standing_time_list)
    return standing_time_list

#计算每一次起站-坐中坐这一子任务花费时间
def get_each_standsit(label_list):
    standsit_time_list = []
    i = 0
    j = 0
    last_frame_list,sts_times = get_sts_times(label_list)
    while i < sts_times:
        standsit_frame = 0
        if(i==0):
            j = 0
            while(j<last_frame_list[0]):
                if (label_list[j] == 3) & (label_list[j+1] != 0):
                    standsit_frame += 1
                    j += 1
                else:
                    j+=1
        
        elif(i<sts_times-1):
            j = last_frame_list[i-1]
            while(j<last_frame_list[i]):
               if (label_list[j] == 3) & (label_list[j+1] != 0):
        
                    standsit_frame += 1
                    j += 1
               else:
                    j+=1
        else:
           j = last_frame_list[i-1]
           while(j<last_frame_list[i]):
               if(label_list[j] == 3):
                    standsit_frame += 1
                    j += 1
               else:
                    j+=1
    
    
    #while i < sts_times:
    #    standsit_frame = 0
    #    while j < len(label_list)-1:
    #        if (label_list[j] == 3) & (label_list[j + 1] != 0):
    #            standsit_frame += 1
    #            j += 1
    #        else:
    #            if (label_list[j] == 3) & (label_list[j + 1] == 0):
    #                standsit_frame += 1
    #                j += 1
    #                break
    #            else:
    #                if (label_list[j] == 3):
    #                    standsit_frame += 1
    #                    j += 1
    #                    break
    #                else:
    #                    j += 1
        standsit_time = standsit_frame/fps
        standsit_time_list.append(standsit_time)
        i += 1
    return standsit_time_list


#计算每次起坐中，坐-站这一子任务的速度
def get_sitstand_velocity(keypoints, label):
    height = get_individual_height(keypoints)
    sitstand_time_list = get_each_sitstand(label)
    each_sitstand_velocity = []
    for each_sitstand_time in sitstand_time_list:
        velocity = height/each_sitstand_time
        each_sitstand_velocity.append(velocity)
    return each_sitstand_velocity

#计算每次起坐中，站-坐这一子任务的速度
def get_standsit_velocity(keypoints, label):
    height = get_individual_height(keypoints)
    standsit_time_list = get_each_standsit(label)
    each_standsit_velocity = []
    for each_standsit_time in standsit_time_list:
        velocity = height/each_standsit_time
        each_standsit_velocity.append(velocity)
    return each_standsit_velocity

#计算每次起坐的速度
def get_sts_velocity(keypoints, label):
    height = get_individual_height(keypoints)
    sts_time_list = get_single_sts_duration(label)
    #print(sts_time_list)
    #print(sts_time_list)
    each_sts_velocity = []
    for each_sts_time in sts_time_list:
        velocity = height/each_sts_time
        each_sts_velocity.append(velocity)
    return each_sts_velocity


#计算每次起坐中，坐-站一子任务的平均速度
def get_mean_sitstand_velocity(keypoints, label):
    each_sitstand_velocity = get_sitstand_velocity(keypoints, label)
    sum_velocity = 0
    for each in each_sitstand_velocity:
        sum_velocity += each
    mean_sitstand_velocity = sum_velocity/len(each_sitstand_velocity)
    return mean_sitstand_velocity

#计算每次起坐中，站-坐一子任务的平均速度
def get_mean_standsit_velocity(keypoints, label):
    each_standsit_velocity = get_standsit_velocity(keypoints, label)
    sum_velocity = 0
    for each in each_standsit_velocity:
        sum_velocity += each
    mean_standsit_velocity = sum_velocity/len(each_standsit_velocity)
    return mean_standsit_velocity

#计算起坐的平均速度
def get_mean_sts_velocity(keypoints, label):
    each_sts_velocity = get_sts_velocity(keypoints, label)
    sum_velocity = 0
    for each in each_sts_velocity:
        sum_velocity += each
    mean_sts_velocity = sum_velocity/5
    return mean_sts_velocity

#计算每次起坐中，坐-站一子任务的最大速度
def get_max_sitstand_velocity(keypoints, label):
    each_sitstand_velocity = get_sitstand_velocity(keypoints, label)
    max_sitstand_velocity = max(each_sitstand_velocity)
    return max_sitstand_velocity

#计算每次起坐中，站-坐一子任务的最大速度
def get_max_standsit_velocity(keypoints, label):
    each_standsit_velocity = get_standsit_velocity(keypoints, label)
    max_standsit_velocity = max(each_standsit_velocity)
    return max_standsit_velocity

#计算起坐的最大速度
def get_max_sts_velocity(keypoints, label):
    each_sts_velocity = get_sts_velocity(keypoints, label)
    max_sts_velocity = max(each_sts_velocity)
    return max_sts_velocity

#计算每次起坐中，坐-站一子任务的最小速度
def get_min_sitstand_velocity(keypoints, label):
    each_sitstand_velocity = get_sitstand_velocity(keypoints, label)
    min_sitstand_velocity = min(each_sitstand_velocity)
    return min_sitstand_velocity

#计算每次起坐中，站-坐一子任务的最小速度
def get_min_standsit_velocity(keypoints, label):
    each_standsit_velocity = get_standsit_velocity(keypoints, label)
    min_standsit_velocity = min(each_standsit_velocity)
    return min_standsit_velocity

#计算起坐的最小速度
def get_min_sts_velocity(keypoints, label):
    each_sts_velocity = get_sts_velocity(keypoints, label)
    min_sts_velocity = min(each_sts_velocity)
    return min_sts_velocity

#计算每次起坐中，坐-站这一子任务的加速度
def get_sitstand_accelaration(keypoints, label):
    each_sitstand_velocity = get_sitstand_velocity(keypoints, label)
    sitstand_time_list = get_each_sitstand(label)
    each_sitstand_accelaration = []
    i = 0
    while i < len(sitstand_time_list):
        accelaration = each_sitstand_velocity[i]/sitstand_time_list[i]
        each_sitstand_accelaration.append(accelaration)
        i += 1
    return each_sitstand_accelaration


#计算每次起坐中，站-坐这一子任务的加速度
def get_standsit_accelaration(keypoints, label):
    each_standsit_velocity = get_standsit_velocity(keypoints, label)
    standsit_time_list = get_each_standsit(label)
    each_standsit_accelaration = []
    i = 0
    while i < len(standsit_time_list):
        accelaration = each_standsit_velocity[i]/standsit_time_list[i]
        each_standsit_accelaration.append(accelaration)
        i += 1
    return each_standsit_accelaration

#计算每次起坐的加速度
def get_each_sts_accelaration(keypoints, label):
    each_sts_velocity = get_sts_velocity(keypoints, label)
    sts_time_list = get_single_sts_duration(label)
    each_sts_accelaration = []
    i = 0
    while i < len(sts_time_list):
        accelaration = each_sts_velocity[i]/sts_time_list[i]
        each_sts_accelaration.append(accelaration)
        i += 1
    return each_sts_accelaration

#计算整个起坐过程的加速度
def get_sts_accelaration(keypoints, label):
    each_sts_accelaration = get_each_sts_accelaration(keypoints, label)
    sum_accelaration = 0
    for each_accelaration in each_sts_accelaration:
        sum_accelaration += each_accelaration
    sts_mean_accelaration = sum_accelaration/len(each_sts_accelaration)
    return sts_mean_accelaration


#计算起坐中坐这一子任务的平均花费时间
def get_mean_sitting(label_list):
    i = 1
    sitting_list = []
    sts_times = 0
    while i<len(label_list):
        if(label_list[i]==1)&(label_list[i-1]==0):
            break;
        i+=1
    while i < len(label_list):
        if label_list[i] == 0:
                sitting_list.append(i)
        i += 1
    sitting_frame = np.array(sitting_list)
    all_sitting_frames = (len(sitting_frame))/fps
    last_frame_list, sts_times = get_sts_times(label_list)
    mean_sitting_duration = all_sitting_frames/sts_times
    return mean_sitting_duration

#计算起坐中站这一子任务的平均花费时间
def get_mean_standing(label_list):
    i = 0
    standing_list = []
    while i < len(label_list):
        if label_list[i] == 2:
            standing_list.append(i)
        i += 1
    standing_frame = np.array(standing_list)
    all_standing_frames = (len(standing_frame))/fps
    last_frame_list, sts_times = get_sts_times(label_list)
    mean_standing_duration = all_standing_frames/sts_times
    return mean_standing_duration

#计算每一次起坐花费的时间
def get_single_sts_duration(label_list):
    i = 0
    last_frame_list = []
    sts_duration_list = []
    while i < len(label_list)-1:
        if (label_list[i] == 3) & (label_list[i + 1] == 0):
            last_frame_list.append(i)
        i += 1
    if((i==(len(label_list)-1))&(len(last_frame_list)==4)):
       if((label_list[i]== 3)&(label_list[i-1]==3)):
           last_frame_list.append(i)
    sts_times = len(last_frame_list)
    if(sts_times>5):
        i = 1
        while(i<sts_times):
            pre_last_frame = last_frame_list[i-1]
            now_last_frame = last_frame_list[i]
            if(now_last_frame-pre_last_frame<60):
                del last_frame_list[i]
                sts_times = sts_times-1
            i+=1
   # print(last_frame_list) 
    #last_frame_list.append((len(label_list))-1)
    #print(last_frame_list) 
    i = 1
    while i<len(label_list):
        if(label_list[i]==1)&(label_list[i-1]==0):
            first_sts_start_frame = i
            break
        i+=1
    sts_duration_list.append((last_frame_list[0]-first_sts_start_frame)/fps)
    i = 1
    while i < len(last_frame_list):
        sts_duration = (last_frame_list[i] - last_frame_list[i-1])/fps
        sts_duration_list.append(sts_duration)
        i += 1
    return sts_duration_list





