import math
import numpy as np



num_keypoints = 25
dimension_point = 3
def get_individual_height(keypoints):
    nose = keypoints[:, 0, : ]
    # Lfoot = keypoints[:, 3, :] - chunhua
    Lfoot = keypoints[:, 11, :]
    # Rfoot = keypoints[:, 6, :] - chunhua
    Rfoot = keypoints[:, 14, :]
    sum_foot = Lfoot + Rfoot
    Midfoot = sum_foot/2
    i = 0
    height_list = []
    while i < len(keypoints):
        height = abs((Midfoot[i][1] - nose[i][1]))
        height_list.append(height)
        i += 1
    max_height = max(height_list)
    return max_height

def get_keypoints(path):
    keypointsList = np.load(path, allow_pickle=True)
    keypointsList = np.array(keypointsList)
    num_frame = int(len(keypointsList) / num_keypoints)
    keypoints_arr = np.reshape(keypointsList, (num_frame, num_keypoints))
    keypoints_arr = np.array(keypoints_arr)
    keypoints = np.zeros((num_frame, num_keypoints, dimension_point))
    i = 0
    while i < len(keypoints_arr):
        j = 0
        while j < num_keypoints:
            point = keypoints_arr[i][j]
            x = point.x
            y = point.y
            z = point.z
            point = np.array([x, y, z])
            keypoints[i][j] = point
            j += 1
        i += 1
    return keypoints


