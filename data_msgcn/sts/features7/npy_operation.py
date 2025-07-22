import numpy as np
np.set_printoptions(threshold=np.inf, suppress= True)

data_path = "./"
file_name = "1.npy"

dataSource = np.load(data_path+file_name)
print("dataSource.shape", dataSource.shape)
print("dataSource:\n", dataSource)
