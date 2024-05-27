import numpy as np
from scipy.io import loadmat


# Load data

target_data = np.reshape(target_data, (target_data.shape[0] * target_data.shape[1], -1))
target_label = target_label.flatten()


data_class_num = len(np.unique(target_label)) - 1
num_per_class = 5

repeat_times = 10
LG_all_edges = 400
LG_extra_edges = LG_all_edges - data_class_num * num_per_class

data_name = "EHangzhou-RPaviaHR"


def get_random_idx(n, x):

    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index


def find_LG_index(target_label, data_name, times):
    all_idx = np.arange(0, target_label.shape[0])
    zero_idx = np.argwhere(target_label == 0).flatten()
    occupied_idx = np.load(
        "./training_sample/" + data_name + "/target_train_idx" + str(times) + ".npy"
    )
    delete_idx = np.append(zero_idx, occupied_idx, axis=0)
    remain_idx = np.delete(all_idx, delete_idx)
    remain_idx_num = remain_idx.shape[0]
    random = get_random_idx(remain_idx_num, LG_extra_edges)
    LG_idx = remain_idx[random]
    return LG_idx


for times in range(repeat_times):
    target_LG_idx = find_LG_index(target_label, data_name, times)
    np.save(
        "./LG_sample/" + data_name + "/target_LG_idx" + str(times) + ".npy",
        target_LG_idx,
    )
