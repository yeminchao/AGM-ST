from scipy.io import loadmat
import numpy as np
import random

# Load data

source_label = source_label.flatten()
target_label = target_label.flatten()
source_data = np.reshape(source_data, (source_data.shape[0] * source_data.shape[1], -1))
target_data = np.reshape(target_data, (target_data.shape[0] * target_data.shape[1], -1))


class_num = len(np.unique(source_label)) - 1

source_each_calss_number = 400
target_each_calss_number = 5
repeat_times = 10
data_name = "RPaviaU-DPaviaC"


def find_training_index(label, class_num, each_calss_number):
    for i in range(class_num):
        idx = np.array(np.where(label == (i + 1))).flatten()
        rand_sample = random.sample(list(idx), each_calss_number)
        if i == 0:
            train_idx = rand_sample
        else:
            train_idx = np.concatenate((train_idx, rand_sample), axis=0)
    return train_idx


for times in range(repeat_times, data_name):
    source_train_idx = find_training_index(
        source_label, class_num, source_each_calss_number
    )
    target_train_idx = find_training_index(
        target_label, class_num, target_each_calss_number
    )
    np.save(
        "./training_sample/" + data_name + "/source_train_idx" + str(times) + ".npy",
        source_train_idx,
    )
    np.save(
        "./training_sample/" + data_name + "/target_train_idx" + str(times) + ".npy",
        target_train_idx,
    )
