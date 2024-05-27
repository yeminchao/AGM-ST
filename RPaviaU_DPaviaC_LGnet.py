from scipy.io import loadmat
from data_pre import LGnet_data_pre
from LGnet_training_process import training


# Load data

patch_size = 9

repeat_times = 1

data_name = "RPaviaU-DPaviaC"

network_name = "LGnet"

sepctral_firrst_hidden_dim = 50

r_list = [5, 10, 15, 20]

epochs = 1000


def a_group_test(r):
    for times in range(repeat_times):
        print("r ", r, "times ", times)
        (
            source_train_data_spectral,
            target_train_data_spectral,
            target_test_data_spectral,
            source_train_data_spatial,
            target_train_data_spatial,
            target_test_data_spatial,
            source_train_label,
            target_train_label,
            target_test_label,
            class_num,
            target_labeled_num,
        ) = LGnet_data_pre(
            initial_source_data,
            initial_target_data,
            initial_source_label,
            initial_target_label,
            data_name,
            times,
            patch_size,
        )

        training(
            source_train_data_spectral,
            source_train_data_spatial,
            source_train_label,
            target_train_data_spectral,
            target_train_data_spatial,
            target_train_label,
            target_test_data_spectral,
            target_test_data_spatial,
            target_test_label,
            class_num,
            times,
            patch_size,
            data_name,
            network_name,
            sepctral_firrst_hidden_dim,
            target_labeled_num,
            r,
            epochs,
        )


if __name__ == "__main__":

    for r in r_list:
        r = 20
        a_group_test(r)
        break
