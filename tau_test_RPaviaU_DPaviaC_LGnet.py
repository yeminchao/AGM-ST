from scipy.io import loadmat
from data_pre import LGnet_data_pre
from tau_test_LGnet_training_process import training


# Load data

patch_size = 9

repeat_times = 10

data_name = "RPaviaU-DPaviaC"

network_name = "LGnet"

sepctral_firrst_hidden_dim = 50

r = 25

epochs = 1000

tau_list = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def a_group_test(tau):
    for times in range(repeat_times):
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
            r,
            epochs,
            tau,
        )


if __name__ == "__main__":

    for tau in tau_list:
        a_group_test(tau)
