from scipy.io import loadmat
from data_pre import LGnet_data_pre, CDSSnet_LG_data_pre
from LGnet_training_process import training
from CDSSnet_LG_training_process import training_testing
import numpy as np

# Load data

patch_size = 9

repeat_times = 1

data_name = "RPaviaU-DPaviaC"

first_stage_network_name = "LGnet"

second_stage_network_name = "CDSSnet-LG"

sepctral_first_hidden_dim = 50

r_list = [25]

epochs = 1000

graph_modeling_num = 400

OAacc_ave = 0.0
AAacc_ave = 0.0
kappa_ave = 0.0

pseudo_number_per_stage = 100


def a_group_test(r, OAacc_ave, AAacc_ave, kappa_ave):
    for times in range(repeat_times):
        times = 3
        tag = 0
        LG_all_edge_num = 400
        LG_labeled_idx = np.arange(LG_all_edge_num - 35, LG_all_edge_num)
        pseudoed_idx = np.array([])
        round = 1
        while tag == 0:
            print("r ", r, "times ", times, "round ", round)
            print("The first stage of AGM-ST")
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
                first_stage_network_name,
                sepctral_first_hidden_dim,
                target_labeled_num,
                r,
                epochs,
                round,
            )

            print("The second stage of AGM-ST")

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
            ) = CDSSnet_LG_data_pre(
                initial_source_data,
                initial_target_data,
                initial_source_label,
                initial_target_label,
                data_name,
                times,
                patch_size,
            )

            test_acc, aa, kappa, tag, pseudoed_idx = training_testing(
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
                second_stage_network_name,
                first_stage_network_name,
                graph_modeling_num,
                target_labeled_num,
                r,
                epochs,
                pseudo_number_per_stage,
                tag,
                LG_labeled_idx,
                pseudoed_idx,
                round,
            )
            round += 1

        OAacc_ave = test_acc + OAacc_ave
        AAacc_ave = aa + AAacc_ave
        kappa_ave = kappa + kappa_ave

    OAacc_ave = OAacc_ave / repeat_times
    AAacc_ave = AAacc_ave / repeat_times
    kappa_ave = kappa_ave / repeat_times
    print(
        str(r)
        + "-dimensional OAaccuracy: "
        + str(format(OAacc_ave, ".4f"))
        + ", AAaccuracy: "
        + str(format(AAacc_ave, ".4f"))
        + ", kappa: "
        + str(format(kappa_ave, ".4f"))
        + " after "
        + str(repeat_times)
        + " averages"
    )
    print("-------------------")
    print("finish!")


if __name__ == "__main__":

    for r in r_list:
        a_group_test(r, OAacc_ave, AAacc_ave, kappa_ave)
