from scipy.io import loadmat
from CDSSnet_LG_training_process import training_testing
from data_pre import CDSSnet_LG_data_pre


# Load data

patch_size = 9

repeat_times = 1

data_name = "RPaviaU-DPaviaC"

network_name = "CDSSnet-LG"

pretraining_name = "LGnet"

graph_modeling_num = 400

OAacc_ave = 0.0
AAacc_ave = 0.0
kappa_ave = 0.0

r_list = [10]


epochs = 1000

pseudo_number_per_stage = 100


def a_group_test(r, OAacc_ave, AAacc_ave, kappa_ave):
    for times in range(repeat_times):
        times = 1
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
        ) = CDSSnet_LG_data_pre(
            initial_source_data,
            initial_target_data,
            initial_source_label,
            initial_target_label,
            data_name,
            times,
            patch_size,
        )

        test_acc, aa, kappa = training_testing(
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
            pretraining_name,
            graph_modeling_num,
            target_labeled_num,
            r,
            epochs,
            pseudo_number_per_stage,
        )

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
        break
