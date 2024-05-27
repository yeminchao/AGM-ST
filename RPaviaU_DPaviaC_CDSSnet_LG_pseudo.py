from scipy.io import loadmat
from CDSSnet_LG_training_process_pseudo import training_testing
from data_pre import CDSSnet_LG_data_pre


# Load data

patch_size = 9

repeat_times = 10

data_name = "RPaviaU-DPaviaC"

network_name = "CDSSnet-LG"

pretraining_name = "LGnet"

target_num = 400

OAacc_ave = 0.0
AAacc_ave = 0.0
kappa_ave = 0.0

r_list = [25]

epochs = 1000

sepctral_firrst_hidden_dim = 50


def a_group_test(r, OAacc_ave, AAacc_ave, kappa_ave):
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
            target_num,
            target_labeled_num,
            sepctral_firrst_hidden_dim,
            r,
            epochs,
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
        + str(OAacc_ave)
        + ", AAaccuracy: "
        + str(AAacc_ave)
        + ", kappa: "
        + str(kappa_ave)
        + " after "
        + str(repeat_times)
        + " averages"
    )
    print("-------------------")
    print("finish!")


if __name__ == "__main__":

    for r in r_list:
        r = 25
        a_group_test(r, OAacc_ave, AAacc_ave, kappa_ave)
