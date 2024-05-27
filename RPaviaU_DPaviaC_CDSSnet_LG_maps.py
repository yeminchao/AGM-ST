from scipy.io import loadmat
import numpy as np
from itertools import cycle
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as sm

from data_set import DataSet
from data_pre import CDSSnet_LG_data_pre, CDSSnet_LG_data_pre_all_test


# Load data

patch_size = 9

repeat_times = 10

data_name = "RPaviaU-DPaviaC"
network_name = "CDSSnet-LG"
pretraining_name = "LGnet"
target_num = 400


def AA(m):

    a = np.sum(m, axis=1)
    zuseracclist = []
    for aai in range(m.shape[0]):
        zuseracclist.append(m[aai][aai] / a[aai])
    b = np.average(zuseracclist)
    return b


if __name__ == "__main__":

    colors = ["white", "green", "#9a6200", "#75bbfd", "r", "#751973", "grey", "black"]
    cmap = mpl.colors.ListedColormap(colors)

    hidden_dim = 50
    r = 25

    OAacc_ave = 0.0
    AAacc_ave = 0.0
    kappa_ave = 0.0
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
            target_labeled_num,
        ) = CDSSnet_LG_data_pre_all_test(
            initial_source_data,
            initial_target_data,
            initial_source_label,
            initial_target_label,
            data_name,
            times,
            patch_size,
        )

        target_test_set_spectral = DataSet(target_test_data_spectral, target_test_label)

        target_test_num = target_test_label.shape[0]

        target_test_set_spatial = DataSet(target_test_data_spatial, target_test_label)

        target_test_loader_spectral = DataLoader(
            dataset=target_test_set_spectral, batch_size=1000, shuffle=False
        )

        target_test_loader_spatial = DataLoader(
            dataset=target_test_set_spatial, batch_size=1000, shuffle=False
        )

        source_input_dim = source_train_data_spectral.shape[1]
        target_input_dim = target_train_data_spectral.shape[1]

        net = torch.load(
            "./"
            + network_name
            + "/"
            + data_name
            + "/r"
            + str(r)
            + "/model"
            + str(times)
            + ".pkl"
        )

        net.eval()
        total = 0
        correct = 0
        predlabel = torch.Tensor(np.array([]))
        realtestlabel = torch.Tensor(np.array([]))
        source_data_spatial = torch.ones(1, source_input_dim, patch_size, patch_size)
        source_data_spectral = torch.ones(1, source_input_dim)
        with torch.no_grad():
            for i, data in enumerate(
                zip(target_test_loader_spectral, cycle(target_test_loader_spatial))
            ):
                target_data_spectral, target_label = data[0][0], data[0][1]
                target_data_spatial, _ = data[1][0], data[1][1]

                (
                    source_hidden_spectral_1,
                    source_input_hat_1,
                    source_kld_1,
                    source_hidden_spectral_spatial,
                    source_out_spatial,
                    source_hidden_spectral_2,
                    source_input_hat_2,
                    source_kld_2,
                    target_hidden_spectral_1,
                    target_input_hat_1,
                    target_kld_1,
                    target_hidden_spectral_spatial,
                    target_out_spatial,
                    target_hidden_spectral_2,
                    target_input_hat_2,
                    target_kld_2,
                    rho,
                    source_out,
                    target_out,
                ) = net(
                    source_data_spectral,
                    target_data_spectral,
                    source_data_spatial,
                    target_data_spatial,
                )

                i, predicted = torch.max(target_out.data, dim=1)
                predlabel = torch.cat((predlabel, predicted), -1)
                realtestlabel = torch.cat((realtestlabel, target_label), -1)
                total += target_label.size(0)
                correct += (predicted == target_label).sum()
            test_acc = float(correct) / total
            predlabel = predlabel + 1
            realtestlabel = realtestlabel + 1

            C = sm.confusion_matrix(
                realtestlabel.data.cpu().numpy(), predlabel.cpu().numpy()
            )
            kappa = sm.cohen_kappa_score(
                realtestlabel.data.cpu().numpy(),
                predlabel.cpu().numpy(),
                labels=None,
                weights=None,
                sample_weight=None,
            )
            aa = AA(C)

        OAacc_ave = test_acc + OAacc_ave
        AAacc_ave = aa + AAacc_ave
        kappa_ave = kappa + kappa_ave

        print("OA", test_acc, "AA", aa, "kappa", kappa)

        pred_array = np.array(predlabel)
        new_show = np.zeros(
            (initial_target_label.shape[0], initial_target_label.shape[1])
        )
        k = 0
        for i in range(initial_target_label.shape[0]):
            for j in range(initial_target_label.shape[1]):
                if initial_target_label[i][j] != 0:
                    new_show[i][j] = pred_array[k]
                    k += 1
        ground_predict = plt.imshow(new_show, interpolation="none", cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        foo_fig = plt.gcf()
        foo_fig.savefig(
            "./maps_CDSSnet-LG/" + data_name + "/" + data_name + "-AGM-ST.png",
            format="png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.07,
        )
        foo_fig.savefig(
            "./maps_CDSSnet-LG/" + data_name + "/" + data_name + "-AGM-ST.eps",
            format="eps",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.07,
        )
        plt.show()

        new_show_patch = new_show[50:150, 50:150]
        uni = np.unique(new_show_patch)
        new_colors = colors[: len(uni) + 1]
        cmap_patch = mpl.colors.ListedColormap(new_colors)
        ground_predict = plt.imshow(
            new_show_patch, interpolation="none", cmap=cmap_patch
        )
        plt.xticks([])
        plt.yticks([])
        foo_fig = plt.gcf()
        foo_fig.savefig(
            "./maps_CDSSnet-LG/" + data_name + "/" + data_name + "-AGM-ST_patch.png",
            format="png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.show()

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
