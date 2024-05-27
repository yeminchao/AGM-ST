from scipy.io import loadmat
import numpy as np
from itertools import cycle
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as sm

from data_set import DataSet
from data_pre import TDSSnet_data_pre, TDSSnet_data_pre_all_test


# Load data

patch_size = 9

repeat_times = 10

data_name = "RPaviaU-DPaviaC"


def AA(m):

    a = np.sum(m, axis=1)
    zuseracclist = []
    for aai in range(m.shape[0]):
        zuseracclist.append(m[aai][aai] / a[aai])
    b = np.average(zuseracclist)
    return b


if __name__ == "__main__":

    OAacc_ave = 0.0
    AAacc_ave = 0.0
    kappa_ave = 0.0

    hidden_dim = 50
    r = 20

    for times in range(repeat_times):
        (
            target_train_data_spectral,
            target_test_data_spectral,
            target_train_data_spatial,
            target_test_data_spatial,
            target_train_label,
            target_test_label,
            class_num,
            target_labeled_num,
        ) = TDSSnet_data_pre(
            initial_target_data, initial_target_label, data_name, times, patch_size
        )

        target_test_set_spectral = DataSet(target_test_data_spectral, target_test_label)

        target_train_num = target_train_label.shape[0]
        target_test_num = target_test_label.shape[0]

        target_test_set_spatial = DataSet(target_test_data_spatial, target_test_label)

        target_test_loader_spectral = DataLoader(
            dataset=target_test_set_spectral, batch_size=10000, shuffle=False
        )

        target_test_loader_spatial = DataLoader(
            dataset=target_test_set_spatial, batch_size=10000, shuffle=False
        )

        net = torch.load(
            "./TDSSnet/" + data_name + "/r" + str(r) + "/model" + str(times) + ".pkl"
        )
        net.eval()
        total = 0
        correct = 0
        predlabel = torch.Tensor(np.array([]))
        realtestlabel = torch.Tensor(np.array([]))
        with torch.no_grad():
            for _, spectral_spatial_data in enumerate(
                zip(target_test_loader_spectral, cycle(target_test_loader_spatial))
            ):
                target_data_spectral, target_label = (
                    spectral_spatial_data[0][0],
                    spectral_spatial_data[0][1],
                )
                target_data_spatial, _ = (
                    spectral_spatial_data[1][0],
                    spectral_spatial_data[1][1],
                )

                (
                    target_hidden_1,
                    target_input_hat_1,
                    target_kld_1,
                    target_hidden_spectral_spatial,
                    target_input_hat_2,
                    target_kld_2,
                    target_out,
                ) = net(target_data_spectral, target_data_spatial)

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
