from scipy.io import loadmat
import time
import numpy as np
import sklearn.metrics as sm
import torch
from torch.utils.data import DataLoader
from data_set import DataSet
from model import TDSSnet
from data_pre import TDSSnet_data_pre

# Load data


patch_size = 9

repeat_times = 10

data_name = "RPaviaU-DPaviaC"
network_name = "TDSSnet"


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
    r = 5
    epochs = 1000
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

        target_train_set = DataSet(target_train_data_spectral, target_train_label)
        target_test_set = DataSet(target_test_data_spectral, target_test_label)

        target_train_num = target_train_label.shape[0]
        target_test_num = target_test_label.shape[0]

        target_train_set_spatial = DataSet(
            target_train_data_spatial, target_train_label
        )
        target_test_set_spatial = DataSet(target_test_data_spatial, target_test_label)

        target_train_loader = DataLoader(
            dataset=target_train_set, batch_size=target_train_num, shuffle=False
        )
        target_test_loader = DataLoader(
            dataset=target_test_set, batch_size=target_test_num, shuffle=False
        )

        target_train_loader_spatial = DataLoader(
            dataset=target_train_set_spatial, batch_size=target_train_num, shuffle=False
        )
        target_test_loader_spatial = DataLoader(
            dataset=target_test_set_spatial, batch_size=target_test_num, shuffle=False
        )

        target_input_dim = target_train_data_spectral.shape[1]

        net = TDSSnet(target_input_dim, patch_size, hidden_dim, r, class_num)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        criterion_mse, criterion_crossentropy = net.loss()

        since = time.time()
        with open(data_name + "-" + network_name + "_test_cla_rate.txt", "a") as f:
            f.write(
                "\n\n"
                + network_name
                + " r"
                + str(r)
                + "  "
                + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
                + " The "
                + str(times)
                + " time"
                + "\n"
            )
        f.close()

        for epoch in range(epochs):
            net.train()
            for i, (target_data_spectral, target_label) in enumerate(
                target_train_loader
            ):
                target_data_spatial, _ = iter(target_train_loader_spatial).next()

                optimizer.zero_grad()

                (
                    target_hidden_1,
                    target_input_hat_1,
                    target_kld_1,
                    target_hidden_spectral_spatial,
                    target_input_hat_2,
                    target_kld_2,
                    target_out,
                ) = net(target_data_spectral, target_data_spatial)

                target_label = target_label.long()

                theta = 1000

                loss_target_firstl = (
                    criterion_mse(target_input_hat_1, target_data_spectral)
                    + target_kld_1
                )

                loss_target_secondl = (
                    criterion_mse(target_input_hat_2, target_hidden_1) + target_kld_2
                )

                loss_vae_firstl = loss_target_firstl

                loss_vae_secondl = loss_target_secondl

                loss_fe_firstl = loss_vae_firstl

                loss_fe_secondl = loss_vae_secondl

                loss_c_target = criterion_crossentropy(
                    target_out[-target_labeled_num:, :],
                    target_label[-target_labeled_num:],
                )

                loss_c = loss_c_target * theta

                loss = loss_fe_firstl + loss_fe_secondl + loss_c

                loss.backward()

                optimizer.step()

                print(epoch)

            if (epoch + 1) % 100 == 0:
                print("epoch", epoch + 1, "loss", loss.item())

                net.eval()
                total = 0
                correct = 0
                predlabel = torch.Tensor(np.array([]))
                realtestlabel = torch.Tensor(np.array([]))
                with torch.no_grad():
                    for target_data_spectral, target_label in target_test_loader:
                        target_data_spatial, _ = iter(target_test_loader_spatial).next()

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
                with open(
                    data_name + "-" + network_name + "_test_cla_rate.txt", "a"
                ) as f:
                    f.write(
                        str(epoch + 1)
                        + "  OA: "
                        + str(test_acc)
                        + "  AA: "
                        + str(aa)
                        + "  kappa: "
                        + str(kappa)
                        + "\n"
                    )
                f.close()

        time_last = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_last // 60, time_last % 60
            )
        )

        torch.save(
            net,
            "./TDSSnet/" + data_name + "/r" + str(r) + "/model" + str(times) + ".pkl",
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
