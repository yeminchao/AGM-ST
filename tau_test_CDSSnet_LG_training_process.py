import time
import numpy as np
import sklearn.metrics as sm
import torch
from torch.utils.data import DataLoader
from data_set import DataSet
from itertools import cycle
import scipy.io as sio
from tqdm import tqdm


def training_testing(
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
    r,
    epochs,
    tau,
):

    def AA(m):

        a = np.sum(m, axis=1)
        zuseracclist = []
        for aai in range(m.shape[0]):
            zuseracclist.append(m[aai][aai] / a[aai])
        b = np.average(zuseracclist)
        return b

    source_traing_set_spectral = DataSet(source_train_data_spectral, source_train_label)
    target_train_set_spectral = DataSet(target_train_data_spectral, target_train_label)
    target_test_set_spectral = DataSet(target_test_data_spectral, target_test_label)

    source_train_num = source_train_label.shape[0]
    target_train_num = target_train_label.shape[0]

    source_traing_set_spatial = DataSet(source_train_data_spatial, source_train_label)
    target_train_set_spatial = DataSet(target_train_data_spatial, target_train_label)
    target_test_set_spatial = DataSet(target_test_data_spatial, target_test_label)

    source_train_loader_spectral = DataLoader(
        dataset=source_traing_set_spectral, batch_size=source_train_num, shuffle=False
    )
    target_train_loader_spectral = DataLoader(
        dataset=target_train_set_spectral, batch_size=target_train_num, shuffle=False
    )
    target_test_loader_spectral = DataLoader(
        dataset=target_test_set_spectral, batch_size=1000, shuffle=False
    )

    source_train_loader_spatial = DataLoader(
        dataset=source_traing_set_spatial, batch_size=source_train_num, shuffle=False
    )
    target_train_loader_spatial = DataLoader(
        dataset=target_train_set_spatial, batch_size=target_train_num, shuffle=False
    )
    target_test_loader_spatial = DataLoader(
        dataset=target_test_set_spatial, batch_size=1000, shuffle=False
    )

    source_input_dim = source_train_data_spectral.shape[1]

    net = torch.load(
        "./结果/tau = "
        + str(tau)
        + "/"
        + pretraining_name
        + "/"
        + data_name
        + "/"
        + str(target_num)
        + pretraining_name
        + str(times)
        + ".pkl"
    )

    paramater = sio.loadmat(
        "./结果/tau = "
        + str(tau)
        + "/"
        + "./W_new/"
        + data_name
        + "/"
        + "/all"
        + str(target_num)
        + "_parameter"
        + str(times)
        + ".mat"
    )
    W = paramater["W_ij"]
    W = torch.from_numpy(W)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    criterion_mse, criterion_crossentropy, criterion_LG, criterion_graphloss = (
        net.loss()
    )

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

    for epoch in tqdm(range(epochs)):
        net.train()
        for i, (target_data_spectral, target_label) in enumerate(
            target_train_loader_spectral
        ):
            source_data_spectral, source_label = iter(
                source_train_loader_spectral
            ).next()
            target_data_spatial, _ = iter(target_train_loader_spatial).next()
            source_data_spatial, _ = iter(source_train_loader_spatial).next()

            optimizer.zero_grad()

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

            source_label = source_label.long()
            target_label = target_label.long()

            alpha1 = (torch.norm(source_data_spectral)) ** 2 / (
                torch.norm(target_data_spectral)
            ) ** 2

            alpha2 = (torch.norm(source_hidden_spectral_1)) ** 2 / (
                torch.norm(target_hidden_spectral_1)
            ) ** 2

            lambda_ = 0.001

            theta = 1000

            loss_source_VAE_firstl = (
                criterion_mse(source_input_hat_1, source_data_spectral) + source_kld_1
            )

            loss_target_VAE_firstl = (
                criterion_mse(target_input_hat_1, target_data_spectral) + target_kld_1
            )

            loss_source_VAE_secondl = (
                criterion_mse(source_input_hat_2, source_hidden_spectral_1)
                + source_kld_2
            )

            loss_target_VAE_secondl = (
                criterion_mse(target_input_hat_2, target_hidden_spectral_1)
                + target_kld_2
            )

            loss_spectral_firstl = (
                loss_source_VAE_firstl + loss_target_VAE_firstl * alpha1
            )

            loss_spectral_secondl = (
                loss_source_VAE_secondl + loss_target_VAE_secondl * alpha2
            )

            loss_gr_spectral_spatial = criterion_graphloss(
                source_hidden_spectral_spatial,
                target_hidden_spectral_spatial,
                W,
                class_num,
            )

            loss_gr = (loss_gr_spectral_spatial) * lambda_

            loss_c_source = criterion_crossentropy(source_out, source_label)

            loss_c_target = criterion_crossentropy(
                target_out[-target_labeled_num:, :], target_label[-target_labeled_num:]
            )

            loss_c = (loss_c_source + loss_c_target) * theta

            loss = loss_spectral_firstl + loss_spectral_secondl + loss_gr + loss_c

            loss.backward()

            optimizer.step()

        if ((epoch + 1) % 500 == 0) or ((epoch + 1) == epochs):
            net.eval()
            print("tau:", tau)
            print("times", times)
            print("epoch", epoch + 1, "loss", loss.item())

            net.eval()
            total = 0
            correct = 0
            predlabel = torch.Tensor(np.array([]))
            realtestlabel = torch.Tensor(np.array([]))
            source_data_spatial = torch.ones(
                1, source_input_dim, patch_size, patch_size
            )
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
            with open(data_name + "-" + network_name + "_test_cla_rate.txt", "a") as f:
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
        "Training complete in {:.0f}m {:.0f}s".format(time_last // 60, time_last % 60)
    )

    torch.save(
        net,
        "./结果/tau = "
        + str(tau)
        + "/"
        + network_name
        + "/"
        + data_name
        + "/"
        + str(target_num)
        + network_name
        + str(times)
        + ".pkl",
    )

    return test_acc, aa, kappa
