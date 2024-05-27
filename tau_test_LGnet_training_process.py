import time
import torch
from torch.utils.data import DataLoader
from data_set import DataSet
import scipy.io as sio
from loss import (
    discriminative_adjacent_matrix_LG,
    my_adjacent_matrix,
    Loss_B,
    ED_square,
)
from model import CDSSnet_LG
from tqdm import tqdm


def training(
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
):

    source_traing_set = DataSet(source_train_data_spectral, source_train_label)
    target_train_set = DataSet(target_train_data_spectral, target_train_label)
    target_test_set = DataSet(target_test_data_spectral, target_test_label)

    source_train_num = source_train_label.shape[0]
    target_train_num = target_train_label.shape[0]
    target_test_num = target_test_label.shape[0]

    source_traing_set_spatial = DataSet(source_train_data_spatial, source_train_label)
    target_train_set_spatial = DataSet(target_train_data_spatial, target_train_label)
    target_test_set_spatial = DataSet(target_test_data_spatial, target_test_label)

    source_train_loader_spectral = DataLoader(
        dataset=source_traing_set, batch_size=source_train_num, shuffle=False
    )
    target_train_loader_spectral = DataLoader(
        dataset=target_train_set, batch_size=target_train_num, shuffle=False
    )
    target_test_loader_spectral = DataLoader(
        dataset=target_test_set, batch_size=target_test_num, shuffle=False
    )

    source_train_loader_spatial = DataLoader(
        dataset=source_traing_set_spatial, batch_size=source_train_num, shuffle=False
    )
    target_train_loader_spatial = DataLoader(
        dataset=target_train_set_spatial, batch_size=target_train_num, shuffle=False
    )
    target_test_loader_spatial = DataLoader(
        dataset=target_test_set_spatial, batch_size=target_test_num, shuffle=False
    )

    source_input_dim = source_train_data_spectral.shape[1]
    target_input_dim = target_train_data_spectral.shape[1]

    net = CDSSnet_LG(
        source_input_dim,
        target_input_dim,
        patch_size,
        sepctral_firrst_hidden_dim,
        r,
        class_num,
    )

    for i, (target_inputs, target_labels) in enumerate(target_train_loader_spectral):
        source_inputs, source_labels = iter(source_train_loader_spectral).next()
        W_ij_same, W_ij_different = discriminative_adjacent_matrix_LG(
            source_inputs, target_inputs, source_labels, target_labels
        )
        W_ij_same = W_ij_same
        W_ij_different = W_ij_different

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.1)
    criterion_mse, criterion_crossentropy, criterion_LG, criterion_graphloss = (
        net.loss()
    )

    since = time.time()
    net.train()
    for epoch in tqdm(range(epochs)):
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

            gamma = 1000

            loss_source_spectral_firstl = (
                criterion_mse(source_input_hat_1, source_data_spectral) + source_kld_1
            )

            loss_target_spectral_firstl = (
                criterion_mse(target_input_hat_1, target_data_spectral) + target_kld_1
            )

            loss_source_spectral_secondl = (
                criterion_mse(source_input_hat_2, source_hidden_spectral_1)
                + source_kld_2
            )

            loss_target_spectral_secondl = (
                criterion_mse(target_input_hat_2, target_hidden_spectral_1)
                + target_kld_2
            )

            loss_spectral_firstl = (
                loss_source_spectral_firstl + loss_target_spectral_firstl * alpha1
            )

            loss_spectral_secondl = (
                loss_source_spectral_secondl + loss_target_spectral_secondl * alpha2
            )

            loss_spectral = loss_spectral_firstl + loss_spectral_secondl

            loss_gr_spectral_spatial, Wij = criterion_LG(
                source_hidden_spectral_spatial,
                target_hidden_spectral_spatial,
                W_ij_same,
                W_ij_different,
                class_num,
            )

            loss_gr = (loss_gr_spectral_spatial) * lambda_

            distance_spectral_spatial = ED_square(
                source_hidden_spectral_spatial, target_hidden_spectral_spatial
            ).detach()

            loss_B_spectral_spatial = Loss_B(
                tau, (-rho * distance_spectral_spatial).exp(), Wij
            )

            loss_B = loss_B_spectral_spatial * gamma

            loss_c_source = criterion_crossentropy(source_out, source_label)

            loss_c_target = criterion_crossentropy(target_out, target_label)

            loss_c = (loss_c_source + loss_c_target) * theta

            loss = loss_spectral + loss_gr + loss_B + loss_c

            loss.backward()

            optimizer.step()

        if (epoch + 1) % 500 == 0:
            print("tau:", tau)
            print("times", times)
            print("epoch:", epoch + 1)

    time_last = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s with {}".format(
            time_last // 60, time_last % 60, times
        )
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
        + str(target_test_num)
        + network_name
        + str(times)
        + ".pkl",
    )

    net = torch.load(
        "./结果/tau = "
        + str(tau)
        + "/"
        + network_name
        + "/"
        + data_name
        + "/"
        + str(target_test_num)
        + network_name
        + str(times)
        + ".pkl"
    )

    net.eval()
    for i, (target_data_spectral, target_label) in enumerate(
        target_test_loader_spectral
    ):
        source_data_spectral, source_label = iter(source_train_loader_spectral).next()
        target_data_spatial, _ = iter(target_test_loader_spatial).next()
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

        new_distance = my_adjacent_matrix(
            source_hidden_spectral_spatial, target_hidden_spectral_spatial
        )
        new_W = (-rho * new_distance).exp() - tau
        new_W[new_W < 0] = 0
        print(new_W, new_W.shape)
        print(torch.sum(new_W).item())
        print(source_label, target_label)

    print("my_W_ij已得到")
    sio.savemat(
        "./结果/tau = "
        + str(tau)
        + "/"
        + "./W_new/"
        + data_name
        + "/"
        + "/all"
        + str(target_test_num)
        + "_parameter"
        + str(times)
        + ".mat",
        {"rho": rho.item(), "W_ij": new_W.cpu().detach().numpy()},
    )
