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
    graph_modeling_num,
    target_labeled_num,
    r,
    epochs,
    pseudo_number_per_stage,
    tag,
    LG_labeled_idx,
    pseudoed_idx,
    round,
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
    target_input_dim = target_train_data_spectral.shape[1]

    net = torch.load(
        "./"
        + pretraining_name
        + "/"
        + data_name
        + "/r"
        + str(r)
        + "/"
        + str(graph_modeling_num)
        + pretraining_name
        + str(times)
        + ".pkl"
    )

    target_add_pseudo_loader_spectral = torch.load(
        "./"
        + pretraining_name
        + "/"
        + data_name
        + "/r"
        + str(r)
        + "/"
        + str(graph_modeling_num)
        + "target_add_pseudo_loader_spectral_first_stage_save"
        + str(times)
        + ".pkl"
    )

    target_add_pseudo_loader_spatial = torch.load(
        "./"
        + pretraining_name
        + "/"
        + data_name
        + "/r"
        + str(r)
        + "/"
        + str(graph_modeling_num)
        + "target_add_pseudo_loader_spatial_first_stage_save"
        + str(times)
        + ".pkl"
    )

    paramater = sio.loadmat(
        "./W_new/"
        + data_name
        + "/r"
        + str(r)
        + "/"
        + "all400_parameter"
        + str(times)
        + ".mat"
    )
    W = paramater["W_ij"]
    W = torch.from_numpy(W)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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
            + " Round "
            + str(round)
            + "\n"
        )
    f.close()

    for epoch in tqdm(range(epochs), mininterval=5):
        net.train()
        for i, (target_data_spectral, target_label) in enumerate(
            target_train_loader_spectral
        ):
            source_data_spectral, source_label = iter(
                source_train_loader_spectral
            ).next()
            target_data_spatial, _ = iter(target_train_loader_spatial).next()
            source_data_spatial, _ = iter(source_train_loader_spatial).next()

            target_c_spectral, target_label_c = iter(
                target_add_pseudo_loader_spectral
            ).next()
            target_c_spatial, _ = iter(target_add_pseudo_loader_spatial).next()

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
                tau,
                source_out,
                target_out,
            ) = net(
                source_data_spectral,
                target_data_spectral,
                source_data_spatial,
                target_data_spatial,
            )

            _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, target_out_c = net(
                source_data_spectral,
                target_c_spectral,
                source_data_spatial,
                target_c_spatial,
            )

            source_label = source_label.long()

            target_label_c = target_label_c.long()

            alpha1 = (torch.norm(source_data_spectral)) ** 2 / (
                torch.norm(target_data_spectral)
            ) ** 2

            alpha2 = (torch.norm(source_hidden_spectral_1)) ** 2 / (
                torch.norm(target_hidden_spectral_1)
            ) ** 2

            lambda_ = 0.001
            lambda_ = 0.000001

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

            loss_c_target = criterion_crossentropy(target_out_c, target_label_c)

            loss_c = (loss_c_source + loss_c_target) * theta

            loss = loss_spectral_firstl + loss_spectral_secondl + loss_gr + loss_c

            loss.backward()

            optimizer.step()

        if ((epoch + 1) % 500 == 0) or ((epoch + 1) == epochs):
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
                        tau,
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

    net.eval()
    for i, (target_data_spectral, target_label) in enumerate(
        target_train_loader_spectral
    ):
        source_data_spectral, source_label = iter(source_train_loader_spectral).next()
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
            tau,
            source_out,
            target_out,
        ) = net(
            source_data_spectral,
            target_data_spectral,
            source_data_spatial,
            target_data_spatial,
        )

    target_softmax_max_scores, target_pseudo_label = torch.max(target_out.data, dim=1)

    sorted_scores, sorted_indices = torch.sort(
        target_softmax_max_scores, descending=True
    )

    delete_array = np.append(LG_labeled_idx, pseudoed_idx).astype(int)
    for i in delete_array:
        sorted_indices = np.delete(
            sorted_indices, np.where(sorted_indices == i), axis=0
        )

    if sorted_indices.shape[0] >= pseudo_number_per_stage:
        sorted_indices = sorted_indices[:pseudo_number_per_stage]
    else:
        sorted_indices = sorted_indices
        tag = 1

    pseudoed_idx = np.append(pseudoed_idx, sorted_indices)

    for i, (target_add_pseudo_data_spectral, target_add_pseudo_label) in enumerate(
        target_add_pseudo_loader_spectral
    ):
        target_add_pseudo_data_spatial, _ = iter(
            target_add_pseudo_loader_spatial
        ).next()

    print(target_add_pseudo_data_spectral.shape)
    print(target_add_pseudo_data_spatial.shape)
    print(target_add_pseudo_label.shape)
    add_pseudo_spectral = target_train_data_spectral[sorted_indices, :]
    add_pseudo_spatial = target_train_data_spatial[sorted_indices, :]
    add_pseudo_label = target_pseudo_label[sorted_indices]

    target_add_pseudo_data_spectral = np.append(
        target_add_pseudo_data_spectral, add_pseudo_spectral, axis=0
    )
    target_add_pseudo_data_spatial = np.append(
        target_add_pseudo_data_spatial, add_pseudo_spatial, axis=0
    )
    target_add_pseudo_label = np.append(
        target_add_pseudo_label, add_pseudo_label, axis=0
    )
    print(target_add_pseudo_data_spectral.shape)
    print(target_add_pseudo_data_spatial.shape)
    print(target_add_pseudo_label.shape)

    target_train_set_spectral = DataSet(
        target_add_pseudo_data_spectral, target_add_pseudo_label
    )

    target_add_pseudo_train_num = target_add_pseudo_label.shape[0]

    target_train_set_spatial = DataSet(
        target_add_pseudo_data_spatial, target_train_label
    )

    target_add_pseudo_loader_spectral = DataLoader(
        dataset=target_train_set_spectral,
        batch_size=target_add_pseudo_train_num,
        shuffle=False,
    )

    target_add_pseudo_loader_spatial = DataLoader(
        dataset=target_train_set_spatial,
        batch_size=target_add_pseudo_train_num,
        shuffle=False,
    )

    torch.save(
        target_add_pseudo_loader_spectral,
        "./"
        + pretraining_name
        + "/"
        + data_name
        + "/r"
        + str(r)
        + "/"
        + str(target_train_num)
        + "target_add_pseudo_loader_spectral_second_stage_save"
        + str(times)
        + ".pkl",
    )
    torch.save(
        target_add_pseudo_loader_spatial,
        "./"
        + pretraining_name
        + "/"
        + data_name
        + "/r"
        + str(r)
        + "/"
        + str(target_train_num)
        + "target_add_pseudo_loader_spatial_second_stage_save"
        + str(times)
        + ".pkl",
    )

    time_last = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(time_last // 60, time_last % 60)
    )

    import os

    root = "."
    save_dirs = root + "/" + network_name + "/" + data_name + "/r" + str(r)

    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)

    save_name = save_dirs + "/model" + str(times) + ".pkl"

    torch.save(net, save_name)

    return test_acc, aa, kappa, tag, pseudoed_idx
