import torch
import torch.nn as nn
import numpy as np


def ED_square(A, B):

    BT = B.t()
    vecProd = A.mm(BT)
    SqareA = A**2
    sumSqareA = (torch.sum(SqareA, axis=1)).view(1, A.size(0))
    sumSqareA = sumSqareA.t()
    sumSqareAEx = sumSqareA.repeat(1, vecProd.size(1))

    SqareB = B**2
    sumSqareB = (torch.sum(SqareB, axis=1)).view(1, B.size(0))
    sumSqareBEx = sumSqareB.repeat(vecProd.size(0), 1)
    SqareED = sumSqareBEx + sumSqareAEx - 2 * vecProd
    return SqareED


def cosine_distance(matrix1, matrix2):
    matrix1_matrix2 = torch.mm(matrix1, matrix2.T)
    matrix1_norm = torch.sqrt(torch.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm.unsqueeze(1)
    matrix2_norm = torch.sqrt(torch.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm.unsqueeze(1)
    cosine_distance = torch.div(matrix1_matrix2, torch.mm(matrix1_norm, matrix2_norm.T))
    return cosine_distance


class second_stage_graph_loss(nn.Module):

    def __init__(self):
        super(second_stage_graph_loss, self).__init__()

    def forward(self, source_data, target_data, W, class_num):
        distance_square = ED_square(source_data, target_data)

        loss = torch.sum(distance_square * W)

        return loss


def discriminative_adjacent_matrix_LG(X_source, X_target, y_source, y_target):

    X_source = X_source.view(X_source.shape[0], -1)
    X_target = X_target.view(X_target.shape[0], -1)
    W_ij_same = torch.zeros((X_source.shape[0], X_target.shape[0]))
    W_ij_different = torch.zeros((X_source.shape[0], X_target.shape[0]))
    for source_num in range(X_source.shape[0]):
        for target_num in range(X_target.shape[0]):
            if y_source[source_num] == y_target[target_num]:
                W_ij_same[source_num][target_num] = 1
            else:
                W_ij_same[source_num][target_num] = 0

    for source_num in range(X_source.shape[0]):
        for target_num in range(X_target.shape[0]):
            if y_source[source_num] == y_target[target_num]:
                W_ij_different[source_num][target_num] = 0
            else:
                W_ij_different[source_num][target_num] = 1
    return W_ij_same, W_ij_different


class graph_modeling_loss(nn.Module):
    def __init__(self):
        super(graph_modeling_loss, self).__init__()

    def forward(self, X_source, X_target, W_ij_same, W_ij_different, class_num):
        beta = 0.01

        distance_sq = ED_square(X_source, X_target)

        n_same = X_source.shape[0] * X_target.shape[0] / class_num
        n_different = (
            X_source.shape[0] * X_target.shape[0] * (class_num - 1) / class_num
        )

        Wst_same = pow(1 / n_same, 0.5) * W_ij_same
        Wst_different = pow(1 / n_different, 0.5) * W_ij_different
        Wij = Wst_same + Wst_different

        loss = torch.sum(distance_sq * Wst_same) - beta * torch.sum(
            distance_sq * Wst_different
        )

        return loss, Wij


def my_adjacent_matrix(X_source, X_target):

    X_source = X_source.view(X_source.shape[0], -1)
    X_target = X_target.view(X_target.shape[0], -1)
    my_W_ij = ED_square(X_source, X_target)

    return my_W_ij


def Loss_B(tau, cos_similarity, Wij):
    loss = torch.sum((Wij * (cos_similarity - tau)) ** 2)

    return loss


class pseudo_labeled_graph_loss(nn.Module):

    def __init__(self):
        super(second_stage_graph_loss, self).__init__()

    def forward(self, source_data, target_data, W, class_num):
        distance_square = ED_square(source_data, target_data)
        n = source_data.shape[0] * target_data.shape[0] / class_num
        W_norm = (1 / n) ** 0.5 * W
        loss = torch.sum(distance_square * W_norm)

        return loss
