import torch
import torch.nn as nn


def cosine_matrix(out0, out1):
    # out_size : (batch * 2208)
    dot = torch.matmul(out0, torch.t(out1))
    norm = torch.matmul(torch.norm(out0, dim=1), torch.norm(out1, dim=1))

    return dot / norm


def mask_matrix(matrix):
    mask = 1 - torch.eye(matrix.size()[0])
    return matrix * mask


def ContrastiveLoss(out0, out1, t=2.0):
    matrix_00 = torch.exp(cosine_matrix(out0, out1) / t)
    matrix_01 = torch.exp(cosine_matrix(out0, out1) / t)
    matrix_11 = torch.exp(cosine_matrix(out1, out1) / t)

    mask_00 = mask_matrix(matrix_00)
    mask_01 = mask_matrix(matrix_01)
    mask_11 = mask_matrix(matrix_11)

    positives = torch.diag(matrix_01, 0)
    negative_0 = torch.sum(mask_01, dim=1) + torch.sum(mask_00, dim=1)
    negative_1 = torch.sum(mask_01, dim=0) + torch.sum(mask_11, dim=1)
    loss = torch.log(negative_1 * negative_0) - torch.log(positives * positives)

    loss = torch.mean(loss)
    return loss
