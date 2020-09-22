import torch
import numpy as np


def covariance_matrix(emb):
    mean = torch.mean(emb, dim=0, keepdim=True)
    mean_adjust = emb - mean
    cov = torch.matmul(torch.t(mean_adjust), mean_adjust) / (emb.size()[0] - 1)
    cov = torch.unsqueeze(cov, 0)
    return cov, mean


def compute_covar_mean(args, emb, labels):
    """
    Compute Covar and Mean for each classes
    input : Batch * 2048
    """
    for cls in range(args.n_cls):
        cls_emb = emb[torch.where(labels == cls)]
        cov, mean = covariance_matrix(cls_emb)

        cov_class = cov if cls == 0 else torch.cat([cov_class, cov], dim=0)
        mean_class = mean if cls == 0 else torch.cat([mean_class, mean], dim=0)

    return cov_class, mean_class


def density_score(args, emb, cov, mean, device, inv=False, norm=False):
    score_classes = np.array([])
    for ind, (cov_, mean_) in enumerate(zip(cov, mean)):  # 各クラス毎に密度推定
        dif = torch.unsqueeze((emb - mean_), 0)
        if inv:
            cov_inv = torch.inverse(cov_)
        else:
            cov_ = torch.diag(torch.diagonal(cov_))
            eps = 1e-10
            cov_inv = torch.diag(1 / (torch.diagonal(cov_) + eps))

        tmp = torch.matmul(dif, cov_inv)
        left = torch.matmul(tmp, torch.t(dif))
        if norm:
            right = torch.log(
                torch.pow(2 * torch.from_numpy(np.pi), cov_.size()[0]) * torch.det(cov_)
            )
            score_classes = -left.cpu().data.numpy() - right.cpu().data.numpy()
        else:
            score_class = -left.cpu().data.numpy()
        score_classes = np.append(score_classes, score_class)

    return np.argmax(score_classes), np.max(score_classes)
