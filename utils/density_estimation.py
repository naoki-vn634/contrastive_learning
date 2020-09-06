import torch
import numpy as np


def covariance_matrix(emb):
    mean = torch.mean(emb, dim=0, keepdim=True)
    mean_adjust = emb - mean
    cov = torch.matmul(torch.t(mean_adjust), mean_adjust) / (emb.size()[0] - 1)

    return cov, mean


def compute_covar_mean(args, emb, labels):
    """
    Compute Covar and Mean for each classes
    input : Batch * 2048
    """
    for cls in args.cls:
        cls_emb = emb[torch.where(labels == cls)]
        cov, mean = covariance_matrix(cls_emb)

        cov_class = cov if cls == 0 else torch.cat([cov_class, cov], dim=0)
        mean_class = mean if cls == 0 else torch.cat([mean_class, mean], dim=0)

    return cov_class, mean_class


def density_score(args, emb, cov, mean):
    score_classes = torch.tensor([])
    for ind, (cov_, mean_) in enumerate(zip(cov, mean)):  # 各クラス毎に密度推定
        dif = torch.unsqueeze((emb - mean_), 0)
        cov_inv = torch.inverse(cov_)
        left = torch.matmul(torch.matmul(dif, cov_inv), torch.t(dif))
        right = torch.log(
            torch.pow(2 * torch.from_numpy(np.pi), cov_.size()[0]) * torch.det(cov_)
        )

        score_class = -left - right
        score_classes = (
            score_class if ind == 0 else torch.cat([score_classes, score_class], dim=0)
        )

    return torch.argmax(score_classes), torch.max(score_classes)
