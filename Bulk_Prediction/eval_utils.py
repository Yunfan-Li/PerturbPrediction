import torch
import numpy as np


# Mean Rowwise Root Mean Squared Error
def MRRMSE(y_true, y_pred):
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)))


# Mean Rowwise Root Mean Squared Error (PyTorch)
def mrrmse_loss(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=1)))


def val(net, val_dataloader):
    val_de_gt, val_de_pred = [], []
    for de, cell_type, sm_name in val_dataloader:
        cell_type = cell_type.cuda()
        sm_name = sm_name.cuda()
        pred = net(cell_type, sm_name)
        pred = pred
        val_de_gt.append(de.numpy())
        val_de_pred.append(pred.detach().cpu().numpy())
    val_de_gt = np.concatenate(val_de_gt, axis=0)
    val_de_pred = np.concatenate(val_de_pred, axis=0)
    mrrmse = MRRMSE(val_de_gt, val_de_pred)
    return mrrmse
