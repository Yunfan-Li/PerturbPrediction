import torch
import misc
import math
import sys
import numpy as np
from torch import nn
from eval_utils import MRRMSE


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    neg_mean,
    neg_std,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    mse_loss = nn.SmoothL1Loss(beta=5)
    neg_mean = neg_mean.to(device)
    neg_std = neg_std.to(device)

    for data_iter_step, (neg_x, neg_sf, type, sm, tgt_rc) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        neg_x = neg_x.to(device, non_blocking=True)
        neg_sf = neg_sf.to(device, non_blocking=True)
        type = type.to(device, non_blocking=True)
        sm = sm.to(device, non_blocking=True)
        tgt_rc = tgt_rc.to(device, non_blocking=True)

        pred = model(neg_x, type, sm)
        # Recover the raw int count
        pred = torch.expm1((pred + neg_x) * neg_std + neg_mean) * neg_sf
        loss_mse = mse_loss(pred, tgt_rc)
        loss = loss_mse

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_mse=loss_mse.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, neg_mean, neg_std):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    print_freq = 10

    neg_mean = neg_mean.to(device)
    neg_std = neg_std.to(device)

    # switch to evaluation mode
    model.eval()
    tgt_count_gt, tgt_count_pred = [], []
    for neg_x, neg_sf, type, sm, tgt_rc in metric_logger.log_every(
        data_loader, print_freq, header
    ):
        neg_x = neg_x.to(device, non_blocking=True)
        neg_sf = neg_sf.to(device, non_blocking=True)
        type = type.to(device, non_blocking=True)
        sm = sm.to(device, non_blocking=True)
        tgt_rc = tgt_rc.to(device, non_blocking=True)

        pred = model(neg_x, type, sm)
        # Recover the raw int count
        pred = torch.expm1((pred + neg_x) * neg_std + neg_mean) * neg_sf
        pred = pred.int()

        tgt_count_gt.append(tgt_rc.cpu().numpy())
        tgt_count_pred.append(pred.cpu().numpy())
    tgt_count_gt = np.concatenate(tgt_count_gt, axis=0)
    tgt_count_pred = np.concatenate(tgt_count_pred, axis=0)

    # Set negative count predictions to 0
    tgt_count_pred[tgt_count_pred < 0] = 0
    mrrmse = MRRMSE(tgt_count_gt.astype(float), tgt_count_pred.astype(float))
    l1 = np.abs(tgt_count_gt - tgt_count_pred).mean()

    metric_logger.meters["mrrmse"].update(mrrmse)
    metric_logger.meters["l1"].update(l1)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
