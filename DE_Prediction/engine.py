import torch
import misc
import math
import sys
import numpy as np
from eval_utils import MRRMSE, mrrmse_loss


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (de, type, sm) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        de = de.to(device, non_blocking=True)
        type = type.to(device, non_blocking=True)
        sm = sm.to(device, non_blocking=True)

        pred = model(type, sm)
        pred_loss = mrrmse_loss(pred, de)

        loss = pred_loss

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

        metric_logger.update(pred_loss=pred_loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    print_freq = 10

    # switch to evaluation mode
    model.eval()
    de_gt, de_pred = [], []
    for de, type, sm in metric_logger.log_every(data_loader, print_freq, header):
        de = de.to(device, non_blocking=True)
        type = type.to(device, non_blocking=True)
        sm = sm.to(device, non_blocking=True)

        pred = model(type, sm)

        de_gt.append(de.cpu().numpy())
        de_pred.append(pred.cpu().numpy())
    de_gt = np.concatenate(de_gt, axis=0)
    de_pred = np.concatenate(de_pred, axis=0)
    mrrmse = MRRMSE(de_gt, de_pred)

    metric_logger.meters["mrrmse"].update(mrrmse)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
