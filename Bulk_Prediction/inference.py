import torch
import argparse
import numpy as np
import scanpy as sc
from pathlib import Path
from model_utils import Net
from data_utils import prepare_dataset_train_val, prepare_dataset_test


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Training configs
    parser.add_argument("--pre_sm", action="store_true")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=3000, type=int)
    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    return parser


def main(args):
    device = torch.device(args.device)

    (
        dataset_train,
        dataset_val,
        sm_feature,
        cell_type_rna,
        cell_type_atac,
        negative_bulk_mean,
        negative_bulk_std,
    ) = prepare_dataset_train_val(full_train=True)
    (
        dataset_test,
        sm_feature,
        cell_type_rna,
        cell_type_atac,
        train_meta,
        test_meta,
        negative_bulk_mean,
        negative_bulk_std,
    ) = prepare_dataset_test()

    data_loader_val = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    model = Net(
        gene_num=18211,
        compound_num=146,
        sm_feature=sm_feature if args.pre_sm else None,
        type_rna=cell_type_rna,
        type_atac=cell_type_atac,
    )
    model.to(device)
    model.load_state_dict(
        torch.load(args.output_dir + "checkpoint-" + str(args.epochs) + ".pth")["model"]
    )
    model.type_rna = model.type_rna.to(device)
    model.type_atac = model.type_atac.to(device)
    if args.pre_sm:
        model.sm_emb = model.sm_emb.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    model.eval()
    neg_mean = negative_bulk_mean.to(device)
    neg_std = negative_bulk_std.to(device)
    # Val prediction
    tgt_count_gt, tgt_count_pred = [], []
    for neg_x, neg_sf, type, sm, tgt_rc in data_loader_val:
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
    adata = sc.AnnData(tgt_count_pred, obs=train_meta, dtype=int)
    sc.write(args.output_dir + "train_adata_infered.h5ad", adata)

    # Test prediction
    test_count = []
    for neg_x, neg_sf, type, sm in dataloader_test:
        neg_x = neg_x.to(device, non_blocking=True)
        neg_sf = neg_sf.to(device, non_blocking=True)
        type = type.to(device, non_blocking=True)
        sm = sm.to(device, non_blocking=True)

        pred = model(neg_x, type, sm)
        # Recover the raw int count
        pred = torch.expm1((pred + neg_x) * neg_std + neg_mean) * neg_sf
        pred = pred.int()

        test_count.append(pred.detach().cpu().numpy())
    test_count = np.concatenate(test_count, axis=0)
    # Set negative count predictions to 0
    test_count[test_count < 0] = 0
    adata = sc.AnnData(test_count, obs=test_meta, dtype=int)
    print(adata.X.min(), adata.X.max(), adata.X.mean())
    sc.write(args.output_dir + "test_adata.h5ad", adata)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
