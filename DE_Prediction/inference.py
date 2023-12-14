import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from model_utils import Net
from data_utils import prepare_dataset_train_val, prepare_dataset_test


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Training configs
    parser.add_argument("--pre_type", action="store_true")
    parser.add_argument("--pre_sm", action="store_true")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=1950, type=int)
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
        cell_type_rna_count,
        sm_feature,
    ) = prepare_dataset_train_val(full_train=True)
    dataset_test = prepare_dataset_test()
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    model = Net(
        type_num=6,
        compound_num=146,
        gene_num=18211,
        cell_type_rna_count=cell_type_rna_count if args.pre_type else None,
        sm_feature=sm_feature if args.pre_sm else None,
    )
    model.to(device)
    model.load_state_dict(
        torch.load(args.output_dir + "checkpoint-" + str(args.epochs) + ".pth")["model"]
    )
    if args.pre_type:
        model.type_emb = model.type_emb.to(device)
    if args.pre_sm:
        model.sm_emb = model.sm_emb.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    model.eval()

    # Test prediction
    test_de = []
    for cell_type, sm_name in dataloader_test:
        cell_type = cell_type.to(device, non_blocking=True)
        sm_name = sm_name.to(device, non_blocking=True)
        pred = model(cell_type, sm_name)
        test_de.append(pred.detach().cpu().numpy())
    test_de = np.concatenate(test_de, axis=0)
    sample_submission = pd.read_csv(
        "../data/sample_submission.csv", index_col=0, header=0
    )
    submit_submission = pd.DataFrame(
        test_de,
        index=sample_submission.index,
        columns=sample_submission.columns,
    )
    print(submit_submission)
    submit_submission.to_csv("../data/submission.csv")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
