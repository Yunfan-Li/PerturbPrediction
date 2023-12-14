import argparse
import misc
import os
import torch
import json
import numpy as np
from data_utils import prepare_dataset_train_val
from pathlib import Path
from engine import train_one_epoch, evaluate
from model_utils import Net
from misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Training configs
    parser.add_argument("--pre_type", action="store_true")
    parser.add_argument("--pre_sm", action="store_true")
    parser.add_argument("--full_train", action="store_true")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_freq", default=100, type=int, help="saving frequency")
    parser.add_argument(
        "--eval_freq", default=10, type=int, help="evaluation frequency"
    )
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    # Optimizer configs
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    # Distributed training configs
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    (
        dataset_train,
        dataset_val,
        cell_type_rna_count,
        sm_feature,
    ) = prepare_dataset_train_val(full_train=args.full_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
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
    if args.pre_type:
        model.type_emb = model.type_emb.to(device)
    if args.pre_sm:
        model.sm_emb = model.sm_emb.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * misc.get_world_size()
    print("effective batch size: %d" % eff_batch_size)
    args.lr = args.blr * eff_batch_size / 128

    print("base lr: %.2e" % (args.lr * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )
    model_without_ddp = model.module

    loss_scaler = NativeScaler()
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(0, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
        )
        if args.output_dir and (
            epoch % args.save_freq == 0 or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            model_without_ddp = model.module
            test_stats = evaluate(data_loader_val, model, device)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if (
            args.output_dir
            and misc.is_main_process()
            and epoch % args.eval_freq == 0
            or epoch + 1 == args.epochs
        ):
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if not args.full_train:
        args.save_freq = 1000
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
