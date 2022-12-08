r"""PyTorch Detection Training.
To run in a multi-gpu environment, use the distributed launcher::
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""

import os
import sys
import yaml
import random
import numpy as np
from addict import Dict

import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.engine import train_one_epoch, evaluate
from references import utils
import references.transforms as T
from references.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler

import eval

from model import *



DATASET_PATH = "/labeled/labeled"
EXP_ID = "ft"
MODEL_ID = 8

def init_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_transform(train, use_jitter=False):
    transforms = []
    transforms.append(lambda x, y: (torchvision.transforms.functional.to_tensor(x), y))
    if train:
        if use_jitter:
            transforms.append(T.Jitter())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=20, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")

    parser.add_argument("--output-dir", default="./checkpoints", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    return parser


def main(args_multi_gpu):
    if args_multi_gpu.output_dir:
        utils.mkdir(args_multi_gpu.output_dir)

    utils.init_distributed_mode(args_multi_gpu)
    print(args_multi_gpu)

    # init config
    config_path = os.path.abspath(f'./configs/{EXP_ID}.yaml')
    assert os.path.isfile(config_path)
    args = Dict(yaml.safe_load(open(config_path)))
    print('Loading Args:')
    for k, v in args.items():
        print(f'[{k}]: {v}')

    # init training
    init_seed(args.seed)
    
    print("Loading data")
    # init dataloaders
    train_dataset = eval.LabeledDataset(
        root=DATASET_PATH,
        split='training',
        transforms=get_transform(train=True, use_jitter=args.use_jitter)
    )
    valid_dataset = eval.LabeledDataset(
        root=DATASET_PATH,
        split='validation',
        transforms=get_transform(train=False)
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model()
    # move model to the right device
    model.to(device)

    model_without_ddp = model
    if args_multi_gpu.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args_multi_gpu.gpu])
        model_without_ddp = model.module

    # freeze
    for p in model_without_ddp.backbone.parameters():
        p.requires_grad = False

    params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    
    # unfreeze
    for p in model_without_ddp.backbone.parameters():
        p.requires_grad = True
        
    # init optimizer & scheduler
    # optim only not freeze params
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # optimizer = torch.optim.AdamW(utils.get_params(model_without_ddp, mode=args.optim_mode), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    if args_multi_gpu.resume:
        checkpoint = torch.load(args_multi_gpu.resume, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args_multi_gpu.resume)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args_multi_gpu.start_epoch = checkpoint["epoch"] + 1


    print("Creating data loaders")
    if args_multi_gpu.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args_multi_gpu.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args_multi_gpu.workers, collate_fn=eval.collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, sampler=valid_sampler, num_workers=args_multi_gpu.workers, collate_fn=eval.collate_fn
    )


    for epoch in range(args_multi_gpu.start_epoch, args_multi_gpu.epochs):
        if args_multi_gpu.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            device=device,
            epoch=epoch,
            print_freq=args.train_print_freq,
            warmup_iter=args.warmup_iter
        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_valid, device=device)

        if args_multi_gpu.output_dir:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(to_save, os.path.join(args_multi_gpu.output_dir, f"fasterrcnn-{MODEL_ID}-{epoch}.pth"))


if __name__ == "__main__":
    args_multi_gpu = get_args_parser().parse_args()
    main(args_multi_gpu)
