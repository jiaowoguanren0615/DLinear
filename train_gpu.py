""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import datetime
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import os
from pathlib import Path
from timm.utils import NativeScaler, get_state_dict, ModelEma


from util.samplers import RASampler
from util import utils as utils
from util.optimizer import Lion
from util.loss import loss_family
from util.engine import train_one_epoch, evaluate
from util.lr_sched import create_lr_scheduler

from datasets import build_dataset

from models import DLinear, Informer, Reformer, FEDFormer, Transformer, AutoFormer, NLinear, Linear


def get_args_parser():
    parser = argparse.ArgumentParser('TimeSeriesModelSoups training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--loss_type', default='RMSE', type=str,
                        choices=['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'RSE', 'CORR'])

    # Dataset parameters
    parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/usr/local/MyObjData/TimeseriesData/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # Model parameters
    parser.add_argument('--model', default='Informer', type=str, metavar='MODEL',
                        choices=['DLinear', 'Informer', 'Reformer', 'FEDFormer', 'Transformer', 'AutoFormer', 'NLinear', 'Linear'],
                        help='Name of model to train')
    parser.add_argument('--train_only', type=bool, default=False,
                        help='perform training on full input dataset without validation and testing')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=720,
                        choices=[96, 192, 336, 720], help='24 36 48 60 for national_illness dataset')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    # Formers
    parser.add_argument('--embed_type', type=int, default=3,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', default=False, type=bool, help='whether to output attention in ecoder')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    #Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--writer_output', default='./',
                        help='path where to save SummaryWriter, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Finetuning params
    parser.add_argument('--set_ln_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')

    # training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')
    return parser




def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.writer_output, 'runs'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    dataset_train, _ = build_dataset(args=args, flag='train')
    dataset_val, _ = build_dataset(args=args, flag='val')

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    assert args.model in ['DLinear', 'Informer', 'Reformer', 'FEDFormer', 'Transformer', 'Autoformer', 'NLinear', 'Linear'], 'You must choose a right model'

    print(f"Creating model: {args.model}")

    if args.model == 'DLinear':
        model = DLinear(args.seq_len, args.pred_len, args.individual, args.enc_in)
    elif args.model == 'Informer':
        model = Informer(args)
    elif args.model == 'Transformer':
        model = Transformer(args)
    elif args.model == 'AutoFormer':
        model = AutoFormer(args)
    elif args.model == 'Reformer':
        model = Reformer(args)
    elif args.model == 'FEDformer':
        model = FEDFormer(args)
    elif args.model == 'NLinear':
        model = NLinear(args)
    elif args.model == 'Linear':
        model = Linear(args)
    else:
        model = DLinear(args.seq_len, args.pred_len, args.individual, args.enc_in)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    #
    # print('*****************')
    # print('Initial LR is ', linear_scaled_lr)
    # print('*****************')

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), args.lr, weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()
    lr_scheduler = create_lr_scheduler(optimizer,
                                       num_step=len(data_loader_train),
                                       epochs=args.epochs,
                                       warmup=True,
                                       warmup_epochs=1,
                                       warmup_factor=1e-3)


    # criterion = torch.nn.MSELoss()
    print(f'Create loss calculation: {args.loss_type}')
    criterion = loss_family[args.loss_type]
    best_loss = 100.0

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
    if args.resume or os.path.exists(f'{args.output_dir}/{args.model}_sl{args.seq_len}_pl{args.pred_len}_ll{args.label_len}_et{args.embed_type}_dm{args.d_model}_best_checkpoint.pth'):
        args.resume = f'{args.output_dir}/{args.model}_sl{args.seq_len}_pl{args.pred_len}_ll{args.label_len}_et{args.embed_type}_dm{args.d_model}_best_checkpoint.pth'
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'])
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():  # load parameters to cuda
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            best_loss = checkpoint['best_score']
            print(f'Now best_loss is {best_loss}')
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        # util.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        print(f"Evaluating model: {args.model}")
        print(f'No Visualization')
        test_stats = evaluate(data_loader_val, model, criterion, device, None, None, args, visualization=False)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema,
            # set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            set_training_mode=True,
            set_ln_eval=args.set_ln_eval,  # set bn to eval if finetune
            writer=writer,
            lr_scheduler=lr_scheduler,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, criterion, device, epoch, writer, args, visualization=True)
        print(f"Loss of the network on the {len(dataset_val)} test text-data: {test_stats['valid_loss']:.4f}")

        if test_stats["valid_loss"] < best_loss:
            best_loss = test_stats["valid_loss"]
            if args.output_dir:
                ckpt_path = os.path.join(output_dir, f'{args.model}_sl{args.seq_len}_pl{args.pred_len}_ll{args.label_len}_et{args.embed_type}_dm{args.d_model}_best_checkpoint.pth')
                checkpoint_paths = [ckpt_path]
                print("Saving checkpoint to {}".format(ckpt_path))
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'best_score': best_loss,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Min {args.loss_type} loss: {best_loss:.4f}')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeSeriesModelSoups training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)