"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.utils import ModelEma

from util import utils as utils


def set_ln_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._LazyNormBase):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.MSELoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None,
                    set_training_mode=True,
                    set_ln_eval=False,
                    writer=None,
                    lr_scheduler=None,
                    args=None):

    model.train(set_training_mode)
    num_steps = len(data_loader)

    if set_ln_eval:
        set_ln_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # compute output
        with torch.cuda.amp.autocast():
            if 'Linear' in args.model:
                outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        learning_rate = optimizer.param_groups[0]["lr"]
        metric_logger.update(train_loss=loss_value)
        metric_logger.update(lr=learning_rate)


        if idx % print_freq == 0:
            if args.local_rank == 0:
                iter_all_count = epoch * num_steps + idx
                writer.add_scalar('train_loss', loss, iter_all_count)
                # writer.add_scalar('grad_norm', grad_norm, iter_all_count)
                writer.add_scalar('lr', learning_rate, iter_all_count)

        lr_scheduler.step()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, criterion, device, epoch, writer, args, visualization=True):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    print_freq = 20
    for idx, ((batch_x, batch_y, batch_x_mark, batch_y_mark)) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # compute output
        with torch.cuda.amp.autocast():
            if 'Linear' in args.model:
                outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)

        batch_size = batch_x.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['valid_loss'].update(loss.item(), n=batch_size)


    if visualization and args.local_rank == 0:
        writer.add_scalar('valid_loss', loss.item(), epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}