import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import threading

import utils
from model import Model
from model_decl import num_split, device, module


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(module, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    for i in range(len(args.lr_decay_milestones)):
        # 保证总epoch数大于milestone的最大数
        if epoch >= args.lr_decay_milestones[-1]:
            scaling = len(args.lr_decay_milestones)
            break
        elif epoch < args.lr_decay_milestones[i]:
            scaling = i
            break
    lr = args.lr * 10 ** (-scaling)
    """Warmup"""
    if epoch < args.warm_up_epochs:
        lr = 0.01 * args.lr + (args.lr - 0.01 * args.lr) * (step + 1 + epoch * len_epoch) / (
            args.warm_up_epochs * len_epoch)
    for m in range(args.num_split):
        for param_group in module[m].optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def accuracy(args, output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    args.batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / args.batch_size))
    return res

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

# input_1, input_2分别为两个augmentation图片的输出


def trainfdg(module, input_1, input_2, args):
    if not module.last_layer:
        if input_1 is not None and input_2 is not None:
            module.train()
            args.receive_grad[module.module_num] = module.backward()

            if module.update_count >= args.ac_step:
                module.step()
                module.zero_grad()
                module.update_count = 0

            output_1 = module(input_1)
            output_2 = module(input_2)

            module.output_1.append(output_1)
            module.output_2.append(output_2)

            if not module.first_layer:
                module.input_1.append(input_1)
                module.input_2.append(input_2)
                oldest_input_1 = module.input_1.popleft()
                oldest_input_2 = module.input_2.popleft()
                if oldest_input_1 is None or oldest_input_2 is None:
                    print('no input gradients obtained in module {}'.format(
                        module.module_num))
                else:
                    module.input_grad_1 = oldest_input_1.grad
                    module.input_grad_2 = oldest_input_2.grad

    elif module.last_layer:
        if input_1 is not None and input_2 is not None:
            module.train()
            # choose mode B
            if args.receive_grad[module.module_num - 1] is True:
                if module.update_count >= args.ac_step:
                    module.step()
                    module.zero_grad()
                    module.update_count = 0
            else:
                pass
            output_1 = module(input_1)
            output_2 = module(input_2)
            # compute simclr loss
            # [2*B, D]
            out = torch.cat([output_1, output_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(
                torch.mm(out, out.t().contiguous()) / args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 *
                    args.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(
                mask).view(2 * args.batch_size, -1)

            # compute loss
            pos_sim = torch.exp(
                torch.sum(output_1 * output_2, dim=-1) / args.temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            loss.backward()
            module.update_count += 1
            # TODO: Update accuracy
            acc1 = accuracy(args, output_1.data, output_2.data, topk=(1,))
            module.input_grad_1 = input_1.grad
            module.input_grad_2 = input_2.grad
            module.acc = acc1
        else:
            pass


def train_decl(train_loader, module, epoch, args):
    for m in range(args.num_split):
        args.receive_grad[m] = False

    # 暂时将total epoch设置成300
    pbar = tqdm(enumerate(train_loader), desc='Training Epoch {}/{}'.format(str(epoch + 1), 300),
                total=len(train_loader), unit='batch')

    # 可记录单次和平均时间
    batch_time = AverageMeter()
    data_time = AverageMeter()
    com_time = AverageMeter()  # 通信时间
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # pos_i: [batch_size, 3, 32, 32]
    # 总的输入batch为 2*batch_size, neg_sample数量为 2*batch_size - 1
    for i, (pos_1, pos_2, _) in pbar:
        lr = adjust_learning_rate(
            module=module, epoch=epoch, step=i, len_epoch=len(train_loader))
        data_time.update(time.time() - end)

        # dataLoader中pin_memory=True时，才能使用non_blocking，此时不使用虚拟内存，加快速度
        pos_1 = pos_1.to(device[0], non_blocking=True)
        pos_2 = pos_2.to(device[0], non_blocking=True)

        # newest batch input: 每个module的input，module 0的input为data
        args.input_info1[0] = pos_1
        args.input_info2[0] = pos_2
        # train in fdg/adl mode
        processes = []

        for m in range(args.num_split):
            p = threading.Thread(target=trainfdg, args=(
                module[m], args.input_info1[m], args.input_info2[m]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # TODO: Communication
        # Set previous module's output as current module's input for next round
        from torch.autograd import Variable
        for m in reversed(range(1, args.num_split)):
            previous_module_output_1, previous_module_output_2 = module[m-1].get_output(
            )
            args.input_info1[m] = Variable(previous_module_output_1.detach().clone().to(
                device[m]), requires_grad=True) if previous_module_output_1 is not None else None
            args.input_info2[m] = Variable(previous_module_output_2.detach().clone().to(
                device[m]), requires_grad=True) if previous_module_output_2 is not None else None
        # Set next module's input_grad as current module's delayed grad for next round
        for m in range(args.num_split-1):
            module[m].dg_1 = module[m+1].input_grad_1.clone().to(device[m]
                                                                 ) if module[m+1].input_grad_1 is not None else None
            module[m].dg_2 = module[m+1].input_grad_2.clone().to(device[m]
                                                                 ) if module[m+1].input_grad_2 is not None else None
        # TODO: Compute communication time
        # HERE
        # TODO: Update accuracy
        last_idx = args.num_split - 1
        if module[last_idx].acc != 0:
            top1.update(to_python_float(
                module[last_idx].acc), args.batch_size)
            top5.update(to_python_float(
                module[last_idx].acc5), args.batch_size)
        if module[last_idx].loss != 0:
            losses.update(to_python_float(module[last_idx].loss), args.batch_size)

def main():
    # define data loader
    # 此处均为CIFAR10
    # num_workers 16改为0
    train_data = utils.CIFAR10Pair(
        root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    #
    memory_data = utils.CIFAR10Pair(
        root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(
        memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_data = utils.CIFAR10Pair(
        root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    args.receive_grad = []
    args.input_info1 = {}
    args.input_info2 = {}

    for m in range(args.num_split):
        args.input_info1[m] = None
        args.input_info2[m] = None
        args.receive_grad.append(False)
    print('Training begins')

    for epoch in range(0, args.epochs):
        train_decl(train_loader, module, epoch, args)

    # 暂时省略simclr的validation


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description='SimCLR in decoupled training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('-free-compute-graph', type=bool, default=False,
                        help='Whether to free compute graph of aug1')

    args = parser.parse_args()

    main()
