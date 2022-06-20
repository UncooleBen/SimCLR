import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
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

# Broken
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
    for m in range(num_split):
        for param_group in module[m].optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def accuracy(args, output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # args.batch_size = target.size(0)
    return 0

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
            # print('not last and has aug1 aug2')
            module.train()
            args.receive_grad[module.module_num] = module.backward()

            if module.update_count >= args.ac_step:
                module.step()
                module.zero_grad()
                module.update_count = 0

            module.output_1 = module.forward_nograd(input_1)
            module.output_2 = module.forward_nograd(input_2)
            # print(f'input_1 = {input_1.mean()}')
            # print(f'input_2 = {input_2.mean()}')

            module.input_1.append(input_1)
            module.input_2.append(input_2)
            oldest_input_1 = module.input_1.popleft()
            oldest_input_2 = module.input_2.popleft()
            if oldest_input_1 is None or oldest_input_2 is None:
                print('no input gradients obtained in module {}'.format(
                    module.module_num))
            elif not module.first_layer:
                module.input_grad_1 = oldest_input_1.grad
                module.input_grad_2 = oldest_input_2.grad

    elif module.last_layer:
        if input_1 is not None and input_2 is not None:
            # print(f'last and has aug1 aug2')
            module.train()
            # choose mode B
            if args.receive_grad[module.module_num - 1] is True:
                if module.update_count >= args.ac_step:
                    module.step()
                    module.zero_grad()
                    module.update_count = 0
            else:
                pass
            # print(f'input_1 = {input_1.mean()}')
            # print(f'input_2 = {input_2.mean()}')
            output_1 = module(input_1)
            output_2 = module(input_2)

            # 最后一层对输出向量归一化后计算loss
            output_1 = F.normalize(output_1, dim=-1)
            output_2 = F.normalize(output_2, dim=-1)
            # print(f'output_1 = {output_1}')
            # print(f'output_2 = {output_2}')
            # compute simclr loss
            # [2*B, D]
            out = torch.cat([output_1, output_2], dim=0)
            # print(f'out= {torch.norm(out, 2, dim=1)}')
            # [2*B, 2*B]
            sim_matrix = torch.exp(
                torch.mm(out, out.t().contiguous()) / args.temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 *
                    args.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(
                mask).view(2 * args.batch_size, -1)
            # print(f'sim_matrix= {sim_matrix}')
            # compute loss
            pos_sim = torch.exp(
                torch.sum(output_1 * output_2, dim=-1) / args.temperature)
            # print(f'pos_sim = {pos_sim}')
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            # print(f'loss = {loss}')
            loss.backward()
            module.update_count += 1
            # Update loss to module
            module.loss = loss

            # TODO: Update accuracy
            acc1 = accuracy(args, output_1.data, output_2.data, topk=(1,))
            module.input_grad_1 = input_1.grad
            module.input_grad_2 = input_2.grad
            module.acc = acc1
        else:
            pass


def train_decl(train_loader, module, epoch, args):
    for m in range(num_split):
        args.receive_grad[m] = False

    # 暂时将total epoch设置成1
    pbar = tqdm(enumerate(train_loader), desc='Training Epoch {}/{}'.format(str(epoch + 1), args.epochs),
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
        # print(f'i = {i}')
        # lr = adjust_learning_rate(
        #   module=module, epoch=epoch, step=i, len_epoch=len(train_loader))
        data_time.update(time.time() - end)

        # dataLoader中pin_memory=True时，才能使用non_blocking，此时不使用虚拟内存，加快速度
        pos_1 = pos_1.to(device[0], non_blocking=True)
        pos_2 = pos_2.to(device[0], non_blocking=True)

        # newest batch input: 每个module的input，module 0的input为data
        args.input_info1[0] = pos_1
        args.input_info2[0] = pos_2
        # train in fdg/adl mode
        processes = []

        for m in range(num_split):
            # print(f'epoch = {epoch} m = {m}')
            p = threading.Thread(target=trainfdg, args=(
                module[m], args.input_info1[m], args.input_info2[m], args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # 串行调试
        # for m in range(num_split):
        #     trainfdg(module[m], args.input_info1[m], args.input_info2[m], args)

        # TODO: Communication
        # Set previous module's output as current module's input for next round
        from torch.autograd import Variable
        for m in reversed(range(1, num_split)):
            previous_module_output_1, previous_module_output_2 = module[m-1].get_output()
            # del args.input_info1[m]
            args.input_info1[m] = Variable(previous_module_output_1.detach().clone().to(
                device[m]), requires_grad=True) if previous_module_output_1 is not None else None
            # del args.input_info2[m]
            args.input_info2[m] = Variable(previous_module_output_2.detach().clone().to(
                device[m]), requires_grad=True) if previous_module_output_2 is not None else None
        # Set current module's delayed grad as next module's input_grad for next round
        for m in range(num_split-1):
            # del module[m].dg_1
            module[m].dg_1 = module[m+1].input_grad_1.clone().to(device[m]
                                                                 ) if module[m+1].input_grad_1 is not None else None
            # del module[m+1].input_grad_1
            module[m + 1].input_grad_1 = None

            # del module[m].dg_2
            module[m].dg_2 = module[m+1].input_grad_2.clone().to(device[m]
                                                                 ) if module[m+1].input_grad_2 is not None else None
            # del module[m + 1].input_grad_2
            module[m + 1].input_grad_2 = None
        # TODO: Compute communication time
        # HERE
        # TODO: Update accuracy
        last_idx = num_split - 1
        if module[last_idx].acc != 0:
            top1.update(to_python_float(
                module[last_idx].acc), args.batch_size)
            top5.update(to_python_float(
                module[last_idx].acc5), args.batch_size)
        if module[last_idx].loss != 0:
            losses.update(to_python_float(module[last_idx].loss), args.batch_size)

        pbar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, module[last_idx].loss))

    return module[last_idx].loss


# Validation for one epoch (using KNN)
def test(memory_data_loader, test_data_loader, module, epoch, args):
    for m in range(num_split):
        module[m].eval()

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            # 迭代num_split个module获取f输出的feature
            feature = data
            for m in range(num_split):
                if m != num_split - 1:
                    feature = module[m](feature)
                else:
                    # 最后一个module只输出f的output
                    feature = module[m].get_feature(feature)

            # feature, out = net(data.cuda(device='cuda:2', non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(device='cuda:2', non_blocking=True), target.cuda(device='cuda:2',
                                                                                      non_blocking=True)
            feature = data
            for m in range(num_split):
                if m != num_split - 1:
                    feature = module[m](feature)
                else:
                    # 最后一个module只输出f的output
                    feature = module[m].get_feature(feature)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / args.temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, args.c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, args.c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


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

    args.c = len(memory_data.classes)

    args.receive_grad = []
    args.input_info1 = {}
    args.input_info2 = {}

    for m in range(num_split):
        args.input_info1[m] = None
        args.input_info2[m] = None
        args.receive_grad.append(False)
    print('Training begins')

    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    for epoch in range(0, args.epochs):
        train_loss = train_decl(train_loader, module, epoch, args)
        results['train_loss'].append(train_loss)

        test_acc_1, test_acc_5 = test(memory_loader, test_loader, module, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description='SimCLR in decoupled training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('-free-compute-graph', type=bool, default=True,
                        help='Whether to free compute graph of aug1')
    parser.add_argument('--ac-step', type=int, default=1,
                        help='')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--clip', default=1e10, type=float, help='Gradient clipping')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')

    args = parser.parse_args()

    main()
