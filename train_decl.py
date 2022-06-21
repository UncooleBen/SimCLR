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


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

# input_1, input_2分别为两个augmentation图片的输出
def trainfdg(module, input_1, input_2, args):
    if not module.is_last_layer():
        if input_1 is not None and input_2 is not None:
            # print('not last and has aug1 aug2')
            module.train()
            args.receive_grad[module.get_module_num()] = module.backward()

            if module.get_update_count() >= args.ac_step:
                module.step()
                module.zero_grad()
                module.clear_update_count()

            module.set_input(input_1, input_2)
            module.set_output(module.forward(input_1, free_grad=args.free_compute_graph), module.forward(
                input_2, free_grad=args.free_compute_graph))

            oldest_input_1, oldest_input_2 = module.get_oldest_input()

            if oldest_input_1 is None or oldest_input_2 is None:
                print('no input gradients obtained in module {}'.format(
                    module.get_module_num()))
            elif not module.first_layer:
                module.set_input_grad(oldest_input_1.grad, oldest_input_2.grad)

    elif module.is_last_layer():
        if input_1 is not None and input_2 is not None:
            # print(f'last and has aug1 aug2')
            module.train()
            # choose mode B
            if args.receive_grad[module.get_module_num() - 1] is True:
                if module.get_update_count() >= args.ac_step:
                    module.step()
                    module.zero_grad()
                    module.clear_update_count()
            else:
                pass
            output_1 = module(input_1)
            output_2 = module(input_2)

            # 最后一层对输出向量归一化后计算loss
            output_1 = F.normalize(output_1, dim=-1)
            output_2 = F.normalize(output_2, dim=-1)
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
            module.inc_update_count()
            # Update loss to module
            module.set_loss(loss.detach())
            module.set_input_grad(input_1.grad, input_2.grad)
        else:
            pass


def train_decl(train_loader, module, epoch, args):
    for m in range(num_split):
        args.receive_grad[m] = False

    # 暂时将total epoch设置成1
    pbar = tqdm(enumerate(train_loader), desc='Training Epoch {}/{}'.format(str(epoch + 1), args.epochs),
                total=len(train_loader), unit='batch')

    # 用于计算每个epoch中的平均loss
    total_loss, total_num = 0.0, 0

    # pos_i: [batch_size, 3, 32, 32]
    # 总的输入batch为 2*batch_size, neg_sample数量为 2*batch_size - 1
    for i, (pos_1, pos_2, _) in pbar:
        # lr = adjust_learning_rate(
        #   module=module, epoch=epoch, step=i, len_epoch=len(train_loader))
        # data_time.update(time.time() - end)

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

        # Set previous module's output as current module's input for next round
        from torch.autograd import Variable
        for m in reversed(range(1, num_split)):
            previous_module_output_1, previous_module_output_2 = module[m-1].get_output(
            )
            args.input_info1[m] = Variable(previous_module_output_1.detach().clone().to(
                device[m]), requires_grad=True) if previous_module_output_1 is not None else None
            args.input_info2[m] = Variable(previous_module_output_2.detach().clone().to(
                device[m]), requires_grad=True) if previous_module_output_2 is not None else None
        # Set current module's delayed grad as next module's input_grad for next round
        for m in range(num_split-1):
            next_module_input_grads_1, next_module_input_grads_2 = module[m+1].get_input_grad(
            )
            next_module_input_grads_1 = next_module_input_grads_1.clone().to(
                device[m]) if next_module_input_grads_1 is not None else None
            next_module_input_grads_2 = next_module_input_grads_2.clone().to(
                device[m]) if next_module_input_grads_2 is not None else None
            module[m].set_dg(next_module_input_grads_1,
                             next_module_input_grads_2)
        # TODO: Compute communication time
        last_idx = num_split - 1
        total_num += args.batch_size
        total_loss += module[last_idx].get_loss() * args.batch_size
        pbar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num


# Validation for one epoch (using KNN)
def test(memory_data_loader, test_data_loader, module, epoch, args):
    for m in range(num_split):
        module[m].model.eval()

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            # 迭代num_split个module获取f输出的feature
            feature = data.to(device[0], non_blocking=True)

            for m in range(num_split):
                if m != num_split - 1:
                    feature = module[m](feature)
                    feature = feature.to(device[m + 1])
                else:
                    # 最后一个module只输出f的output
                    feature = module[m].get_feature(feature)

            # feature, out = net(data.cuda(device='cuda:2', non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(device='cuda:2', non_blocking=True), target.cuda(device=feature_bank.device,
                                                                                      non_blocking=True)
            feature = data.to(device[0], non_blocking=True)
            for m in range(num_split):
                if m != num_split - 1:
                    feature = module[m](feature)
                    feature = feature.to(device[m + 1])
                else:
                    # 最后一个module只输出f的output
                    feature = module[m].get_feature(feature)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(
                data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / args.temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(
                data.size(0) * args.k, args.c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(
                data.size(0), -1, args.c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def main():
    # Load 不同的 decl implementation 到 global scope
    global num_split, device, module
    if args.free_compute_graph:
        print('run in free_compute_graph version')
        from model_decl_nograph import num_split, device, module
    else:
        print('run in vanilla version')
        from model_decl_vanilla import num_split, device, module

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
    save_name_pre = '{}_{}_{}_{}_{}'.format(128, args.temperature, args.k, args.batch_size, args.epochs)

    if not os.path.exists('results_decl'):
        os.mkdir('results_decl')

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_decl(train_loader, module, epoch, args)
        results['train_loss'].append(train_loss.item())  # convert the data from torch.tensor to float

        test_acc_1, test_acc_5 = test(
            memory_loader, test_loader, module, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results_decl/{}statistics.csv'.format(save_name_pre), index_label='epoch')

    # TODO: save models and params
    # if args.save:
    #     pass

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
    parser.add_argument('--temperature', default=0.5,
                        type=float, help='Temperature used in softmax')
    parser.add_argument('--clip', default=1e10, type=float,
                        help='Gradient clipping')
    parser.add_argument('--k', default=200, type=int,
                        help='Top k most similar images used to predict the label')

    args = parser.parse_args()

    main()
