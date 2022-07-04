import argparse
import os
import shutil
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import threading

import utils

from model_decl_aug import devices, module_list, model, num_split


print(devices)

def trainfdg(module, input, args):
    # not last module implementation
    if not module.is_last_module():
        if input is not None:
            module.train()

            # MODE B: do bp part first

            # set TRUE if current module has input gradient from next module and executes backward
            # set FALSE if there is no backward
            args.receive_grad[module.get_module_num()] = module.backward()

            # used for ADL, accumulate gradients and update params every args.ac_step
            # update_count will not increase if there is no backward
            if module.get_update_count() >= args.ac_step:
                module.step()
                module.zero_grad()
                module.clear_update_count()

            module.set_input(input)
            # not need computation graph for 1st Forward
            module.set_output(module.forward(input, free_grad=True))

            # pop the oldest input
            oldest_input = module.get_oldest_input()

            if oldest_input is None:
                print('no input gradients obtained in module {}'.format(
                    module.get_module_num()))
            elif not module.is_first_module():
                module.set_input_grad(oldest_input.grad)
        else:
            pass

    # last module implementation
    else:
        if input is not None:
            module.train()

            if args.receive_grad[module.get_module_num() - 1] is True:
                if module.get_update_count() >= args.ac_step:
                    module.step()
                    module.zero_grad()
                    module.clear_update_count()

            # set input to the input queue of the last module
            module.set_input(input)
            # put the output projection into a cache temporarily waiting for computing loss
            output = module.forward(input, free_grad=True)
            output = F.normalize(output, dim=-1)
            module.projection_cache.append(output)

            ####################
            # start calculating loss when cache size is equals to 2
            if len(module.projection_cache) >= 2:
                # initialized as 0 at the start of training
                # idc = 1 or 2, when current batch is aug1, idc = 1; aug2, idc = 2
                args.idc += 1

                # current computation graph that needs backprop
                current_output = module.forward(module.input_q[0], free_grad=False)
                current_output = F.normalize(current_output, dim=-1)

                # choose sample batch from projection cache to calculate loss
                sample_output = module.projection_cache[2 - args.idc]

                # compute simclr loss
                # [2*B, D]
                out = torch.cat([current_output, sample_output], dim=0)
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
                    torch.sum(current_output * sample_output, dim=-1) / args.temperature)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

                # Update loss to module
                module.set_loss(loss.detach())

                loss.backward()
                module.inc_update_count()

                # when all batches according to current loss complete backprop, pop all depended projections out
                if args.idc >= 2:
                    args.idc = 0
                    for _ in range(2):
                        module.projection_cache.popleft()

            # pop the oldest input
            oldest_input = module.get_oldest_input()

            # set the gradient for passing it to the previous module
            if oldest_input is not None:
                module.set_input_grad(oldest_input.grad)
            else:
                print('no input gradients obtained in the last module {}'.format(
                    module.get_module_num()))
        else:
            pass


# decoupled contrastive learning with delayed updating
def train_ddcl(train_loader, module_list, epoch, args):
    for m in range(num_split):
        args.receive_grad[m] = False

    pbar = tqdm(enumerate(train_loader), desc='Training Epoch {}/{}'.format(str(epoch + 1), args.epochs),
                total=len(train_loader), unit='batch')

    # 用于计算每个epoch中的平均loss
    total_loss, total_num = 0.0, 0

    # pos_i: [batch_size, 3, 32, 32]
    for _, (pos_1, pos_2, _) in pbar:
        # 先后遍历两个aug后的batch
        for pos in [pos_1, pos_2]:
            pos = pos.to(devices[0], non_blocking=True)

            args.input_info[0] = pos

            processes = []
            for m in range(num_split):
                p = threading.Thread(target=trainfdg, args=(
                    module_list[m], args.input_info[m], args))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            # # 串行调试
            # for m in range(num_split):
            #     trainfdg(module_list[m], args.input_info[m], args)

            # communication among modules
            from torch.autograd import Variable
            for m in reversed(range(1, num_split)):
                previous_module_output = module_list[m - 1].get_output()
                args.input_info[m] = Variable(previous_module_output.detach().clone().to(devices[m]),
                                              requires_grad=True) if previous_module_output is not None else None

            for m in range(num_split - 1):
                next_module_input_grads = module_list[m + 1].get_input_grad()
                next_module_input_grads = next_module_input_grads.clone().to(devices[m]) \
                    if next_module_input_grads is not None else None
                module_list[m].set_dg(next_module_input_grads)

            last_idx = num_split - 1
            total_num += args.batch_size
            total_loss += module_list[last_idx].get_loss() * args.batch_size
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
            feature = data.to(devices[0], non_blocking=True)

            for m in range(num_split):
                if m != num_split - 1:
                    feature = module_list[m](feature)
                    feature = feature.to(devices[m + 1])
                else:
                    # 最后一个module只输出f的output
                    feature = module_list[m].get_feature(feature)

            # feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(device=feature_bank.device,
                                                                     non_blocking=True)
            feature = data.to(devices[0], non_blocking=True)
            for m in range(num_split):
                if m != num_split - 1:
                    feature = module_list[m](feature)
                    feature = feature.to(devices[m + 1])
                else:
                    # 最后一个module只输出f的output
                    feature = module_list[m].get_feature(feature)

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
                                     .format(epoch, args.epochs, total_top1 / total_num * 100,
                                             total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def main():
    print('run in DDCL version')

    # used for maintaining cache
    args.idc = 0

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
    args.input_info = {}

    for m in range(num_split):
        args.input_info[m] = None
        args.receive_grad.append(False)
    print('Training begins')

    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(128, args.temperature, args.k, args.batch_size, args.epochs)

    if not os.path.exists('results_decl'):
        os.mkdir('results_decl')

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_ddcl(train_loader, module_list, epoch, args)
        results['train_loss'].append(train_loss.item())  # convert the data from torch.tensor to float

        test_acc_1, test_acc_5 = test(
            memory_loader, test_loader, module_list, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results_ddcl/{}statistics.csv'.format(save_name_pre), index_label='epoch')

        is_best = test_acc_1 > best_acc

        # save best top-1 model weights
        if args.save and is_best:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results_ddcl/{}_model.pth'.format(save_name_pre))


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description='SimCLR in decoupled training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the model (default: False)')
    # parser.add_argument('--num_split', type=int, default=2,
    #                     help='split number of model')
    parser.add_argument('--ac-step', type=int, default=1,
                        help='gradient accumulate steps')
    parser.add_argument('--temperature', default=0.5,
                        type=float, help='Temperature used in softmax')
    parser.add_argument('--clip', default=1e10, type=float,
                        help='Gradient clipping')
    parser.add_argument('--k', default=200, type=int,
                        help='Top k most similar images used to predict the label')

    args = parser.parse_args()

    main()
