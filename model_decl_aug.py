import sys
from collections import deque
from typing import overload
from model_decl_interface import IDeclModule

import torch
from torchvision.models.resnet import resnet50
import torch.nn as nn
from model import Model
from utils import Flatten
import torch.nn.functional as F
import torch.optim as optim

model_list = {}
num_split = 2

# resnet50 + projection head
model = Model()

if num_split == 2:
    model_list[0] = nn.Sequential(
        model.f[0], model.f[1], model.f[2], model.f[3], model.f[4])
    model_list[1] = nn.Sequential(
        model.f[5], model.f[6], model.f[7], Flatten(), model.g)
elif num_split == 3:
    model_list[0] = nn.Sequential(
        model.f[0], model.f[1], model.f[2])
    model_list[1] = nn.Sequential(
        model.f[3], model.f[4], model.f[5], )
    model_list[2] = nn.Sequential(
        model.f[6], model.f[7], Flatten(), model.g)
elif num_split == 4:
    model_list[0] = nn.Sequential(
        model.f[0], model.f[1], model.f[2], model.f[3][:2])
    model_list[1] = nn.Sequential(
        model.f[3][2:], model.f[4][:1])
    model_list[2] = nn.Sequential(
        model.f[4][1:], model.f[5][:1])
    model_list[3] = nn.Sequential(
        model.f[5][1:], model.f[6], model.f[7], Flatten(), model.g)
else:
    print('invalid split number!')
    sys.exit()


class DeclModuleImpl(nn.Module):
    def __init__(self, model, optimizer, split_loc, num_split):
        super(DeclModuleImpl, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.module_num = split_loc

        # delay
        self.delay = 2 * (num_split - split_loc - 1) + 1

        # gradient
        self.dg = None

        # used for ADL gradient accumulation
        self.update_count = 0

        self.output = None

        # input queue存放未进行bp，等待2nd forward的input batch
        self.input_q = deque(maxlen=self.delay + 1)
        for _ in range(self.delay + 1):
            self.input_q.append(None)

        self.input_grad = None

        if split_loc == 0:
            self.first_module = True
            self.last_module = False
        elif split_loc == num_split - 1:
            self.first_module = False
            self.last_module = True
        else:
            self.first_module = False
            self.last_module = False

        # last module中额外定义projection cache存放累积的projection
        if self.last_module:
            self.projection_cache = deque(maxlen=3)

        self.loss = 0.0

    # set free_grad to True to avoid generating any computation graph
    def forward(self, x, free_grad=False):
        if free_grad:
            with torch.no_grad():
                res = self.model(x)
        else:
            res = self.model(x)
        return res

    def backward(self):
        if self.dg is not None and self.input_q[0] is not None:
            # 2nd forward for the oldest input
            oldest_output = self.forward(self.input_q[0], free_grad=False)
            oldest_output.backward(self.dg)

            del oldest_output
            del self.dg
            self.dg = None

            rev_grad = True

        else:
            rev_grad = False
            print('no backward in module {} dg is None {} input is None {}'.format(self.module_num, self.dg is None,
                                                                                   self.input_q[0] is None))

        # used for ADL
        self.inc_update_count()

        return rev_grad

    def last_module_backward(self):
        if self.dg is not None and self.input_q[0] is not None:
            # 2nd forward for the oldest input
            oldest_output = self.forward(self.input_q[0], free_grad=False)
            oldest_output.backward(self.dg)

            del oldest_output

            rev_grad = True

        else:
            rev_grad = False
            print('no backward in the last module {} dg is None {} input is None {}'.format(self.module_num,
                                                                                            self.dg is None,
                                                                                            self.input_q[0] is None))

    # 获取当前module中第一层的梯度传送给上一个module，同时让input出队
    # TODO: delete??
    def get_grad(self):
        return self.input_q.popleft().grad

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output

    def set_input(self, input_batch):
        self.input_q.append(input_batch)

    def get_oldest_input(self):
        return self.input_q.popleft()

    def train(self):
        self.model.train()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    # used for the last module
    def get_feature(self, x):
        if self.last_module:
            with torch.no_grad():
                feature = self.model[:-1](x)
                # 1 * 2048
                feature = F.normalize(feature, dim=-1)
                return feature
        else:
            return None

    def get_update_count(self):
        return self.update_count

    def inc_update_count(self):
        self.update_count += 1

    def clear_update_count(self):
        self.update_count = 0

    def get_input_grad(self):
        return self.input_grad

    def set_input_grad(self, input_grad):
        self.input_grad = input_grad

    def get_module_num(self):
        return self.module_num

    def is_last_module(self):
        return self.last_module

    def is_first_module(self):
        return self.first_module

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self):
        return self.loss

    def set_dg(self, dg):
        self.dg = dg


# set mulgpu True for debugging
mulgpu = True
devices = {}

if torch.cuda.is_available():
    if mulgpu:
        for i in range(num_split):
            devices[i] = torch.device('cuda:' + str(i))
    # put all modules on single CUDA
    else:
        for i in range(num_split):
            devices[i] = torch.device('cuda:' + str(0))
else:
    # CPU mode
    for i in range(num_split):
        devices[i] = torch.device('mps')

optimizer = {}
# scheduler = {}

for m in model_list:
    model_list[m] = model_list[m].to(devices[m])
    # 使用adam优化器
    optimizer[m] = optim.Adam(
        model_list[m].parameters(), lr=1e-3, weight_decay=1e-6)

# 所有module的集合
module_list = {}

for m in range(num_split):
    module_list[m] = DeclModuleImpl(
        model=model_list[m], optimizer=optimizer[m], split_loc=m, num_split=num_split)
