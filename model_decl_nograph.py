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

num_split = 2
model_list = {}

# 定义模型: resnet50 + projection_head
model = Model()

if num_split == 2:
    model_list[0] = nn.Sequential(model.f[0], model.f[1], model.f[2], model.f[3], model.f[4])
    model_list[1] = nn.Sequential(model.f[5], model.f[6], model.f[7], Flatten(), model.g)


class DeclModuleImpl(IDeclModule):
    def __init__(self, model, optimizer, split_loc, num_split):
        super(DeclModuleImpl, self).__init__()

        self.model = model
        self.optimizer = optimizer

        # delay: 2 * (K-k)
        self.delay = 2 * (num_split - split_loc - 1)
        self.output_dq = self.delay

        self.dg_1, self.dg_2 = None, None
        self.update_count = 0
        self.module_num = split_loc

        self.output_1 = None
        self.output_2 = None
        # self.output_1 = deque(maxlen=self.output_dq)
        # self.output_2 = deque(maxlen=self.output_dq)
        # for _ in range(self.output_dq):
        #     self.output_1.append(None)
        #     self.output_2.append(None)
        self.input_1 = deque(maxlen=self.delay + 1)
        self.input_2 = deque(maxlen=self.delay + 1)

        for _ in range(self.delay + 1):
            self.input_1.append(None)
            self.input_2.append(None)

        self.input_grad_1 = None
        self.input_grad_2 = None

        if split_loc == 0:
            self.first_layer = True
            self.last_layer = False
        elif split_loc == num_split - 1:
            self.first_layer = False
            self.last_layer = True
        else:
            self.first_layer = False
            self.last_layer = False

        self.loss = 0

    def forward(self, x):
        return self.model(x)
    
    def forward_nograd(self, x):
        res = None
        with torch.no_grad():
            res = self.model(x)
        return res

    def backward(self):
        # backward on aug1
        if self.dg_1 is not None and self.input_1[0] is not None:
            oldest_output_1 = self.forward(self.input_1[0])
            oldest_output_1.backward(self.dg_1)
            del self.dg_1
            self.dg_1 = None
            rev_grad_1 = True
        else:
            rev_grad_1 = False
            print('no backward for aug1 in module {} dg_1 is None {} input1 is None {}'.format(self.module_num, self.dg_1 is None, self.input_1[0] is None))

        # backward on aug2
        
        if self.dg_2 is not None and self.input_2[0] is not None:
            oldest_output_2 = self.forward(self.input_2[0])
            oldest_output_2.backward(self.dg_2)
            del self.dg_2
            self.dg_2 = None
            rev_grad_2 = True
        else:
            rev_grad_2 = False
            print('no backward for aug2 in module {} dg_2 is None {} input2 is None {}'.format(self.module_num, self.dg_2 is None, self.input_2[0] is None))

        self.inc_update_count()
        return rev_grad_1 and rev_grad_2

    def get_grad(self):
        return self.input_1.popleft().grad, self.input_2.popleft().grad


    def set_output(self, output_1, output_2):
        self.output_1 = output_1
        self.output_2 = output_2

    def get_output(self):
        return self.output_1, self.output_2

    def set_input(self, input_1, input_2):
        self.input_1.append(input_1)
        self.input_2.append(input_2)

    def get_oldest_input(self):
        return self.input_1.popleft(), self.input_2.popleft()

    def train(self):
        self.model.train()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    # used for the last module
    def get_feature(self, x):
        if self.last_layer:
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
        return self.input_grad_1, self.input_grad_2

    def set_input_grad(self, input_grad_1, input_grad_2):
        self.input_grad_1 = input_grad_1
        self.input_grad_2 = input_grad_2

    def get_module_num(self):
        return self.module_num

    def is_last_layer(self):
        return self.last_layer

    def is_first_layer(self):
        return self.first_layer

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self):
        return self.loss

    def set_dg(self, dg_1, dg_2):
        self.dg_1 = dg_1
        self.dg_2 = dg_2

    

# set devices
mulgpu = 1
device = {}

if torch.cuda.is_available():
    if mulgpu:
        for i in range(num_split):
            device[i] = torch.device('cuda:' + str(i))
    else:
        for i in range(num_split):
            device[i] = torch.device('cuda:' + str(0))
else:
    for i in range(num_split):
        device[i] = torch.device('cpu')

# check if the split is valid or not （只测试projection_head输出的128维张量）
test_input = torch.randn(1, 3, 224, 224).to(device[0])

with torch.no_grad():
    model = model.to(device[0])
    model.eval()

    for m in model_list:
        # model_list[m] = model_list[m].to(device[m])
        model_list[m].eval()

    feature1, outputs1_out = model(test_input)

    # test for output
    outputs2 = test_input
    for m in model_list:
        outputs2 = model_list[m](outputs2)

    outputs2 = F.normalize(outputs2, dim=-1)

    # test for feature
    feature2 = test_input
    for m in range(num_split):
        if m != num_split - 1:
            feature2 = model_list[m](feature2)
        else:
            # 最后一个module只输出f的output
            feature2 = model_list[m][:-1](feature2)

    feature2 = F.normalize(feature2, dim=-1)

    diff1 = outputs1_out - outputs2
    diff2 = feature1 - feature2
    if diff1.sum() == 0 and diff2.sum() == 0:
        print('split valid!')
    else:
        print('split invalid!')
        sys.exit()

optimizer = {}
scheduler = {}

for m in model_list:
    model_list[m] = model_list[m].to(device[m])
    # 使用adam优化器
    optimizer[m] = optim.Adam(model_list[m].parameters(), lr=1e-3, weight_decay=1e-6)

    # scheduler[m] = LRS.MultiStepLR(optimizer[m], milestones=args.lr_decay_milestones, gamma=args.lr_decay_fact)

# 实例化m个module
module = {}

for m in range(num_split):
    module[m] = DeclModuleImpl(model=model_list[m], optimizer=optimizer[m], split_loc=m, num_split=num_split)


# test for feature forwarding
# feature = torch.randn(1, 3, 224, 224).to(device[0])
# for m in range(num_split):
#     if m != num_split - 1:
#         feature = module[m](feature)
#         feature = feature.to(device[m + 1])
#         print(feature.device)
#     else:
#     # 最后一个module只输出f的output
#         print(next(module[m].parameters()).device)
#         feature = module[m].get_feature(feature)
#
# print(feature)
# print(feature.size())