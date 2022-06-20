import torch.nn as nn

class IDeclModule(nn.Module):
    def __init__(self):
        super(IDeclModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError()

    def forward_nograd(self, x):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def get_grad(self):
        raise NotImplementedError()

    def set_output(self, output_1, output_2):
        raise NotImplementedError()

    def get_output(self):
        raise NotImplementedError()

    def set_input(self, input_1, input_2):
        raise NotImplementedError()

    def get_oldest_input(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        raise NotImplementedError()

    # used for the last module
    def get_feature(self, x):
        raise NotImplementedError()
    
    def get_update_count(self):
        raise NotImplementedError()

    def inc_update_count(self):
        raise NotImplementedError()

    def clear_update_count(self):
        raise NotImplementedError()

    def get_input_grad(self):
        raise NotImplementedError()

    def set_input_grad(self, input_grad_1, input_grad_2):
        raise NotImplementedError()

    def get_module_num(self):
        raise NotImplementedError()

    def is_last_layer(self):
        raise NotImplementedError()

    def is_first_layer(self):
        raise NotImplementedError()

    def set_loss(self, loss):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def set_dg(self, dg_1, dg_2):
        raise NotImplementedError()


    