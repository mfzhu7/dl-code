## 本文件实现了基础版本的循环神经网络，参考代码来源于李沐老师
## 数据集也参考来自李沐老师的代码


import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)



class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device_name):
        self.vocab_size = vocab_size 
        self.num_hiddens = num_hiddens
        self.device = device_name
        self.param  = self.params(vocab_size, num_hiddens, self.device)

    def normal(self, shape):
        return torch.randn(size=shape)

    def params(self, vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        W_xh = self.normal((num_inputs, num_hiddens))
        W_hh = self.normal((num_hiddens,num_hiddens))
        b_h = torch.zeros(num_hiddens,device=self.device)

        W_hq = self.normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=self.device)

        params = [W_xh, W_hh, b_h, W_hq, b_q]

        for param in params:
            param.requires_grad_(True)
        
        return params 


test = RNNModelScratch(28,512, d2l.try_gpu())
print(test.params)



