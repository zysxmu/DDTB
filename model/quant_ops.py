#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

import collections
import math
import pdb
import random
import time
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function as F




def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8

def TorchRound():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply

class quant_weight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = TorchRound()

    def forward(self, input):

        if self.k_bits == 32:
            return input

        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = self.round(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight

class quant_weight_asym99(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight_asym99, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = TorchRound()

    def forward(self, input):
        # max_val = torch.max(input)
        # min_val = torch.min(input)

        if self.k_bits == 32:
            return input

        max_val = torch.tensor(np.percentile(input.detach().cpu().numpy(), 99)).cuda().float()
        min_val = torch.tensor(np.percentile(input.detach().cpu().numpy(), 1)).cuda().float()
        input = torch.max(torch.min(input, max_val), min_val)

        n = 2 ** self.k_bits - 1
        scale_factor = n / (max_val - min_val)
        zero_point = scale_factor * min_val
        zero_point = self.round(zero_point)
        zero_point += 2 ** (self.k_bits - 1)

        weight = scale_factor * input - zero_point
        q_weight = self.round(weight)
        q_weight = (q_weight + zero_point) / scale_factor

        return q_weight

class quant_activation(nn.Module):
    """
    Quantization function for quantize activation with maximum and minimum, only for gate.
    """

    def __init__(self, k_bits=8):
        super(quant_activation, self).__init__()
        self.k_bits = k_bits
        self.round = TorchRound()

    def forward(self, input):

        act = input.detach()
        max_val, min_val = torch.max(act), torch.min(act)

        n = 2 ** self.k_bits - 1
        scale_factor = n / (max_val - min_val)
        zero_point = scale_factor * min_val

        zero_point = self.round(zero_point)
        zero_point += 2 ** (self.k_bits - 1)

        act = scale_factor * act - zero_point
        q_act = self.round(act)
        q_act = (q_act + zero_point) / scale_factor
        return q_act

class DDTB_quant_act(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """
    def __init__(self, k_bits, ema_epoch=1, decay=0.9997):
        super(DDTB_quant_act, self).__init__()
        self.decay = decay
        self.k_bits = k_bits
        self.qmax = 2. ** (self.k_bits -1) -1.
        self.round = TorchRound()
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.ema_epoch = ema_epoch
        self.epoch = 1
        self.iteration = 0
        # self.max_val = 0
        self.ema_scale = 1
        self.register_buffer('max_val', torch.zeros(1))
        self.error = 0
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.alpha, 10)

    def _ema(self, x):
        max_val = torch.mean(torch.max(torch.max(torch.max(torch.abs(x),dim=1)[0],dim=1)[0],dim=1)[0])
        # max_val = torch.tensor(np.percentile(x.detach().cpu().numpy(), 99)).cuda().float()
        if self.epoch == 1:
            # print('aa')
            self.max_val = max_val
        else:
            # print('xx')
            self.max_val = (1.0-self.decay) * max_val + self.decay * self.max_val

    def forward(self, x):

        if self.k_bits == 32:
            if self.alpha==10:
                self._ema(x)
                self.alpha.data = self.max_val.unsqueeze(0)
                print(self.alpha)
            return x

        if self.epoch > self.ema_epoch or not self.training:
            act = torch.max(torch.min(x, self.alpha), -self.alpha)

        elif self.epoch <= self.ema_epoch and self.training:
            act = x
            if self.alpha == 10:
                self._ema(x)
                self.alpha.data = self.max_val.unsqueeze(0)

        act = act * self.qmax / self.alpha
        q_act = self.round(act)
        q_act = q_act * self.alpha / self.qmax
        self.error = torch.mean((q_act.detach() - x.detach()) ** 2).item()


        return q_act


class DDTB_quant_act_asym_dynamic_quantized(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """

    def __init__(self, k_bits, ema_epoch=1, decay=0.9997, inplanes=64, M=0):
        super(DDTB_quant_act_asym_dynamic_quantized, self).__init__()
        self.decay = decay
        self.k_bits = k_bits
        self.qmax = 2. ** (self.k_bits) - 1.
        self.round = TorchRound()
        self.alpha_upper = nn.Parameter(torch.Tensor(1))
        self.alpha_lower = nn.Parameter(torch.Tensor(1))
        self.ema_epoch = ema_epoch
        self.epoch = 1
        self.iteration = 0
        self.ema_scale = 1
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('min_val', torch.zeros(1))
        self.M = M

        self.first_stage_epoch = None
        self.test_only = False

        self.gate_bit = 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten_c = nn.Sequential(
            quant_activation(k_bits=self.gate_bit),
            QuantConv2d(inplanes, 32, kernel_size=1, stride=1, bias=False, k_bits=self.gate_bit),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            quant_activation(k_bits=self.gate_bit),
            QuantConv2d(32, 2, kernel_size=1, stride=1, bias=False, k_bits=self.gate_bit),
        )
        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()

        self.fp_max_list = []
        self.fp_min_list = []
        self.register_buffer('is_dynamic', torch.zeros(1))

    def reset_parameter(self):
        nn.init.constant_(self.alpha_upper, 10)
        nn.init.constant_(self.alpha_lower, -10)

    def _ema(self, x):

        max_val = torch.tensor(np.percentile(x.detach().cpu().numpy(), self.M)).cuda().float()
        min_val = torch.tensor(np.percentile(x.detach().cpu().numpy(), 100-self.M)).cuda().float()

        if max_val == min_val:
            print('max_val == min_val', max_val, min_val)
            import IPython
            IPython.embed()


        # if self.epoch == 1:
        if True:
            # print('aa')
            self.max_val = max_val
            self.min_val = min_val
        else:
            # print('xx')
            self.max_val = (1.0 - self.decay) * max_val + self.decay * self.max_val
            self.min_val = (1.0 - self.decay) * min_val + self.decay * self.min_val

    def forward(self, x):

        if self.k_bits == 32:
            self.fp_max_list.append(torch.max(x).detach().cpu().numpy())
            self.fp_min_list.append(torch.min(x).detach().cpu().numpy())
            return x
        if self.k_bits == 2:
            # only calibration once
            if self.training and self.alpha_upper == 10:
                print('clibration!')
                self._ema(x)
                self.alpha_upper.data = self.max_val.unsqueeze(0)
                self.alpha_lower.data = self.min_val.unsqueeze(0)

            self.fp_max_list.append(torch.max(x).detach().cpu().numpy())
            self.fp_min_list.append(torch.min(x).detach().cpu().numpy())


        if not self.is_dynamic:
            act = torch.max(torch.min(x, self.alpha_upper), self.alpha_lower)
            n = 2 ** self.k_bits - 1

            scale_factor = n / (self.alpha_upper - self.alpha_lower)
            zero_point = scale_factor * self.alpha_lower
            zero_point = self.round(zero_point)
            zero_point += 2 ** (self.k_bits - 1)

            act = scale_factor * act - zero_point
            q_act = self.round(act)
            q_act = (q_act + zero_point) / scale_factor

            if torch.any(torch.isnan(q_act)):
                print('nan !')
                import IPython
                IPython.embed()

            return q_act

        if self.epoch > self.ema_epoch or not self.training:

            if self.epoch < self.first_stage_epoch and not self.test_only:
                context = self.avg_pool(x.detach())  # [N, C, 1, 1]
                # transform
                c_in = self.atten_c(context)  # [N, 1, 1, 1]
                # scale
                scale = self.sigmoid(c_in) * 2

                self.sau = scale[:, 0, :, :]
                self.sal = scale[:, 1, :, :]

                act = torch.max(torch.min(x, self.alpha_upper), self.alpha_lower)
                n = 2 ** self.k_bits - 1

                scale_factor = n / (self.alpha_upper - self.alpha_lower)
                zero_point = scale_factor * self.alpha_lower
                zero_point = self.round(zero_point)
                zero_point += 2 ** (self.k_bits - 1)

                act = scale_factor * act - zero_point
                q_act = self.round(act)
                q_act = (q_act + zero_point) / scale_factor

                if torch.any(torch.isnan(q_act)):
                    print('nan !')
                    import IPython
                    IPython.embed()

            else:
                context = self.avg_pool(x.detach())  # [N, C, 1, 1]
                # transform
                c_in = self.atten_c(context)  # [N, 1, 1, 1]
                # scale
                scale = self.sigmoid(c_in) * 2

                self.sau = scale[:, 0, :, :]
                self.sal = scale[:, 1, :, :]

                act = torch.max(torch.min(x, (scale[:, 0, :, :] * self.alpha_upper).unsqueeze(dim=1)),
                                (scale[:, 1, :, :] * self.alpha_lower).unsqueeze(dim=1))
                n = 2 ** self.k_bits - 1

                scale_factor = n / (
                            scale[:, 0, :, :] * self.alpha_upper - scale[:, 1, :, :] * self.alpha_lower).unsqueeze(
                    dim=1)
                zero_point = scale_factor * (scale[:, 1, :, :] * self.alpha_lower).unsqueeze(dim=1)
                zero_point = self.round(zero_point)
                zero_point += 2 ** (self.k_bits - 1)

                act = scale_factor * act - zero_point
                q_act = self.round(act)
                q_act = (q_act + zero_point) / scale_factor

                if torch.any(torch.isnan(q_act)):
                    print('nan !')
                    import IPython
                    IPython.embed()

        elif self.epoch <= self.ema_epoch and self.training:
            pirnt('xxxxxx')
            print('xxxxxx')
            act = x
            self._ema(x)
            self.alpha_upper.data = self.max_val.unsqueeze(0)
            self.alpha_lower.data = self.min_val.unsqueeze(0)

            n = 2 ** self.k_bits - 1
            scale_factor = n / (self.alpha_upper - self.alpha_lower)
            zero_point = scale_factor * self.alpha_lower
            zero_point = self.round(zero_point)
            zero_point += 2 ** (self.k_bits - 1)

            act = scale_factor * act - zero_point
            q_act = self.round(act)
            q_act = (q_act + zero_point) / scale_factor

        return q_act


class QuantConv2d_asym(nn.Module):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=32,):
        super(QuantConv2d_asym, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quant_weight_asym = quant_weight_asym(k_bits = k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        return nn.functional.conv2d(input, self.quant_weight_asym(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantConv2d(nn.Module):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=32,):
        super(QuantConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quant_weight = quant_weight(k_bits = k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        return nn.functional.conv2d(input, self.quant_weight(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

class quant_weight_asym(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight_asym, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = TorchRound()

    def forward(self, input):
        # max_val = torch.max(input)
        # min_val = torch.min(input)

        if self.k_bits == 32:
            return input

        max_val = torch.max(input.detach())
        min_val = torch.min(input.detach())

        n = 2 ** self.k_bits - 1
        scale_factor = n / (max_val - min_val)
        zero_point = scale_factor * min_val
        zero_point = self.round(zero_point)
        zero_point += 2 ** (self.k_bits - 1)

        weight = scale_factor * input - zero_point
        q_weight = self.round(weight)
        q_weight = (q_weight + zero_point) / scale_factor

        return q_weight

class QuantConv2d_asym99(nn.Module):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=32,):
        super(QuantConv2d_asym99, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quant_weight_asym = quant_weight_asym99(k_bits = k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        return nn.functional.conv2d(input, self.quant_weight_asym(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)




def conv3x3(in_channels, out_channels,kernel_size=3,stride=1,padding =1,bias= True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def quant_conv3x3_asym(in_channels, out_channels,kernel_size=3,padding = 1,stride=1,k_bits=32,bias = False):
    return QuantConv2d_asym(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride,padding=padding,k_bits=k_bits,bias = bias)

def quant_conv3x3_asym99(in_channels, out_channels,kernel_size=3,padding = 1,stride=1,k_bits=32,bias = False):
    return QuantConv2d_asym99(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride,padding=padding,k_bits=k_bits,bias = bias)

