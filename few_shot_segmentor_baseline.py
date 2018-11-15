"""Few-Shot_learning Segmentation"""

import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from data_utils import split_batch
import torch.nn.functional as F
from squeeze_and_excitation import squeeze_and_excitation as se


class Conditioner(nn.Module):
    """
    A conditional branch of few shot learning regressing the parameters for the segmentor
    """

    def __init__(self, params):
        super(Conditioner, self).__init__()
        params['num_channels'] = 1
        params['num_filters'] = 32
        self.genblock1 = sm.GenericBlock(params)
        self.squeeze_conv1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        params['num_channels'] = params['num_filters']
        params['num_filters'] = 64
        self.genblock2 = sm.GenericBlock(params)
        self.squeeze_conv2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        params['num_channels'] = params['num_filters']
        params['num_filters'] = 128
        self.genblock3 = sm.GenericBlock(params)
        self.squeeze_conv3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        params['num_channels'] = params['num_filters']
        params['num_filters'] = 256
        self.genblock4 = sm.GenericBlock(params)
        self.squeeze_conv4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        params['num_channels'] = params['num_filters']
        params['num_filters'] = 512
        self.genblock5 = sm.GenericBlock(params)
        self.squeeze_conv5 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Block - 1
        o1 = self.genblock1(input)
        o1_1 = self.squeeze_conv1(o1)
        o1_1 = self.sigmoid(F.interpolate(o1_1, scale_factor=(1 / 8, 1 / 8)))
        # Block - 2
        o2 = self.maxpool(o1)
        o2 = self.genblock2(o2)
        o2_2 = self.squeeze_conv2(o2)
        o2_2 = self.sigmoid(F.interpolate(o2_2, scale_factor=(1 / 2, 1 / 2)))
        # Block - 3
        o3 = self.maxpool(o2)
        o3 = self.genblock3(o3)
        o3_3 = self.squeeze_conv3(o3)
        o3_3 = self.sigmoid(F.interpolate(o3_3, scale_factor=(2, 2)))
        # Block - 4
        o4 = self.maxpool(o3)
        o4 = self.genblock4(o4)
        o4_4 = self.squeeze_conv4(o4)
        o4_4 = self.sigmoid(F.interpolate(o4_4, scale_factor=(8, 8)))
        # Block - 5
        o5 = self.maxpool(o4)
        o5 = self.genblock5(o5)
        o5_5 = self.squeeze_conv5(o5)
        o5_5 = self.sigmoid(F.interpolate(o5_5, scale_factor=(16, 16)))
        return o1_1, o2_2, o3_3, o4_4, o5_5


# class Conditioner(nn.Module):
#     """
#     A conditional branch of few shot learning regressing the parameters for the segmentor
#
#     """
#
#     def __init__(self, params):
#         super(Conditioner, self).__init__()
#         params['num_channels'] = 1
#         self.genblock1 = sm.GenericBlock(params)
#         params['num_channels'] = 64
#         self.genblock2 = sm.GenericBlock(params)
#         self.genblock3 = sm.GenericBlock(params)
#         self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'])
#         self.tanh = nn.Tanh()
#
#     def forward(self, input):
#         o1 = self.genblock1(input)
#         o2 = self.maxpool(o1)
#         o3 = self.genblock2(o2)
#         o4 = self.maxpool(o3)
#         o5 = self.genblock3(o4)
#         batch_size, num_channels, H, W = o1.size()
#         # o6 = o1.view(batch_size, num_channels, -1).mean(dim=2)
#         # o6 = self.tanh(o1.view(batch_size, num_channels, -1).mean(dim=2))
#         batch_size, num_channels, H, W = o3.size()
#         # o7 = o3.view(batch_size, num_channels, -1).mean(dim=2)
#         # o7 = self.tanh(o3.view(batch_size, num_channels, -1).mean(dim=2))
#         batch_size, num_channels, H, W = o5.size()
#         o8,_ = o5.view(batch_size, num_channels, -1).max(dim=2)
#         # o8 = self.tanh(o5.view(batch_size, num_channels, -1).mean(dim=2))
#         return o8


class Segmentor(nn.Module):
    """
    Segmentor Code

    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':1
        'se_block': True,
        'drop_out':0
    }

    """

    def __init__(self, params):
        super(Segmentor, self).__init__()
        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params)
        self.decode2 = sm.DecoderBlock(params)
        self.decode3 = sm.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inpt, weights=None):
        w1, w2, w3, w4, w5 = weights
        # w1 = w1.unsqueeze(dim=1)
        e1, out1, ind1 = self.encode1(inpt)
        e2, out2, ind2 = self.encode2(e1)
        e3, out3, ind3 = self.encode3(e2)

        bn = self.bottleneck.forward(e3)
        bn = torch.mul(bn, w1)

        d3 = self.decode1(bn, out3, ind3)
        d3 = torch.mul(d3, w2)
        d2 = self.decode2(d3, out2, ind2)
        d2 = torch.mul(d2, w3)
        d1 = self.decode3(d2, out1, ind1)
        d1 = torch.mul(d1, w4)
        logit = self.classifier.forward(d1)
        logit = torch.mul(logit, w5)
        prob = self.sigmoid(logit)

        return prob


class FewShotSegmentorBaseLine(nn.Module):
    '''
    Class Combining Conditioner and Segmentor for few shot learning
    '''

    def __init__(self, params):
        super(FewShotSegmentorBaseLine, self).__init__()
        self.conditioner = Conditioner(params)
        self.segmentor = Segmentor(params)
        self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        self.weights = []

    def _print_grads(self, x):
        print("Median of gradient --- " + str(x.median().item()))
        # print("Max" + str(x.max()))
        # print("Min" + str(x.min()))

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        # weights.register_hook(self._print_grads)
        segment = self.segmentor(input2, weights)
        return segment

    def enable_test_dropout(self):
        attr_dict = self.__dict__['_modules']
        for i in range(1, 5):
            encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def predict(self, X, y, query_label, device=0, enable_dropout=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()
        input1, input2, y2 = split_batch(X, y, query_label)
        input1, input2, y2 = to_cuda(input1, device), to_cuda(input2, device), to_cuda(y2, device)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(input1, input2)

        # max_val, idx = torch.max(out, 1)
        idx = out > 0.5
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx
        return prediction


def to_cuda(X, device):
    if type(X) is np.ndarray:
        X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
    elif type(X) is torch.Tensor and not X.is_cuda:
        X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)
    return X
