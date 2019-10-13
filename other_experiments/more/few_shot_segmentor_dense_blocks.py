"""Few-Shot_learning Segmentation"""

import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from data_utils import split_batch
from squeeze_and_excitation import squeeze_and_excitation as se


class Conditioner(nn.Module):
    """
    A conditional branch of few shot learning regressing the parameters for the segmentor

    """

    def __init__(self, params):
        super(Conditioner, self).__init__()
        params['num_channels'] = 1
        self.genblock1 = sm.GenericBlock(params)
        params['num_channels'] = 64
        self.genblock2 = sm.GenericBlock(params)
        self.genblock3 = sm.GenericBlock(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'])
        self.tanh = nn.Tanh()

    def forward(self, input):
        o1 = self.genblock1(input)
        o2 = self.maxpool(o1)
        o3 = self.genblock2(o2)
        o4 = self.maxpool(o3)
        o5 = self.genblock3(o4)
        batch_size, num_channels, H, W = o1.size()
        o6, _ = o1.view(batch_size, num_channels, -1).max(dim=2)
        # o6 = self.tanh(o1.view(batch_size, num_channels, -1).mean(dim=2))
        batch_size, num_channels, H, W = o3.size()
        o7, _ = o3.view(batch_size, num_channels, -1).max(dim=2)
        # o7 = self.tanh(o3.view(batch_size, num_channels, -1).mean(dim=2))
        batch_size, num_channels, H, W = o5.size()
        o8, _ = o5.view(batch_size, num_channels, -1).max(dim=2)
        # o8 = self.tanh(o5.view(batch_size, num_channels, -1).mean(dim=2))
        return o6, o7, o8


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
        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.DenseBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)
        self.decode2 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)
        self.decode3 = sm.DecoderBlock(params, se_block_type=se.SELayer.NONE)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inpt, weights=None):
        w1, w2, w3 = weights if weights else (None, None, None)
        e1, out1, ind1 = self.encode1(inpt)
        e2, out2, ind2 = self.encode2(e1)
        e3, out3, ind3 = self.encode3(e2)

        bn = self.bottleneck.forward(e3)

        d3 = self.decode1(bn, out3, ind3, w1)
        d2 = self.decode2(d3, out2, ind2, w2)
        d1 = self.decode3(d2, out1, ind1, w3)
        logit = self.classifier.forward(d1)
        prob = self.sigmoid(logit)

        return prob

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class FewShotSegmentor(nn.Module):
    '''
    Class Combining Conditioner and Segmentor for few shot learning
    '''

    def __init__(self, params):
        super(FewShotSegmentor, self).__init__()
        self.conditioner = Conditioner(params)
        self.segmentor = Segmentor(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
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
