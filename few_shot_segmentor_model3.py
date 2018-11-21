"""Few-Shot_learning Segmentation"""

import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from data_utils import split_batch
# import torch.nn.functional as F
# from squeeze_and_excitation import squeeze_and_excitation as se


class SDnetConditioner(nn.Module):
    """
    A conditional branch of few shot learning regressing the parameters for the segmentor
    """

    def __init__(self, params):
        super(SDnetConditioner, self).__init__()
        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                       kernel_size=(1, 1),
                                       padding=(0, 0),
                                       stride=1)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        e1, out1, ind1 = self.encode1(input)
        e2, out2, ind2 = self.encode2(e1)
        e3, out3, ind3 = self.encode3(e2)

        bn = self.bottleneck(e3)

        d3 = self.decode1(bn, out3, ind3)
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, out2, ind2)
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode3(d2, out1, ind1)
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))
        logit = self.classifier.forward(d1)
        cls_w = self.sigmoid(logit)

        return d_w3, d_w2, d_w1, cls_w


class SDnetSegmentor(nn.Module):
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
        super(SDnetSegmentor, self).__init__()
        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        self.soft_max = nn.Softmax2d()

    def forward(self, inpt, weights=None):
        d_w3, d_w2, d_w1, cls_w = weights
        e1, out1, ind1 = self.encode1(inpt)
        e2, out2, ind2 = self.encode2(e1)
        e3, out3, ind3 = self.encode3(e2)

        bn = self.bottleneck(e3)

        d3 = self.decode1(bn, out3, ind3)
        d3 = torch.mul(d3, d_w3)
        d2 = self.decode2(d3, out2, ind2)
        d2 = torch.mul(d2, d_w2)
        d1 = self.decode3(d2, out1, ind1)
        d1 = torch.mul(d1, d_w1)
        logit = self.classifier.forward(d1)
        logit = torch.mul(logit, cls_w)
        prob = self.soft_max(logit)

        return prob


class FewShotSegmentorDoubleSDnet(nn.Module):
    '''
    Class Combining Conditioner and Segmentor for few shot learning
    '''

    def __init__(self, params):
        super(FewShotSegmentorDoubleSDnet, self).__init__()
        self.conditioner = SDnetConditioner(params)
        self.segmentor = SDnetSegmentor(params)
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
