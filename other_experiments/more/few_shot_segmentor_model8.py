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
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.channel_conv_e1 = nn.Linear(params['num_filters'], params['num_filters'], bias=True)

        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.channel_conv_e2 = nn.Linear(params['num_filters'], params['num_filters'], bias=True)

        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.channel_conv_e3 = nn.Linear(params['num_filters'], params['num_filters'], bias=True)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.channel_conv_bn = nn.Linear(params['num_filters'], params['num_filters'], bias=True)
        params['num_channels'] = 128
        self.decode1 = sm.SDnetDecoderBlock(params)
        #self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
        #                                 kernel_size=(1, 1),
        #                                 padding=(0, 0),
        #                                 stride=1)
        #self.channel_conv_d1 = nn.Linear(params['num_filters'], params['num_filters'], bias=True)
        self.decode2 = sm.SDnetDecoderBlock(params)
        #self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
        #                                 kernel_size=(1, 1),
        #                                 padding=(0, 0),
        #                                 stride=1)
        #self.channel_conv_d2 = nn.Linear(params['num_filters'], params['num_filters'], bias=True)
        self.decode3 = sm.SDnetDecoderBlock(params)
        #self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
        #                                 kernel_size=(1, 1),
        #                                 padding=(0, 0),
        #                                 stride=1)
        #self.channel_conv_d3 = nn.Linear(params['num_filters'], params['num_filters'], bias=True)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()
        self.channel_conv_cls = nn.Linear(params['num_class'], params['num_class'], bias=True)

    def forward(self, input):
        e1, out1, ind1 = self.encode1(input)
        num_batch, ch, _, _ = e1.size()
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e_c1 = self.sigmoid(self.channel_conv_e1(e1.view(num_batch, ch, -1).mean(dim=2)))
        e_c1 = e_c1.view(num_batch, ch, 1, 1)

        e2, out2, ind2 = self.encode2(e1)
        num_batch, ch, _, _ = e2.size()
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e_c2 = self.sigmoid(self.channel_conv_e2(e2.view(num_batch, ch, -1).mean(dim=2)))
        e_c2 = e_c2.view(num_batch, ch, 1, 1)

        e3, out3, ind3 = self.encode3(e2)
        num_batch, ch, _, _ = e3.size()
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e_c3 = self.sigmoid(self.channel_conv_e3(e3.view(num_batch, ch, -1).mean(dim=2)))
        e_c3 = e_c3.view(num_batch, ch, 1, 1)

        bn = self.bottleneck(e3)
        num_batch, ch, _, _ = bn.size()
        bn_w = self.sigmoid(self.squeeze_conv_bn(bn))
        bn_c = self.sigmoid(self.channel_conv_bn(bn.view(num_batch, ch, -1).mean(dim=2)))
        bn_c = bn_c.view(num_batch, ch, 1, 1)

        d3 = self.decode1(bn, out3, ind3)
        #num_batch, ch, _, _ = d3.size()
        #d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        #d_c3 = self.sigmoid(self.channel_conv_d3(d3.view(num_batch, ch, -1).mean(dim=2)))
        #d_c3 = d_c3.view(num_batch, ch, 1, 1)

        d2 = self.decode2(d3, out2, ind2)
        #num_batch, ch, _, _ = d2.size()
        #d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        #d_c2 = self.sigmoid(self.channel_conv_d2(d2.view(num_batch, ch, -1).mean(dim=2)))
        #d_c2 = d_c2.view(num_batch, ch, 1, 1)

        d1 = self.decode3(d2, out1, ind1)
        #num_batch, ch, _, _ = d1.size()
        #d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))
        #d_c1 = self.sigmoid(self.channel_conv_d1(d1.view(num_batch, ch, -1).mean(dim=2)))
        #d_c1 = d_c1.view(num_batch, ch, 1, 1)

        logit = self.classifier.forward(d1)
        num_batch, ch, _, _ = logit.size()
        cls_w = self.sigmoid(logit)
        cls_c = self.sigmoid(self.channel_conv_cls(logit.view(num_batch, ch, -1).mean(dim=2)))
        cls_c = cls_c.view(num_batch, ch, 1, 1)

        #space_w = (e_w1, e_w2, e_w3, bn_w, d_w3, d_w2, d_w1, cls_w)
        #channel_w = (e_c1, e_c2, e_c3, bn_c, d_c3, d_c2, d_c1, cls_c)

        space_w = (e_w1, e_w2, e_w3, bn_w ,cls_w)
        channel_w = (e_c1, e_c2, e_c3, bn_c, cls_c)

        return space_w, channel_w


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
        self.soft_max = nn.Softmax2d()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inpt, weights=None):
        space_w, channel_w = weights
        #e_w1, e_w2, e_w3, bn_w, d_w3, d_w2, d_w1, cls_w = space_w
        #e_c1, e_c2, e_c3, bn_c, d_c3, d_c2, d_c1, cls_c = channel_w
        e_w1, e_w2, e_w3, bn_w, cls_w = space_w
        e_c1, e_c2, e_c3, bn_c, cls_c = channel_w

        e1, out1, ind1 = self.encode1(inpt)
        es1 = torch.mul(e1, e_w1)
        ec1 = torch.mul(e1, e_c1)
        e1 = torch.max(es1, ec1)

        e2, out2, ind2 = self.encode2(e1)
        es2 = torch.mul(e2, e_w2)
        ec2 = torch.mul(e2, e_c1)
        e2 = torch.max(es2, ec2)

        e3, out3, ind3 = self.encode3(e2)
        es3 = torch.mul(e3, e_w3)
        ec3 = torch.mul(e3, e_c3)
        e3 = torch.max(es3, ec3)

        bn = self.bottleneck(e3)
        bns = torch.mul(bn, bn_w)
        bnc = torch.mul(bn, bn_c)
        bn = torch.max(bns, bnc)

        d3 = self.decode1(bn, out3, ind3)
        #ds3 = torch.mul(d3, d_w3)
        #dc3 = torch.mul(d3, d_c3)
        #d3 = torch.max(ds3, dc3)

        d2 = self.decode2(d3, out2, ind2)
        #ds2 = torch.mul(d2, d_w2)
        #dc2 = torch.mul(d2, d_c2)
        #d2 = torch.max(ds2, dc2)

        d1 = self.decode3(d2, out1, ind1)
        #ds1 = torch.mul(d1, d_w1)
        #dc1 = torch.mul(d1, d_c1)
        #d1 = torch.max(ds1, dc1)

        logit = self.classifier.forward(d1)
        logit_s = torch.mul(logit, cls_w)
        logit_c = torch.mul(logit, cls_c)
        logit = torch.max(logit_s, logit_c)

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
