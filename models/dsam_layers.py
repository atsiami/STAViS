import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    # fixed indexing for PyTorch 0.4
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])

class dsam_score_dsn(nn.Module):

    def __init__(self, prev_layer, prev_nfilters, prev_nsamples):

        super(dsam_score_dsn, self).__init__()
        i = prev_layer
        self.avgpool = nn.AvgPool3d((prev_nsamples, 1, 1), stride=1)
        # Make the layers of the preparation step
        self.side_prep = nn.Conv2d(prev_nfilters, 16, kernel_size=3, padding=1)
        # Make the layers of the score_dsn step
        self.score_dsn = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.upscale_ = nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)
        self.upscale = nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)

    def forward(self, x, crop_h, crop_w):

        self.crop_h = crop_h
        self.crop_w = crop_w
        x = self.avgpool(x).squeeze(2)
        side_temp = self.side_prep(x)
        side = center_crop(self.upscale(side_temp), self.crop_h, self.crop_w)
        side_out_tmp = self.score_dsn(side_temp)
        side_out = center_crop(self.upscale_(side_out_tmp), self.crop_h, self.crop_w)
        return side, side_out, side_out_tmp


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def spatial_softmax(x):
    x = torch.exp(x)
    sum_batch = torch.sum(torch.sum(x, 2, keepdim=True), 3, keepdim=True)
    x_soft = torch.div(x,sum_batch)
    return x_soft

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data