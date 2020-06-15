import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from models.dsam_layers import center_crop

class av_module(nn.Module):

    def __init__(self, rgb_nfilters, audio_nfilters, img_size, temp_size, hidden_layers):

        super(av_module, self).__init__()

        self.rgb_nfilters = rgb_nfilters
        self.audio_nfilters = audio_nfilters
        self.hidden_layers = hidden_layers
        self.out_layers = 64
        self.img_size = img_size
        self.avgpool_rgb = nn.AvgPool3d((temp_size, 1, 1), stride=1)
        # Make the layers numbers equal
        self.relu = nn.ReLU()
        self.affine_rgb = nn.Linear(rgb_nfilters, hidden_layers)
        self.affine_audio = nn.Linear(audio_nfilters, hidden_layers)
        self.w_a_rgb = nn.Bilinear(hidden_layers, hidden_layers, self.out_layers, bias=True)
        self.upscale_ = nn.Upsample(scale_factor=8, mode='bilinear')


    def forward(self, rgb, audio, crop_h, crop_w):

        self.crop_w = crop_w
        self.crop_h = crop_h
        dgb = rgb[:,:,rgb.shape[2]//2-1:rgb.shape[2]//2+1,:,:]
        rgb = self.avgpool_rgb(dgb).squeeze(2)
        rgb = rgb.permute(0, 2, 3, 1)
        rgb = rgb.view(rgb.size(0), -1, self.rgb_nfilters)
        rgb = self.affine_rgb(rgb)
        rgb = self.relu(rgb)
        audio1 = self.affine_audio(audio[0].squeeze())
        audio1 = self.relu(audio1)

        a_rgb_B = self.w_a_rgb(rgb.contiguous(), audio1.unsqueeze(1).expand(-1, self.img_size[0] * self.img_size[1], -1).contiguous())
        sal_bilin = a_rgb_B
        sal_bilin = sal_bilin.view(-1, self.img_size[0], self.img_size[1], self.out_layers)
        sal_bilin = sal_bilin.permute(0, 3, 1, 2)
        sal_bilin = center_crop(self.upscale_(sal_bilin), self.crop_h, self.crop_w)

        return sal_bilin






