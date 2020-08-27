import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from functools import reduce
from models.dsam_layers import dsam_score_dsn, interp_surgery, spatial_softmax
from models.av_fusion import av_module

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))



def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 audiovisual=True):

        self.audiovisual = audiovisual
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)

        score_dsn = nn.modules.ModuleList()
        in_channels_dsn = [64,
                           64  * block.expansion,
                           128 * block.expansion,
                           256 *block.expansion]
        temp_size_prev = [sample_duration,
                          int(sample_duration / 2),
                          int(sample_duration / 4),
                          int(sample_duration /8)]
        temp_img_size_prev = [int(sample_size / 2),
                              int(sample_size / 4),
                              int(sample_size / 8),
                              int(sample_size / 16)]
        for i in range(1,5):
            score_dsn.append(dsam_score_dsn(i, in_channels_dsn[i-1], temp_size_prev[i-1]))
        self.score_dsn = score_dsn

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.fuseav = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.soundnet8 = nn.Sequential(  # Sequential,
            nn.Conv2d(1, 16, (1, 64), (1, 2), (0, 32)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 8), (1, 8)),
            nn.Conv2d(16, 32, (1, 32), (1, 2), (0, 16)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 8), (1, 8)),
            nn.Conv2d(32, 64, (1, 16), (1, 2), (0, 8)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (1, 8), (1, 2), (0, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (1, 4), (1, 2), (0, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(256, 512, (1, 4), (1, 2), (0, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (1, 4), (1, 2), (0, 2)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

        self.fusion3 = av_module(in_channels_dsn[2],
                                1024,
                                [temp_img_size_prev[2], temp_img_size_prev[2]],
                                temp_size_prev[3],
                                128)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)
            if isinstance(m ,nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Bilinear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.fuse.bias.data = torch.tensor([-6.0])
        self.fuseav.bias.data = torch.tensor([-6.0])
        for i in range(0, 4):
           self.score_dsn[i].score_dsn.bias.data = torch.tensor([-6.0])


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, aud):

        aud = self.soundnet8(aud)
        aud = [aud]

        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        side = []
        side_out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        (tmp, tmp_, att_tmp) = self.score_dsn[0](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)
        x =torch.mul(1+att, x)
        x = self.maxpool(x)

        x = self.layer1(x)

        (tmp, tmp_, att_tmp) = self.score_dsn[1](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)
        x = torch.mul(1+att, x)
        x = self.layer2(x)

        if self.audiovisual:
            y = self.fusion3(x, aud, crop_h, crop_w)
        (tmp, tmp_, att_tmp) = self.score_dsn[2](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)
        x = torch.mul(1+att, x)
        x = self.layer3(x)

        (tmp, tmp_, att_tmp) = self.score_dsn[3](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)

        out = torch.cat(side[:], dim=1)

        if self.audiovisual:
            appendy = torch.cat((out, y), dim=1)
            x_out = self.fuseav(appendy)
            side_out = []
        else:
            x_out = self.fuse(out)
        side_out.append(x_out)

        x_out = {'sal': side_out}

        return x_out


def get_fine_tuning_parameters(model, lr, wd):

    ft_module_names_global = []
    ft_module_names_global.append('conv1')
    ft_module_names_global.append('bn1')
    for i in range(1, 4):
        ft_module_names_global.append('layer{}'.format(i))

    ft_module_names_sal = []
    ft_module_names_sal.append('score_dsn')
    ft_module_names_sal.append('fuse')
    ft_module_names_sal.append('fuseav')

    ft_module_names_soundnet = []
    ft_module_names_soundnet.append('soundnet8')

    ft_module_names_fusion = []
    ft_module_names_fusion.append('fusion3')

    parameters = {'global':[], 'sal':[], 'other':[], 'sound':[], 'fusion':[]}
    name_parameters = {'global': [], 'sal': [], 'other': [], 'sound':[], 'fusion':[]}
    isfound = False
    for k, v in model.named_parameters():
        for ft_module in ft_module_names_global:
            if ft_module in k:
                parameters['global'].append({'params': v})
                name_parameters['global'].append(k)
                isfound = True
                break
        for ft_module in ft_module_names_soundnet:
            if ft_module in k:
                parameters['sound'].append({'params': v})
                name_parameters['sound'].append(k)
                isfound = True
                break
        for ft_module in ft_module_names_fusion:
            if ft_module in k:
                if 'upscale' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': 0, 'initial_lr': 0})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                if 'upscale_' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': 0, 'initial_lr': 0})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                parameters['fusion'].append({'params': v})
                name_parameters['fusion'].append(k)
                isfound = True
                break
        for ft_module in ft_module_names_sal:
            if ft_module in k:
                if 'side_prep' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'weight_decay': wd, 'lr': lr, 'initial_lr': lr})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                    elif 'bias' in k:
                        parameters['sal'].append({'params': v, 'lr': 2 * lr, 'initial_lr': 2 * lr})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                if 'score_dsn' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': lr / 10, 'weight_decay': wd, 'initial_lr': lr / 10})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                    elif 'bias' in k:
                        parameters['sal'].append({'params': v, 'lr': 2 * lr / 10, 'initial_lr': 2 * lr / 10})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                if 'upscale' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': 0, 'initial_lr': 0})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                if 'upscale_' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': 0, 'initial_lr': 0})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                if 'fuse' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': lr / 10, 'initial_lr': lr / 10, 'weight_decay': wd})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                    elif 'bias' in k:
                        parameters['sal'].append({'params': v, 'lr': 2 * lr / 10, 'initial_lr': 2 * lr / 10})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                if 'fuseav' in k:
                    if 'weight' in k:
                        parameters['sal'].append({'params': v, 'lr': lr / 10, 'initial_lr': lr / 10, 'weight_decay': wd})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
                    elif 'bias' in k:
                        parameters['sal'].append({'params': v, 'lr': 2 * lr / 10, 'initial_lr': 2 * lr / 10})
                        name_parameters['sal'].append(k)
                        isfound = True
                        break
        if isfound == False:
            parameters['other'].append({'params': v})
            name_parameters['other'].append(k)

    return parameters, name_parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
