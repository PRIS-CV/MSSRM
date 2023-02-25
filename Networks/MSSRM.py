import torch.nn as nn
import torch
import math
from torchvision import models
from utils import save_net, load_net
import torch.nn.functional as F
from collections import OrderedDict


def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1


class MSSRM(nn.Module):
    def __init__(self, load_weights=False, upscale='x2'):
        super(MSSRM, self).__init__()

        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1280, 1280, kernel_size=1, padding=1)
        )

        if upscale == 'x2':
            self.espcn_part = nn.Sequential(
                nn.Conv2d(1280, 3 * (4 ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(4)
            )
        elif upscale == 'x4':
            self.espcn_part = nn.Sequential(
                nn.Conv2d(1280, 3 * (8 ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(8)
            )
        self.pixel2 = nn.Sequential(
            nn.Conv2d(512, 512 * (2 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(2)
        )

        self.pixel4 = nn.Sequential(
            nn.Conv2d(1, 1 * (4 ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(4)
        )

        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscore8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.upscore5 = nn.UpsamplingBilinear2d(scale_factor=8)

        #self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][
                                                                             1].data[:]

    def forward(self, x, gt, gt_sr, phase = 'test'):
        pd = (4, 4, 4, 4)
        x = F.pad(x, pd, 'constant')
        gt = torch.unsqueeze(gt, 1)
        #stage1
        conv1 = self.frontend[0:4](x)

        #stage2
        conv2 = self.frontend[4:9](conv1)

        #stage3
        conv3 = self.frontend[9:16](conv2)

        #stage4
        conv4 = self.frontend[16:23](conv3)

        #stage5
        conv5 = self.backend(conv4)

        conv5_upscore2 = self.upscore2(conv5)
        if phase == 'train':
            conv4_upscore = self.upscore2(conv4)
            conv3 = crop(conv3, conv5_upscore2)

            espcn_input = torch.cat((conv3, conv4_upscore, conv5_upscore2), 1)

            espcn_input = self.upscore2(espcn_input)
            x_sr = self.espcn_part(espcn_input)
            x_sr = crop(x_sr, gt_sr)

        output = self.reg_layer(conv5_upscore2)
        output = self.upscore4(output)

        output = crop(output, gt)


        if phase == 'train':
            return output, x_sr
        else:
            return output
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    #print(cfg)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                
