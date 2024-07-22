import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from .util import SPADEResnetBlock

class SmirkGenerator(nn.Module):

    def __init__(self, in_channels=6, out_channels=3, init_features=32, res_blocks=3):
        super(SmirkGenerator, self).__init__()

        features = init_features
        self.encoder1 = SmirkGenerator._block(in_channels, features, name="enc1", kernel_size=7) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  #256,256,32
        self.encoder2 = SmirkGenerator._block(features, features * 2, name="enc2", kernel_size=7) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #128,128,64
        self.encoder3 = SmirkGenerator._block(features * 2, features * 4, name="enc3") 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #64,64,128
        self.encoder4 = SmirkGenerator._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  #32,32,256
        self.encoder5 = SmirkGenerator._block(features * 8, features * 8, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  #16,16,256
        

        self.bottleneck = SmirkGenerator._block(features * 8, features * 16, name="bottleneck") #16,16,512
        # add multiple (K) resnet blocks as modulelist
        
        ##resblock
        # resnet_blocks = []
        # for _ in range(res_blocks):
        #     resnet_blocks.append(
        #             ResnetBlock(features * 16, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False)
        #     )
        # self.resnet_blocks = nn.ModuleList(resnet_blocks) #(16,16,512)

        self.upconv5 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        ) #(32,32,256),concat→（32，32，512）
        self.decoder5 = SmirkGenerator._block(features * 16, features * 8, name="dec5") #(32,32,256)
        
        self.upconv4 = nn.ConvTranspose2d(
            features * 8, features * 8, kernel_size=2, stride=2
        ) #(64,64,256)，concat→（64，64，512）
        self.decoder4 = SmirkGenerator._block(features * 16, features * 8, name="dec5") #(64,64,256)
        
        self.SPADEdecoder = SPADEDecoder()

    def forward(self, x, token_list):

        #if use_mask:
        #    mask = (x[:, 3:] == 0).all(dim=1, keepdim=True).float()

        enc1 = self.encoder1(x,token_list[4]) #32,512,512

        enc2 = self.encoder2(self.pool1(enc1),token_list[3]) #64,256,256

        enc3 = self.encoder3(self.pool2(enc2),token_list[2]) #128,128,128

        enc4 = self.encoder4(self.pool3(enc3),token_list[1]) #256,64,64

        enc5 = self.encoder5(self.pool4(enc4),token_list[0]) #256,32,32


        bottleneck = self.bottleneck(self.pool5(enc5),token_list[0]) #256,16,16

        # for resnet_block in self.resnet_blocks:
        #     bottleneck = resnet_block(bottleneck) #16,16,256

        dec5 = self.upconv5(bottleneck) #256,32,32
        dec5 = torch.cat((dec5, enc5), dim=1) #512,32,32
        dec5 = self.decoder5(dec5,token_list[0]) #256,32,32

        dec4 = self.upconv4(dec5)  #256,64,64
        dec4 = torch.cat((dec4, enc4), dim=1) #512,64,64
        dec4 = self.decoder4(dec4,token_list[0]) #256,64,64
        
        #SPADE decoder
        out = self.SPADEdecoder(dec4,token_list)

        # do this better!
        #if use_mask:
        #    out = out[:, :3] * mask + out[:, 3:] * (1 - mask)
        #else:
        #    out = out[:,:3]

        return out

    
    def _block(in_channels, features, name, kernel_size=3):
        return AdainBlock(in_channels, features, 512, kernel_size)
    # @staticmethod
    # def _block(in_channels, features, name):
    # @staticmethod
    # def _block(in_channels, features, name):
    #     return nn.Sequential(
    #         OrderedDict(
    #             [
    #                 (
    #                     name + "conv1",
    #                     nn.Conv2d(
    #                         in_channels=in_channels,
    #                         out_channels=features,
    #                         kernel_size=3,
    #                         padding=1,
    #                         bias=False,
    #                     ),
    #                 ),
    #                 (name + "norm1", nn.BatchNorm2d(num_features=features)),
    #                 (name + "relu1", nn.ReLU(inplace=True)),
    #                 (
    #                     name + "conv2",
    #                     nn.Conv2d(
    #                         in_channels=features,
    #                         out_channels=features,
    #                         kernel_size=3,
    #                         padding=1,
    #                         bias=False,
    #                     ),
    #                 ),
    #                 (name + "norm2", nn.BatchNorm2d(num_features=features)),
    #                 (name + "relu2", nn.ReLU(inplace=True)),
    #             ]
    #         )
    #     )

class AdainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, token_len, kernel_size):
        super(AdainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = (kernel_size-1) // 2, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.adain1 = ADAIN(out_channels, token_len)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.adain2 = ADAIN(out_channels, token_len)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, token):
        out = self.conv1(x)
        out = self.norm1(out)
        # print('-----------')
        # print(token.shape)
        out = self.adain1(out, token)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.adain2(out, token)
        out = self.relu2(out)
        return out



class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class SPADEDecoder(nn.Module):
    def __init__(self, upscale=2, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )

    def forward(self, feature, token_list):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg, token_list[0])
        x = self.G_middle_1(x, seg, token_list[0])
        x = self.G_middle_2(x, seg, token_list[1])
        x = self.G_middle_3(x, seg, token_list[1])
        x = self.G_middle_4(x, seg, token_list[2])
        x = self.G_middle_5(x, seg, token_list[2])

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg, token_list[3])  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg, token_list[4])  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x