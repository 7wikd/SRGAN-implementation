from torch import nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        disc=False,
        use_act=True,
        use_bn=True,
        **kwargs
    ):
        super().__init__()
        self.use_act=use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.act = (
            nn.LeakyReLU(0.2,inplace=True) if disc else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self,x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels* scale_factor **2,3,1,1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)
    
    def forward(self,x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels, 
            in_channels,
            use_act=True,
            use_bn=True,
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.block2 = ConvBlock(
            in_channels, 
            in_channels,
            use_act=False,
            kernel_size=3,
            padding=1,
            stride=1
        )
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        return out+x


####### Generator ##############
# Initial Block: kernels=9,filters=64,stride=1 
# n Residual Blocks: kernels=3,filters=64,stride=1: 
# 2 upsample blocks: kernels=3, filters=256,stride=1
# Final Conv Block: kernels=9,filters=3,stride=1 

class Generator(nn.Module):
    def __init__(self,in_channels=3,num_channels=64,num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels,kernel_size=9,padding=4,stride=1,use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels,kernel_size=3,padding=1,stride=1,use_act=False)
        self.upsample = nn.Sequential(*[UpsampleBlock(num_channels, 2) for _ in range(2)])
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9,stride=1,padding=4)

    def forward(self,x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = torch.add(self.convblock(x), initial)
        x = self.upsample(x)

        return torch.tanh(self.final(x))

###### Discriminator ########
# Initial Block: Conv Block: kernels=3,filters=64,stride=1, act=Leaky ReLU
# 7 Residual Blocks: kernels=3, filters=[64,128,128,256,256,512,512], strides=[2,1,2,1,2,1,2]  
# Final Block: Dense(1024) -> Leaky ReLU -> Dense(1) -> Sigmoid

class Discriminator(nn.Module):
    def __init__(self, in_channels=3,features=[64,64,128,128,256,256,512,512]):
        super().__init__()
        blocks = []

        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels, 
                    feature,
                    kernel_size=3,
                    stride=1 + idx%2,
                    padding=1,
                    disc=True,
                    use_act=True,
                    use_bn=False if idx==0 else True
                )
            )
            in_channels = feature

            self.blocks = nn.Sequential(*blocks)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((6,6)),
                nn.Flatten(),
                nn.Linear(512*6*6, 1024),
                NN.LeakyReLU(0.2,inplace=True),
                nn.Linear(1024, 1)
            )

            def forward(self,x):
                x = self.blocks(x)
                return self.classifier(x)


def test():
    low_res = 24

    with torch.cuda.amp.autocast():
        x = torch.randn((5,3,low_res,low_res))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)

if __name__ == "__main__":
    test()
