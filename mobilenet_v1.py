import numpy as np
import torch
from torch import nn
from pytorch_model_summary import summary

class MobileNetV1(nn.Module):
    def __init__(self, n_classes, alpha = 1):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha

        def conv(ch_in, ch_out, stride):
            ch_out = int(ch_out * alpha)

            return nn.Sequential(
                nn.Conv2d(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=3,
                    stride = stride,
                    padding = 1,
                    bias = False
                ),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        def dw_seperable(ch_in, ch_out, stride):
            ch_in = int(alpha * ch_in)
            ch_out = int(alpha * ch_out)

            return nn.Sequential(
                #depthwise
                nn.Conv2d(
                    in_channels= ch_in,
                    out_channels= ch_in,
                    kernel_size=3,
                    stride = stride,
                    padding= 1,
                    groups = ch_in,
                    bias = False
                ),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True),

                #pointwise
                nn.Conv2d(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=1,
                    stride = 1,
                    padding = 0,
                    bias=False
                ),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv(  ch_in=   3, ch_out=  32, stride=2),
            dw_seperable(ch_in=  32, ch_out=  64, stride=1),
            dw_seperable(ch_in=  64, ch_out= 128, stride=2),
            dw_seperable(ch_in= 128, ch_out= 128, stride=1),
            dw_seperable(ch_in= 128, ch_out= 256, stride=2),
            dw_seperable(ch_in= 256, ch_out= 256, stride=1),
            dw_seperable(ch_in= 256, ch_out= 512, stride=2),
            dw_seperable(ch_in= 512, ch_out= 512, stride=1),
            dw_seperable(ch_in= 512, ch_out= 512, stride=1),
            dw_seperable(ch_in= 512, ch_out= 512, stride=1),
            dw_seperable(ch_in= 512, ch_out= 512, stride=1),
            dw_seperable(ch_in= 512, ch_out= 512, stride=1),
            dw_seperable(ch_in= 512, ch_out=1024, stride=2),
            dw_seperable(ch_in=1024, ch_out=1024, stride=1),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Linear(int(1024*alpha), n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024 * self.alpha))
        x = self.fc(x)
        return x

if __name__=='__main__':
    # model check
    model = MobileNetV1(n_classes=1000, alpha = 0.3)
    print(summary(model, torch.zeros((1, 3, 224, 224)), show_input=True))