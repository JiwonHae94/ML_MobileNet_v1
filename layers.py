import numpy as np
import torch
from torch import nn

class Depthwise(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Depthwise, self).__init__()

