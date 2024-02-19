import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch import nn
from models.vgg import VGGEncoder, VGGDecoder

class AdaINTransfer(nn.Module):
    def __init__(self, add_bn: bool = False):
        super().__init__()
        self.encoder = VGGEncoder(add_bn = add_bn)
        self.decoder = VGGDecoder(add_bn = add_bn)
    
    def adain(self, content_features, style_features):
        ### Mu, Sigma of x
        content_mean = torch.mean(content_features)
        content_std = torch.std(content_features)

        ### Mu, Sigma of y
        style_mean = torch.mean(style_features)
        style_std = torch.std(style_features)

        ### AdaIN
        ### Normalize Content Image Features
        content_normalize = (content_features - content_mean) / content_std
        ### Apply Style Image Moments
        style_adapted = (content_normalize * style_std) + style_mean

        return style_adapted
    
    def forward(self, content, style):
        content = self.encoder(content)
        style = self.encoder(style)
        ### For Style Loss Calculation
        style_features = style.clone()

        style_adapted = self.adain(content, style)

        ### Notations from Paper, t for Content Loss
        t = style_adapted.clone()

        output = self.decoder(style_adapted)
        ### Pass Output Back to the Encoder
        multiscale_outputs = self.forward_for_multiscale(output)

        return multiscale_outputs, t, style_features

    def forward_for_multiscale(self, x):
        ### To relu 1-1
        x = self.encoder.model[:2](x)
        relu1_1 = x.clone()
        ### To relu 2-1
        x = self.encoder.model[2:7](x)
        relu2_1 = x.clone()
        ### To relu 3-1
        x = self.encoder.model[7:12](x)
        relu3_1 = x.clone()
        ### To relu 4-1
        relu4_1 = self.encoder.model[12:21](x)

        return [relu1_1, relu2_1, relu3_1, relu4_1]
    
    def style_transfer(self, content, style):
        ### Encode
        content = self.encoder(content)
        style = self.encoder(style)

        ### AdaIN
        style_adapted = self.adain(content, style)

        ### Decode
        output = self.decoder(style_adapted)

        return output
    
if __name__ == "__main__":
    content_tensor = torch.rand((1, 3, 512, 512))
    style_tensor = torch.randn((1, 3, 512, 512))

    model = AdaINTransfer(add_bn = False)

    multiscale_outputs, t, style_features = model(content_tensor, style_tensor)