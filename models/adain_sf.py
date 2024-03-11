import torch
from torch import nn
from models.vgg import VGGEncoder, VGGDecoder

from utils.utils import get_mean_std

class AdaINTransfer(nn.Module):
    def __init__(self, add_bn: bool = False):
        super().__init__()
        self.encoder = VGGEncoder(add_bn = add_bn)
        self.decoder = VGGDecoder(add_bn = add_bn)
    
    def adain(self, content_features, style_features):
        ### Mu, Sigma of x
        content_mean, content_std = get_mean_std(content_features)

        ### Mu, Sigma of y
        style_mean, style_std = get_mean_std(style_features)

        ### AdaIN
        size = content_features.size()
        ### Normalize Content Image Features
        content_normalized = (content_features - content_mean.expand(size)) / content_std.expand(size)
        ### Apply Style Image Moments
        style_adapted = (content_normalized * style_std.expand(size)) + style_mean.expand(size)

        return style_adapted
    
    def forward(self, content, style):
        content = self.encoder(content)
        multiscale_style = self.forward_for_multiscale(style)

        ### style_adapted == notation t from paper for Content Feature Target
        style_adapted = self.adain(content, multiscale_style[-1])

        output = self.decoder(style_adapted)
        ### Pass Output Back to the Encoder
        multiscale_outputs = self.forward_for_multiscale(output)

        return multiscale_outputs, style_adapted, multiscale_style

    def forward_for_multiscale(self, x):
        ### 논문에서 언급한 세부 층의 출력을 활용해 Style Loss를 계산함
        ### To relu 1-1
        relu1_1 = self.encoder.model[:2](x)
        ### To relu 2-1
        relu2_1 = self.encoder.model[2:7](relu1_1)
        ### To relu 3-1
        relu3_1 = self.encoder.model[7:12](relu2_1)
        ### To relu 4-1
        relu4_1 = self.encoder.model[12:21](relu3_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1]
    
    def style_transfer(self, content, style, alpha = 1.0):
        ### Encode
        content = self.encoder(content)
        style = self.encoder(style)

        ### AdaIN
        style_adapted = self.adain(content, style)

        weighted = style_adapted * alpha + content * (1 - alpha)

        ### Decode
        output = self.decoder(style_adapted)

        return output