import torch
from torch import nn
import torchvision.models as models

### Encoder: Pretrained VGG19 (Down to ReLU 4-1)
### Decoder: Mirrors VGG, Pooling Layers Replaced with Nearest Upsampling

class VGGDecoder(nn.Module):
    def __init__(self, add_bn: bool = False): # , n_categories = 1000):
        super().__init__()
        self.add_bn = add_bn
        self.model = self.build_model()
        self.init_params()

    def build_model(self):
        ### VGG는 Maxpool로 Feature Map Size 축소 후 Channel 확장, Mirroring을 위해서는 Channel 확장 후 Feature Map Upscaling 수행 
        vgg19_reverse_config = [256, 'u', 256, 256, 256, 128, 'u', 128, 64, 'u', 64, 3]
        initial_dim = 512
        layers = []
        for config in vgg19_reverse_config:
            if isinstance(config, int):
                ### 일반 Padding 대신 Reflection Padding 사용
                layers.append(nn.Conv2d(initial_dim, config, kernel_size = (3, 3), stride = 1, padding = 1, padding_mode = "reflect"))
                if self.add_bn:
                    layers.append(nn.BatchNorm2d(config))
                layers.append(nn.ReLU())
                initial_dim = config

            elif config == 'u':
                layers.append(nn.Upsample(scale_factor = 2, mode = "nearest"))
        
        model = nn.Sequential(*layers)

        return model
    
    def init_params(self):
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.normal_(layer.bias, mean = 0, std = 0.01)

    def forward(self, x):
        assert x.shape[1:] == (512, 32, 32), x.shape
        x = self.model(x)
        return x
    


### Encoder는 Pretrained 모델을 사용하기 위해 Pytorch VGG로 불러옴
class VGGEncoder(nn.Module):
    def __init__(self, add_bn: bool = False):
        super().__init__()
        self.add_bn = add_bn
        self.model = self.load_freeze_vgg()
    
    ### Encoder의 출력 결과와 Decoder의 출력이 Encoder를 다시 통과한 결과를 이용해 Loss를 계산하기 때문에
    ### Encoder 모델은 업데이트가 이뤄지지 않아야 함
    def load_freeze_vgg(self):
        ### From Paper, Pretrained Encoder Uses Layers Down to ReLU 4-1
        if self.add_bn:
            model = models.vgg19_bn(weights = models.VGG19_BN_Weights.DEFAULT).features[:30]
        else:
            model = models.vgg19(weights = models.VGG19_Weights.DEFAULT).features[:21]
        
        model = self.convert_padding_mode(model)

        ### Freeze Parameters
        for param in model.parameters():
            param.requires_grad_(False)
        
        return model

    ### Encoder와 Decoder 모두 Reflection Padding을 사용
    def convert_padding_mode(self, model):
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                ### Reflection Padding으로 변경
                layer.padding_mode = "reflect"

        return model
    
    def forward(self, x):
        assert x.shape[1:] == (3, 256, 256)
        out = self.model(x)
        return out