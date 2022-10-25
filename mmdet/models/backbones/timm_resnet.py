from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
import timm

@BACKBONES.register_module()
class TIMM_ResNet(BaseModule):
    """     
    (1, 64, 8, 8)
    (1, 128, 4, 4)
    (1, 256, 2, 2)
    (1, 512, 1, 1) 
    
    """
    def __init__(self,
                 out_indices=(1, 2, 3, 4),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.network = timm.create_model('resnet18d', 
                                        features_only=True, 
                                        out_indices=out_indices, 
                                        pretrained=True)

    def forward(self, x):
        outs = self.network(x)
        return outs
