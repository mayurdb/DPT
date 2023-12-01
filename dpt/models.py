import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), # batch x features // 2 x w x h
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True), # batch x features // 2 x w*2 x h*2
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1), # batch x 32 x w*2 x h*2
            nn.ReLU(True), # batch x 32 x w*2 x h*2
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), # batch x 1 x w*2 x h*2
            nn.ReLU(True) if non_negative else nn.Identity(), # batch x 1 x w*2 x h*2
            nn.Identity(), # batch x 1 x w*2 x h*2
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False), # batch x features x w x h
            nn.BatchNorm2d(features), # batch x features x w x h
            nn.ReLU(True), # batch x features x w x h
            nn.Dropout(0.1, False), # batch x features x w x h
            nn.Conv2d(features, num_classes, kernel_size=1), # batch x num_classes x w x h
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True), # batch x num_classes x w*2 x h*2
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)


class DPTBinaryClassification(DPT):
    def __init__(self, net_w, net_h, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        # Too heavy
        # head = nn.Sequential(
        #     # batch x features x height x width
        #     nn.Conv2d(features, features * 2, kernel_size=3, padding=1, bias=False), # batch x features*2 x height x width
        #     nn.BatchNorm2d(features * 2), # batch x features*2 x height x width
        #     nn.ReLU(True), # batch x features*2 x height x width
        #     nn.Dropout(0.1, False), # batch x features*2 x height x width
        #     nn.Flatten(1), # batch x features*2*height*width
        #     nn.ReLU(True), # batch x features*2*height*width
        #     nn.Linear(in_features=features*2*net_w*net_h, out_features=100), # batch x 100
        #     nn.ReLU(True), # batch x 100
        #     nn.Dropout(0.1, False), # batch x 100
        #     nn.Linear(in_features=100, out_features=2), # batch x 2
        #     nn.Softmax(dim=1) # batch x 2
        # )

        head = nn.Sequential(
            # batch x features x height x width
            nn.Flatten(1), # batch x features*height*width
            nn.Linear(in_features=features * net_w * net_h, out_features=2), # batch x 2
            nn.Softmax(dim=1) # batch x 2
        )

        # Sample output
        # Layer: Conv2d, Output Shape: torch.Size([2, 20, 4, 5])
        # Layer: BatchNorm2d, Output Shape: torch.Size([2, 20, 4, 5])
        # Layer: ReLU, Output Shape: torch.Size([2, 20, 4, 5])
        # Layer: Dropout, Output Shape: torch.Size([2, 20, 4, 5])
        # Layer: Flatten, Output Shape: torch.Size([2, 400])
        # Layer: ReLU, Output Shape: torch.Size([2, 400])
        # Layer: Linear, Output Shape: torch.Size([2, 100])
        # Layer: ReLU, Output Shape: torch.Size([2, 100])
        # Layer: Dropout, Output Shape: torch.Size([2, 100])
        # Layer: Linear, Output Shape: torch.Size([2, 2])
        # Layer: Softmax, Output Shape: torch.Size([2, 2])

        super().__init__(head, **kwargs)

        # self.auxlayer = nn.Sequential(
        #     nn.Flatten(1), # batch x features*height*width
        #     nn.Linear(in_features=features * net_w * net_h, out_features=2), # batch x 2
        #     nn.Softmax(dim=1) # batch x 2
        # )

        # if path is not None:
            # self.load(path)

    def forward(self, x):
        print("Called forward")
        output = super().forward(x).squeeze(dim=1)
        print("Got back")
        return torch.max(output, dim=-1)
