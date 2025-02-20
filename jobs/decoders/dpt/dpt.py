import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_util.blocks import FeatureFusionBlock_custom, Interpolate

from modeling_finetune import VisionTransformer, _cfg, partial


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


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
        hooks=[2, 5, 8, 11], # ViT-S16_224
        features=256,
        channels_last=False,
        use_bn=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last
        # TODO: check
        groups = 1
        expand = False
        self.scratch = _make_scratch(
            [384, 384, 384, 384], features, groups=groups, expand=expand
        )  # ViT-S/16 - 84.6% Top1 (backbone)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, features):
        B, N, C = features[0].shape  # Batch size, tokens, channels
        B2 = B*8
        H = W = int((N/8) ** 0.5)  # Infer spatial dimensions from tokens

        print("\n========== FEAT shapes =======")
        print(features[0].shape)
        print(features[1].shape)
        print(features[2].shape)
        print(features[3].shape)

        # Reshape features into (B, C, H, W)
        layer_1 = features[0].transpose(1, 2).view(B2, C, H, W)
        layer_2 = features[1].transpose(1, 2).view(B2, C, H, W)
        layer_3 = features[2].transpose(1, 2).view(B2, C, H, W)
        layer_4 = features[3].transpose(1, 2).view(B2, C, H, W)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)  # error here
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(
        self, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

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
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
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


class DPTVideoMAE(VisionTransformer):
    def __init__(self, hook_layers=(2, 5, 8, 11), **kwargs):
        super().__init__(**kwargs)
        self.feature_hooks = []
        for hook_layer in hook_layers:
            hook = self.blocks[hook_layer].register_forward_hook(self.save_features)
            self.feature_hooks.append(hook)
        self.decoder = DPTDepthModel()

    def save_features(self, module, input, output):
        self.features.append(output)

    def forward(self, x):
        self.features = []
        x = self.forward_features(x)
        depth = self.decoder(self.features)
        risk = self.head(x)
        return depth, risk

