from typing import Optional, Callable, List, Dict, Tuple, Union
from collections import OrderedDict
import warnings
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch import Tensor, FloatTensor
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNet
from torchvision.ops.feature_pyramid_network import (
    ExtraFPNBlock, LastLevelMaxPool)
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

sys.path.append(str(Path(__file__).parents[1]))
from region_localizer.modified_attention import ModifiedAttention


# Source - torchvision.models.detection.backbone_utils 115
def create_resnet_fpn_extractor(
    backbone: resnet.ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    """Prepare FPN backbone for retina.

    Parameters
    ----------
    backbone : resnet.ResNet
        Backbone from ResNet family to use in FPN.
    trainable_layers : int
        Number of trainable ResNet layers. Must be from 1 to 4.
        The countdown comes from the deeper layers.
    returned_layers : Optional[List[int]], optional
        Layers that will have FPN skip connection.
        The countdown comes from the upper layers.
    extra_blocks : Optional[ExtraFPNBlock], optional
        Extra layers that will be added to ResNet.
    norm_layer : Optional[Callable[..., nn.Module]], optional
        Normalization layer class to add to FPN conv blocks.

    Returns
    -------
    BackboneWithFPN
        FPN based on ResNet.
    """
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(
            'Trainable layers should be in the range [0,5], '
            f'got {trainable_layers}')
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(
            'Each returned layer should be in the range [1,4]. '
            f'Got {returned_layers}')
    return_layers = {
        f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels,
        extra_blocks=extra_blocks, norm_layer=norm_layer)


# Source - torchvision.models.detection.retinanet 50
def create_default_anchorgen() -> AnchorGenerator:
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                         for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator


class ModifiedRetinaHead(RetinaNetHead):
    def __init__(
        self, in_channels: int, num_anchors: int, num_attn_heads: int,
        attn_drop: float = 0.5,
        norm_layer: Optional[Callable[..., Module]] = None,
        num_classes: int = 2
    ):
        super().__init__(in_channels, num_anchors, num_classes, norm_layer)
        # Create attention layer for every level of fpn
        self.attentions = nn.ModuleList([
            ModifiedAttention(embed_dim=in_channels,
                              num_heads=num_attn_heads,
                              dropout=attn_drop, batch_first=True)
            for _ in range(5)])

    def forward(
        self, map_logits: List[Tensor], piece_logits: List[Tensor]
    ) -> Dict[str, Tensor]:
        """Modified forward pass of RetinaNetHead.

        Instead of making predict based on logits from one image it gets two
        different image's logits and combine them by Attention layer.
        TODO дописать подробнее, когда будет готово

        Parameters
        ----------
        map_logits : List[Tensor]
            Logits from map image. List of length equal to the number of
            backbone output feature maps. Each element has shape
            `(b, backbone_out_ch, h_map, w_map)`.
        piece_logits : List[Tensor]
            Logits from local piece image.

        Returns
        -------
        Dict[str, Tensor]
            Predicted bboxes of most likely regions.
        """
        piece_logits = piece_logits.permute(0, 2, 1)  # b, 1, c
        weighted_maps = []
        for i, map_logit in enumerate(map_logits):  # b, c, h, w
            b, embed, h, w = map_logit.shape
            map_logit = map_logit.reshape((b, embed, -1))  # b, c, h * w
            map_logit = map_logit.permute(0, 2, 1)  # b, h * w, c
            weighted_map, _ = self.attentions[i](
                piece_logits, map_logit, map_logit)  # b, h * w, c
            weighted_map = weighted_map.permute(0, 2, 1)  # b, c, h*w
            weighted_map = weighted_map.reshape((b, embed, h, w))  # b, c, h, w
            weighted_maps.append(weighted_map)
        return super().forward(weighted_maps)
    

class ModifiedRetina(RetinaNet):
    # Source - torchvision.models.detection.retinanet 569
    def forward(
        self, map_images: List[FloatTensor], pieces_logits: FloatTensor,
        map_targets: List[Dict[str, Tensor]] = None
    ) -> Union[List[Tensor], Dict[str, Tensor]]:
        """Forward pass of `ModifiedRetina`.

        Parameters
        ----------
        map_images : List[FloatTensor]
            Images of region map. They can have different shapes.
        pieces_logits : FloatTensor
            Processed piece logits with shape `(b, 1, 256)`.
        map_targets : List[Dict[str, Tensor]], optional
            Targets dict with labels and bboxes. By default is `None`.

        Returns
        -------
        Union[List[Tensor], Dict[str, Tensor]]
            TODO
        """
        if self.training:
            if map_targets is None:
                raise ValueError(
                    'Targets should not be none when in training mode')
            else:
                for target in map_targets:
                    boxes = target["boxes"]
                    if not isinstance(boxes, Tensor):
                        raise TypeError(
                            'Expected target boxes to be of type Tensor.')
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            'Expected target boxes to be a tensor of shape '
                            '[N, 4].')

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in map_images:
            val = img.shape[-2:]
            if len(val) != 2:
                raise ValueError(
                    'Expecting the last two dimensions of the Tensor to be H '
                    f'and W instead got {img.shape[-2:]}')
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        map_images, map_targets = self.transform(map_images, map_targets)

        # Check for degenerate boxes
        if map_targets is not None:
            for target_idx, target in enumerate(map_targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        'All bounding boxes should have positive height and '
                        f'width. Found invalid box {degen_bb} for target at '
                        f'index {target_idx}.')

        # get the features from the backbone
        features = self.backbone(map_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        #################################################
        # Changed
        head_outputs = self.head(features, pieces_logits)
        #################################################

        # create the set of anchors
        anchors = self.anchor_generator(map_images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if map_targets is None:
                raise ValueError(
                    'Targets should not be none when in training mode')
            else:
                # compute the losses
                losses = self.compute_loss(map_targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(
                    num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level))
                             for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(
                split_head_outputs, split_anchors, map_images.image_sizes)
            detections = self.transform.postprocess(
                detections, map_images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    'RetinaNet always returns a (Losses, Detections) '
                    'tuple in scripting')
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

    @staticmethod
    def create_modified_retina(
        min_size: int, max_size: int, pretrained: bool, num_classes: int = 2
    ) -> 'ModifiedRetina':
        if pretrained:
            trainable_layers = 3
            weights = resnet.ResNet50_Weights.DEFAULT
        else:
            trainable_layers = 5
            weights = None
        backbone = resnet.resnet50(weights=weights)
        backbone = create_resnet_fpn_extractor(
            backbone, trainable_layers=trainable_layers,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256))
        
        anchor_gen = create_default_anchorgen()
        
        custom_head = ModifiedRetinaHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_gen.num_anchors_per_location()[0],
            num_attn_heads=8, num_classes=num_classes)

        retina = ModifiedRetina(
            min_size=min_size, max_size=max_size,
            backbone=backbone, num_classes=num_classes,
            anchor_generator=anchor_gen, head=custom_head)
        
        return retina


if __name__ == '__main__':
    retina = ModifiedRetina.create_modified_retina(2464, 2464, True)
    print(retina)
