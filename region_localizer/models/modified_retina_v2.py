from typing import Optional, Callable, List, Dict, Tuple
from collections import OrderedDict
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torchvision.models.resnet import ResNet50_Weights, resnet50, ResNet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights, RetinaNetHead, RetinaNet)
from torchvision.ops.feature_pyramid_network import (
    LastLevelP6P7, ExtraFPNBlock, LastLevelMaxPool)

sys.path.append(str(Path(__file__).parents[1]))
from region_localizer.models.modified_retina import create_default_anchorgen
from utils.torch_utils.torch_functions import make_compatible_state_dict


# Source - torchvision.models.detection.backbone_utils 115
def create_resnet_fpn_extractor(
    backbone: ResNet,
    trainable_layers: int,
    first_conv_out_ch: int,
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

    in_channels_stage2 = first_conv_out_ch // 8
    in_channels_list = [
        in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels,
        extra_blocks=extra_blocks, norm_layer=norm_layer)


class ModifiedRetinaV2(RetinaNet):
    # Source - torchvision.models.detection.retinanet 569
    def forward(
        self, map_images: List[FloatTensor],
        map_targets: List[Dict[str, Tensor]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """Forward pass of `ModifiedRetinaV2`.

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
        Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
            Dict with losses and predictions with boxes, scores and labels.
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
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(map_images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        #######################################################################
        # Changed
        if map_targets is not None:
            # compute the losses
            losses = self.compute_loss(map_targets, head_outputs, anchors)

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
        
        return losses, detections
        #######################################################################
    
    @staticmethod
    def create_modified_retina(
        min_size: int, max_size: int, pretrained: bool, num_classes: int
    ) -> 'ModifiedRetinaV2':
        if pretrained:
            trainable_layers = 3
            weights_backbone = ResNet50_Weights.DEFAULT
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        else:
            trainable_layers = 5
            weights_backbone = None
            weights = None

        # source torchvision.model.detection.retinanet.py 830
        weights = RetinaNet_ResNet50_FPN_V2_Weights.verify(weights)
        weights_backbone = ResNet50_Weights.verify(weights_backbone)

        #######################################################################
        # Changes
        backbone = resnet50(weights=weights_backbone)
        first_conv_out_ch = backbone.inplanes
        layers = list(backbone.named_children())[1:]  # cut out conv(3, 64)
        # and add new conv(6, 64)
        layers.insert(0, (
            'conv1', nn.Conv2d(6, 64, kernel_size=7, stride=1, bias=False)))
        backbone = nn.Sequential(OrderedDict(layers))
        backbone = create_resnet_fpn_extractor(
            backbone, trainable_layers=trainable_layers,
            returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256),
            first_conv_out_ch=first_conv_out_ch)
        
        anchor_gen = create_default_anchorgen()
        #######################################################################
        head = RetinaNetHead(
            backbone.out_channels,
            anchor_gen.num_anchors_per_location()[0],
            num_classes,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        head.regression_head._loss_type = "giou"
        model = ModifiedRetinaV2(
            backbone, num_classes, anchor_generator=anchor_gen, head=head,
            min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406] * 2,
            image_std=[0.229, 0.224, 0.225] * 2
        )  # *2 to normalize 2 stacked images

        if weights is not None:
            state_dict = weights.get_state_dict(progress=True)
            state_dict = make_compatible_state_dict(model, state_dict)
            model.load_state_dict(state_dict, strict=False)
        return model


if __name__ == '__main__':
    min_size = 1024
    max_size = 1024
    b = 2
    device = torch.device('cuda')

    model = ModifiedRetinaV2.create_modified_retina(
        min_size=min_size, max_size=max_size, pretrained=True, num_classes=2)
    
    dummy_map = torch.rand((b, 6, max_size, max_size))
    dummy_map = list(torch.unbind(dummy_map))
    dummy_targets = [
        {'boxes': torch.tensor([[50, 500, 100, 1000],
                                [500, 400, 1000, 600]]),
         'labels': torch.tensor([1, 1], dtype=torch.int64)}
        for _ in range(b)]
    
    model.to(device=device)
    for i in range(len(dummy_map)):
        dummy_map[i] = dummy_map[i].to(device=device)
        dummy_targets[i]['boxes'] = dummy_targets[i]['boxes'].to(device=device)
        dummy_targets[i]['labels'] = (dummy_targets[i]['labels']
                                      .to(device=device))

    print(model)
    model.train()
    losses, predicts = model(dummy_map, dummy_targets)
    print(losses)
    model.eval()
    losses, predicts = model(dummy_map, dummy_targets)
    print(predicts)
