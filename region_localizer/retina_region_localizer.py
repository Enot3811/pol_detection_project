from typing import Tuple, List, Dict
from collections import OrderedDict
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torchvision.models import resnet
from torchvision.models.detection.transform import GeneralizedRCNNTransform

sys.path.append(str(Path(__file__).parents[1]))
from region_localizer.modified_retina import ModifiedRetina


class RetinaRegionLocalizer(nn.Module):
    def __init__(
        self, img_min_size: int, img_max_size: int, pretrained: bool,
        image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        num_classes: int = 2
    ) -> None:
        super().__init__()
        self.retina = ModifiedRetina.create_modified_retina(
            min_size=img_min_size, max_size=img_max_size,
            pretrained=pretrained, num_classes=num_classes)
        self.piece_extractor = self.get_piece_extractor(pretrained=pretrained)
        
        # Для карты преобразования встроены в retina
        # Те же самые преобразования здесь для куска
        self.piece_transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, image_mean, image_std)

    def forward(
        self, map_batch: List[FloatTensor], piece_batch: List[FloatTensor],
        map_targets: List[Dict[str, Tensor]] = None
    ):
        piece_batch, _ = self.piece_transform(piece_batch)
        piece_logits = self.piece_extractor(piece_batch.tensors)  # b,256,1,1
        piece_logits = torch.squeeze(piece_logits, -1)  # b, 256, 1
        output = self.retina(map_batch, piece_logits, map_targets)
        return output
        
    def get_piece_extractor(self, pretrained: bool) -> resnet.ResNet:
        if pretrained:
            # Freeze first layer and initial conv like in retina
            weights = resnet.ResNet50_Weights.DEFAULT
            freezed_layers = ["layer1", "conv1", 'bn1']
        else:
            weights = None
            freezed_layers = []
        extractor = resnet.resnet50(weights=weights)
        layers = list(extractor.named_children())[:8]
        layers.append((
            'inner_conv',
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)))
        
        layers.append(('extra_p6', nn.Conv2d(256, 256, 3, 2, 1)))
        layers.append(('extra_relu', nn.ReLU()))
        layers.append(('extra_p7', nn.Conv2d(256, 256, 3, 2, 1)))
        layers.append(('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1))))
        extractor = nn.Sequential(OrderedDict(layers))

        for name, parameter in extractor.named_parameters():
            if any([name.startswith(layer) for layer in freezed_layers]):
                parameter.requires_grad_(False)
        return extractor


if __name__ == '__main__':
    img_size = 1024
    b = 2
    pretrained = True
    model = RetinaRegionLocalizer(
        img_min_size=img_size, img_max_size=img_size, pretrained=pretrained)

    dummy_map = torch.rand((b, 3, img_size, img_size))
    dummy_map = torch.unbind(dummy_map)
    dummy_piece = torch.rand((b, 3, img_size, img_size))
    dummy_piece = torch.unbind(dummy_piece)
    dummy_targets = [
        {'boxes': torch.tensor([[50, 500, 100, 1000],
                                [200, 400, 500, 600]]),
         'labels': torch.tensor([1, 1], dtype=torch.int64)}
        for _ in range(b)]

    print(model)
    model.train()
    loss, predicts = model(dummy_map, dummy_piece, dummy_targets)
    print(loss)
    model.eval()
    loss, predict = model(dummy_map, dummy_piece, dummy_targets)
    print(predict)
