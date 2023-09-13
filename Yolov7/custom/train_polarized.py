from functools import partial
from pathlib import Path

import torch
from pytorch_accelerated.callbacks import (
    ModelEmaCallback,
    ProgressBarCallback,
    SaveBestModelCallback,
    get_default_callbacks,
)
from pytorch_accelerated.schedulers import CosineLrScheduler

import sys
sys.path.append(str(Path(__file__).parents[1]))

from yolov7 import create_yolov7_model
from yolov7.dataset import (
    Yolov7Dataset,
    create_base_transforms,
    create_yolov7_transforms,
    yolov7_collate_fn,
)
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from yolov7.loss_factory import create_yolov7_loss
from yolov7.mosaic import MosaicMixupDataset, create_post_mosaic_transform
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from yolov7.utils import SaveBatchesCallback, Yolov7ModelEma
from polarized_dataset import PolarizedDatasetAdaptor, load_df


DATA_PATH = Path(__file__).absolute().parents[2] / 'data/polarized_dataset'


def main(
    data_path: str = DATA_PATH,
    image_size: int = 640,
    pretrained: bool = False,
    num_epochs: int = 15,
    batch_size: int = 16,
):

    # load data
    data_path = Path(data_path)
    images_path = data_path / "Images"
    annotations_file_path = data_path / "annotations.csv"
    train_df, valid_df, lookups = load_df(annotations_file_path, images_path)
    num_classes = 1

    # create datasets
    train_ds = PolarizedDatasetAdaptor(
        images_path, train_df, transforms=create_base_transforms(image_size)
    )
    eval_ds = PolarizedDatasetAdaptor(images_path, valid_df)

    mds = MosaicMixupDataset(
        train_ds,
        apply_mixup_probability=0.15,
        post_mosaic_transforms=create_post_mosaic_transform(
            output_height=image_size, output_width=image_size
        ),
        pad_colour=(114,) * CHANNELS
    )

    if pretrained:
        # disable mosaic if finetuning
        mds.disable()

    train_yds = Yolov7Dataset(
        mds,
        create_yolov7_transforms(
            training=True, image_size=(image_size, image_size)))

    eval_yds = Yolov7Dataset(
        eval_ds,
        create_yolov7_transforms(
            training=False, image_size=(image_size, image_size)))

    # create model, loss function and optimizer
    model = create_yolov7_model(
        architecture="yolov7", num_classes=num_classes,
        pretrained=pretrained, num_channels=CHANNELS
    )
    param_groups = model.get_parameter_groups()

    loss_func = create_yolov7_loss(model, image_size=image_size)

    optimizer = torch.optim.SGD(
        param_groups["other_params"], lr=0.01, momentum=0.937, nesterov=True
    )

    # create evaluation callback and trainer
    calculate_map_callback = (
        CalculateMeanAveragePrecisionCallback.create_from_targets_df(
            targets_df=valid_df.query("has_annotation == True")[
                ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
            ],
            image_ids=set(valid_df.image_id.unique()),
            iou_threshold=0.7,
        )
    )

    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        filter_eval_predictions_fn=partial(filter_eval_predictions,
                                           confidence_threshold=0.01,
                                           nms_threshold=0.7),
        callbacks=[
            calculate_map_callback,
            ModelEmaCallback(
                decay=0.9999,
                model_ema=Yolov7ModelEma,
                callbacks=[ProgressBarCallback, calculate_map_callback],
            ),
            SaveBestModelCallback(watch_metric="map", greater_is_better=True),
            SaveBatchesCallback("./batches", num_images_per_batch=3),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    # calculate scaled weight decay and gradient accumulation steps
    total_batch_size = (
        batch_size * trainer._accelerator.num_processes
    )  # batch size across all processes

    nominal_batch_size = 64
    num_accumulate_steps = max(round(nominal_batch_size / total_batch_size), 1)
    base_weight_decay = 0.0005
    scaled_weight_decay = (
        base_weight_decay * total_batch_size * num_accumulate_steps /
        nominal_batch_size
    )

    optimizer.add_param_group(
        {"params": param_groups["conv_weights"],
         "weight_decay": scaled_weight_decay}
    )

    # run training
    trainer.train(
        num_epochs=num_epochs,
        train_dataset=train_yds,
        eval_dataset=eval_yds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5,
            num_cooldown_epochs=5,
            k_decay=2,
        ),
        collate_fn=yolov7_collate_fn,
        gradient_accumulation_steps=num_accumulate_steps,
    )


if __name__ == "__main__":
    CHANNELS = 4
    main()
