"""Временный скрипт для обучения yolov7.

Целиком использует старый пайплайн с pandas, адаптером и pytorch accelerated.
"""

import sys
import random
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from pytorch_accelerated.callbacks import (
    ModelEmaCallback,
    ProgressBarCallback,
    SaveBestModelCallback,
    get_default_callbacks)
from pytorch_accelerated.schedulers import CosineLrScheduler

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.yolov7 import create_yolov7_model
from Yolov7.yolov7.dataset import (
    Yolov7Dataset,
    create_base_transforms,
    create_yolov7_transforms,
    yolov7_collate_fn)
from Yolov7.yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from Yolov7.yolov7.loss_factory import create_yolov7_loss
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from Yolov7.yolov7.utils import SaveBatchesCallback, Yolov7ModelEma
from Yolov7.blog_post.post_examples.minimal_finetune_cars import (
    CarsDatasetAdaptor)


def load_tanks_df(annotations_file_path: Path, images_path: Path):
    """Переделка функции из обучения машин.
    
    Всё то же самое, но "car" заменён на "Tank".
    """
    all_images = sorted(set([p.parts[-1] for p in images_path.iterdir()]))
    image_id_to_image = {i: im for i, im in enumerate(all_images)}
    image_to_image_id = {v: k for k, v, in image_id_to_image.items()}

    annotations_df = pd.read_csv(annotations_file_path)
    annotations_df.loc[:, "class_name"] = "Tank"
    annotations_df.loc[:, "has_annotation"] = True

    # add 100 empty images to the dataset
    empty_images = sorted(set(all_images) - set(annotations_df.image.unique()))
    non_annotated_df = pd.DataFrame(list(empty_images)[:100],
                                    columns=["image"])
    non_annotated_df.loc[:, "has_annotation"] = False
    non_annotated_df.loc[:, "class_name"] = "background"

    df = pd.concat((annotations_df, non_annotated_df))

    class_id_to_label = dict(
        enumerate(df.query("has_annotation == True").class_name.unique())
    )
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    df["image_id"] = df.image.map(image_to_image_id)
    df["class_id"] = df.class_name.map(class_label_to_id)

    file_names = tuple(df.image.unique())
    random.seed(42)
    validation_files = set(random.sample(file_names, int(len(df) * 0.2)))
    train_df = df[~df.image.isin(validation_files)]
    valid_df = df[df.image.isin(validation_files)]

    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id,
    }
    return train_df, valid_df, lookups


class TankDsetAdapter(CarsDatasetAdaptor):
    pass


def save_train_history(save_pth: Path, trainer: Yolov7Trainer):
    """Сохранить историю метрик обучения для графиков."""
    for metric_name in trainer.run_history.get_metric_names():
        metric_vals = trainer.run_history.get_metric_values(metric_name)
        metric_vals_str = '\n'.join(list(map(str, metric_vals)))
        file_name = save_pth / f'{metric_name}.txt'
        with open(file_name, 'w') as f:
            f.write(metric_vals_str)


def main(**kwargs):
    data_path = kwargs['data_path']
    work_dir: Path = kwargs['work_dir']
    image_size = kwargs['image_size']
    pretrained = kwargs['pretrained']
    num_epochs = kwargs['num_epochs']
    batch_size = kwargs['batch_size']

    # Prepare work directory
    ckpt_dir = work_dir / 'ckpts'
    batches_dir = work_dir / 'batches'
    metrics_dir = work_dir / 'metrics'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    # load data
    data_path = Path(data_path)
    images_path = data_path / 'images'
    annotations_file_path = data_path / (data_path.name + '.csv')
    train_df, valid_df, lookups = load_tanks_df(annotations_file_path,
                                                images_path)
    num_classes = 1

    # create datasets
    train_ds = TankDsetAdapter(
        images_path, train_df, transforms=create_base_transforms(image_size)
    )
    eval_ds = TankDsetAdapter(images_path, valid_df)

    mds = MosaicMixupDataset(
        train_ds,
        apply_mixup_probability=0.15,
        post_mosaic_transforms=create_post_mosaic_transform(
            output_height=image_size, output_width=image_size
        ),
    )
    if pretrained:
        # disable mosaic if finetuning
        mds.disable()

    train_yds = Yolov7Dataset(
        mds,
        create_yolov7_transforms(
            training=True, image_size=(image_size, image_size)),
    )
    eval_yds = Yolov7Dataset(
        eval_ds,
        create_yolov7_transforms(
            training=False, image_size=(image_size, image_size)),
    )

    # create model, loss function and optimizer
    model = create_yolov7_model(
        architecture="yolov7", num_classes=num_classes, pretrained=pretrained
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
            iou_threshold=0.2
        )
    )

    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.01,
            nms_threshold=0.3),
        callbacks=[
            calculate_map_callback,
            ModelEmaCallback(
                decay=0.9999,
                model_ema=Yolov7ModelEma,
                callbacks=[ProgressBarCallback, calculate_map_callback],
                save_path=str(ckpt_dir / 'ema_model.pt')),
            SaveBestModelCallback(
                save_path=str(ckpt_dir / 'best_model.pt'),
                watch_metric="map",
                greater_is_better=True),
            SaveBatchesCallback(
                str(batches_dir),
                num_images_per_batch=3),
            *get_default_callbacks(progress_bar=True)])

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
            k_decay=2),
        collate_fn=yolov7_collate_fn,
        gradient_accumulation_steps=num_accumulate_steps,
    )
    save_train_history(metrics_dir, trainer)


if __name__ == '__main__':
    data_path = Path(__file__).parents[3] / 'data' / 'tank_2set_rgb'
    work_dir = Path(__file__).parents[2] / 'work_dir' / 'debug_train'
    image_size = 640
    pretrained = True
    num_epochs = 100
    batch_size = 16
    main(data_path=data_path,
         work_dir=work_dir,
         image_size=image_size,
         pretrained=pretrained,
         num_epochs=num_epochs,
         batch_size=batch_size)
