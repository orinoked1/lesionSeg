import tempfile
from ray import tune
import os
from model import LesionModel
from LesionDataModule import LesionDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime


def train_lesion(config, data_dir=None, num_epochs=10, num_gpus=1):
    model = LesionModel(config)
    data_module = LesionDataModule(config)
    logger = TensorBoardLogger('tb_logs', name='my_model')
    metrics = {"iou": "ptl/val_iou"}
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='ptl/val_iou',
        mode='max',  ## for val_loss this should be min
        dirpath='check_point_path/',
        filename='lesion-{epoch:02d}-{val_iou:.2f}'
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=50,
        logger=logger,
        automatic_optimization=True,
        callbacks=[TuneReportCallback(metrics, on="validation_end"),
                   checkpoint_callback])
    trainer.fit(model, data_module)


num_samples = 30
num_epochs = 100
gpus_per_trial = 1 # set this to higher if using GPU


config = {
    'AD': tune.choice([(0,0),(-10,10)]),
    'HF': tune.choice([0,0.3]),
    'VF': tune.choice([0,0.3]),
    'root_dir': r'D:\downloads\neuralNetwork',
    'csv_file': 'labels_lesion.csv',
    'BS':  tune.choice([8,16,32]),
    'A': "unet",
    'EN': tune.choice(["xception","densenet169"]),
    'encoder_weights': 'imagenet',
    'LR': tune.loguniform(5e-4, 5e-3),
    'WD': 1e-5,
    'OC': tune.choice([1,2]),
}

trainable = tune.with_parameters(
    train_lesion,
    data_dir='',
    num_epochs=num_epochs,
    num_gpus=gpus_per_trial
)

def trial_str_creator(trial):
    return "{}".format(trial.trial_id)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 4,
        "gpu": gpus_per_trial
    },
    metric="iou",
    mode="max",
    config=config,
    num_samples=num_samples,
    name="tune_lesions",
    local_dir=r"D:\ray",
    trial_name_creator=trial_str_creator,)

print(analysis.best_config)
a=1
