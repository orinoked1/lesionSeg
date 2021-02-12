import tempfile
from ray import tune
import os
from model import LesionModel
from LesionDataModule import LesionDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def train_lesion(config, data_dir=None, num_epochs=10, num_gpus=1):
    model = LesionModel(config)
    data_module = LesionDataModule(config)
    logger = TensorBoardLogger('tb_logs', name='my_model')
    metrics = {"loss": "ptl/val_loss"}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        progress_bar_refresh_rate=1,
        logger=logger,
        automatic_optimization=True,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, data_module)


num_samples = 3
num_epochs = 3
gpus_per_trial = 1 # set this to higher if using GPU


config = {
    'aug_degrees': (-10, 10),
    'shear_range': (0, 0.3, 0, 0.3),
    'horizontal_flip_p': 0.5,
    'vertical_flip_p': 0.5,
    'root_dir': r'D:\downloads\neuralNetwork',
    'csv_file': 'labels_lesion.csv',
    'batch_size': 8,
    'encoder_name': tune.choice(['resnet18','resnet50']),
    'encoder_weights': 'imagenet',
    'lr': tune.loguniform(1e-4, 1e-2),
}

trainable = tune.with_parameters(
    train_lesion,
    data_dir='',
    num_epochs=num_epochs,
    num_gpus=gpus_per_trial
)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": gpus_per_trial
    },
    metric="loss",
    mode="min",
    config=config,
    num_samples=num_samples,
    name="tune_lesions")

print(analysis.best_config)
a=1
