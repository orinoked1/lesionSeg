# import statements
import torch
import numpy as np
from model import LesionModel
from LesionDataModule import LesionDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    config = {
        'aug_degrees': (-10, 10),
        'shear_range': (0,0.3,0,0.3),
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.5,
        'root_dir': r'D:\downloads\neuralNetwork',
        'csv_file': 'labels_lesion.csv',
        'batch_size': 8,
        'encoder_name': 'resnet18',
        'encoder_weights': 'imagenet',
        'lr': 1e-3,
    }
    logger = TensorBoardLogger('tb_logs', name='my_model')

    model = LesionModel(config)
    data_module = LesionDataModule(config)

    trainer = pl.Trainer(gpus=1, max_epochs=300, progress_bar_refresh_rate=20, automatic_optimization=True,logger=logger)
    trainer.fit(model, data_module)


