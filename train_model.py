# import statements
import torch
import numpy as np
from model import LesionModel
from LesionDataModule import LesionDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    config = {
        'AP': 1,                                                        # augmentation profile 1=none 2=mid 3=high
        'root_dir': r'D:\downloads\neuralNetwork',                      # data path
        'csv_file': 'labels_lesion.csv',                                # marking table
        'BS': 8,                                                        # batch size
        'A': "unet",                                                    # architecture (unet\deeplabv3++)
        'EN': "xception",                                               # encoder see segmentation_models_pytorch for options
        'encoder_weights': 'imagenet',                                  # pre trained weights see segmentation_models_pytorch
        'LR': 3e-4,                                                     # learn rate
        'WD': 1e-5,                                                     # weight decay
        'OC': 2,                                                        # output channels 1=no background channel, 2=with background channel
    }

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='ptl/val_iou',
        mode='max',  ## for val_loss this should be min
        dirpath='cpp_path/',
        filename='lesion-{epoch:02d}-{val_loss:.2f}'
    )

    logger = TensorBoardLogger('tb_logs', name='my_model')
    data_module = LesionDataModule(config)
    model = LesionModel(config)

    trainer = pl.Trainer(gpus=1, max_epochs=200, progress_bar_refresh_rate=50, automatic_optimization=True,
                         logger=logger, checkpoint_callback=[checkpoint_callback])

    trainer.fit(model, data_module)
