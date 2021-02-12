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

PATH_TO_IMAGES = r'C:\Users\gal\Documents\Gal\University\Shitut\Project\images'#'D:\downloads\neuralNetwork',
PATH_TO_CHECKPOINT = r'C:\Users\gal\Documents\Gal\University\Shitut\Project\models\weights'#'cpp_path/',

# Start with main code
if __name__ == '__main__':
    config = {
        'aug_degrees': (-10, 10),
        'shear_range': (0,0.3,0,0.3),
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.5,
        'root_dir': PATH_TO_IMAGES,
        'csv_file': 'labels_lesion.csv',
        'batch_size': 8,
        'encoder_name': 'resnet18',
        'encoder_weights': 'imagenet',
        'lr': 1e-3,
        'weight_decay':1e-4,
    }

    #https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        mode='min', ## for val_loss this should be min
        dirpath=PATH_TO_CHECKPOINT,
        filename='lesion-{epoch:02d}-{val_loss:.2f}'
    )

    logger = TensorBoardLogger('tb_logs', name='my_model')
    data_module = LesionDataModule(config)
    model = LesionModel(config)

    Load_model = 1
    if Load_model:
        checkpoint = torch.load(PATH_TO_CHECKPOINT + '.ckpt')
        model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(gpus=1, max_epochs=300, progress_bar_refresh_rate=20, automatic_optimization=True,
                         logger=logger, checkpoint_callback=[checkpoint_callback])

    trainer.fit(model, data_module)


