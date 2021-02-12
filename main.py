# import statements
import torch
import numpy as np
from model import LesionModel
from LesionDataModule import LesionDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    data_cfg = {
        'aug_degrees': (-10, 10),
        'shear_range': (0,0.3,0,0.3),
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.5,
        'root_dir': r'C:\Users\gal\Documents\Gal\University\Shitut\Project\images',#r'D:\downloads\neuralNetwork',
        'csv_file': 'labels_lesion.csv',
        'batch_size': 8,
    }
    model_cfg = {
        'encoder_name': 'resnet18',
        'encoder_weights': 'imagenet',
        'lr': 1e-3,
    }

    #https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
    checkpoint_callback = ModelCheckpoint(
        filepath=r'C:\Users\gal\Documents\Gal\University\Shitut\Project\models\weights',
        verbose=True,
        monitor='val_loss',
        mode='min' ## for val_loss this should be min
    )
    logger = TensorBoardLogger('tb_logs', name='my_model')

    model = LesionModel(model_cfg,data_cfg)
    data_module = LesionDataModule(data_cfg)
    # Load_model = 1 ## TODO: Make this work
    # if Load_model:
    #     checkpoint = torch.load(r'C:\Users\gal\Documents\Gal\University\Shitut\Project\models\weights.ckpt',
    #                             map_location=lambda storage, loc: storage)
    #     exp = Experiment(model)
    #     exp.load_state_dict(checkpoint['state_dict'])
    #     trainer = pl.Trainer(experiment=exp)
    # else:
    trainer = pl.Trainer(gpus=1, max_epochs=300, progress_bar_refresh_rate=20, automatic_optimization=True,
                             logger=logger, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, data_module)


