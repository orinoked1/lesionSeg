# import statements
import os
from model import LesionModel
from LesionDataModule import LesionDataModule
import pytorch_lightning as pl


checkpoint_path = r"D:\ray\tune_lesions6\b9aba_00001_1_AD=(-10, 10),BS=16,EN=xception,HF=0,LR=0.0030492,OC=1,VF=0_2021-02-13_15-16-55\check_point_path\lesion-epoch=20-val_iou=0.60.ckpt"


model = LesionModel.load_from_checkpoint(checkpoint_path)
model.checkpoint_path = os.path.dirname(os.path.abspath(checkpoint_path))
data_module = LesionDataModule(model.hparams['config'])

trainer = pl.Trainer()
trainer.test(model, datamodule=data_module)