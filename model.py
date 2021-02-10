import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from dice_loss import DiceLoss


class diceLoss(nn.Module):
    def init(self):
        super(diceLoss, self).init()

    def forward(self, pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class LesionModel(pl.LightningModule):
    def __init__(self, model_cfg,data_cfg):
        super(LesionModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=model_cfg["encoder_name"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=model_cfg["encoder_weights"],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
        self.model_cfg = model_cfg
        self.loss_function = DiceLoss()
        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)

        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_val

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.model_cfg["lr"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, verbose=True)
        mon = 'val_loss'
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': mon
        }
