import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from dice_loss import DiceLoss
import numpy as np
from ray.tune.integration.pytorch_lightning import TuneReportCallback


class IOU(nn.Module):
    def init(self):
        super(IOU, self).init()

    def forward(self, pred, target):
        iousum = 0
        for i in range(target.shape[0]):
            if target.shape[1]==2:
                target_arr = target[i, :, :, :].clone().argmax(0)==0
                predicted_arr = pred[i, :, :, :].clone().argmax(0)==0
            else:
                predicted_arr = torch.sigmoid(pred[i, :, :, :].clone())>0.5
                target_arr = target[i, :, :, :].clone()>0
            intersection = torch.logical_and(target_arr, predicted_arr).sum()
            union = torch.logical_or(target_arr, predicted_arr).sum()
            iou_score = intersection / union if union else 0
            iousum += iou_score

        miou = iousum / target.shape[0]
        return miou


class LesionModel(pl.LightningModule):
    def __init__(self, config):
        super(LesionModel, self).__init__()
        if config["A"]=="unet":
            self.model = smp.Unet(
                encoder_name=config["EN"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=config["encoder_weights"],
                # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config["OC"],  # model output channels (number of classes in your dataset)
            )
        elif config["A"]=="DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=config["EN"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=config["encoder_weights"],
                # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config["OC"],  # model output channels (number of classes in your dataset)
            )
        self.model_cfg = config
        self.loss_function = DiceLoss()
        self.save_hyperparameters()
        self.iou_function = IOU()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)
        iou_val = self.iou_function(y_pred, y)

        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('IOU_loss', iou_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)
        iou_val = self.iou_function(y_pred, y)
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('IOU_loss', iou_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss_val,"val_iou": iou_val}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_iou = torch.stack(
            [x["val_iou"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_iou", avg_iou)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.model_cfg["LR"],weight_decay=self.model_cfg["WD"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, verbose=True)
        mon = 'val_loss'
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': mon
        }
