import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from dice_loss import DiceLoss
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd


class IOU(nn.Module):
    def init(self):
        super(IOU, self).init()

    def forward(self, pred, target):
        iousum = 0
        for i in range(target.shape[0]):
            if target.shape[1] == 2:
                target_arr = target[i, :, :, :].clone().argmax(0) == 0
                predicted_arr = pred[i, :, :, :].clone().argmax(0) == 0
            else:
                predicted_arr = torch.sigmoid(pred[i, :, :, :].clone()) > 0.5
                target_arr = target[i, :, :, :].clone() > 0
            intersection = torch.logical_and(target_arr, predicted_arr).sum()
            union = torch.logical_or(target_arr, predicted_arr).sum()
            iou_score = intersection / union if union else 0
            iousum += iou_score

        miou = iousum / target.shape[0]
        return miou


class LesionModel(pl.LightningModule):
    def __init__(self, config):
        super(LesionModel, self).__init__()
        if config["A"] == "unet":
            self.model = smp.Unet(
                encoder_name=config["EN"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=config["encoder_weights"],
                # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=config["OC"],  # model output channels (number of classes in your dataset)
            )
        elif config["A"] == "DeepLabV3Plus":
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
        self.checkpoint_path = ""

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)
        iou_val = self.iou_function(y_pred, y)

        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_iou', iou_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)
        iou_val = self.iou_function(y_pred, y)
        self.log('val_loss', loss_val, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_iou', iou_val, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return {"val_loss": loss_val, "val_iou": iou_val}

    def test_step(self, batch, batch_idx):
        x, y, image_name = batch['image'], batch['mask'], batch['image_name']
        y_pred = self(x)
        loss_val = self.loss_function(y_pred, y)
        iou_val = self.iou_function(y_pred, y)
        in_image = x[0, :, :, :].detach().cpu().numpy()
        in_image = np.moveaxis(in_image, 0, -1)
        if self.model_cfg["OC"] == 1:
            gt_mask = (y[0, 0, :, :] > 0.5).detach().cpu().numpy()
            out_mask = (y_pred[0, 0, :, :] > 0.5).detach().cpu().numpy()
        else:
            gt_mask = (y[0, 0, :, :] > y[0, 1, :, :]).detach().cpu().numpy()
            out_mask = (y_pred[0, 0, :, :] > y_pred[0, 1, :, :]).detach().cpu().numpy()
        out_img = color.label2rgb(out_mask, in_image, bg_label=0)
        out_img[out_mask == False] = in_image[out_mask == False]
        gt_img = color.label2rgb(gt_mask, in_image, bg_label=0)
        gt_img[gt_mask == False] = in_image[gt_mask == False]
        fig = plt.figure(figsize=(10, 3))
        plt.subplot(131)
        plt.imshow(in_image)
        plt.axis('off')
        plt.title("Original image")
        plt.subplot(132)
        plt.imshow(gt_img)
        plt.axis('off')
        plt.title("GT image")
        plt.subplot(133)
        plt.imshow(out_img)
        plt.title("Result image")
        plt.suptitle("dice score:{:0.3f}, IoU: {:0.3f}".format(loss_val.cpu().numpy(), iou_val.cpu().numpy()))
        plt.axis('off')
        plt.tight_layout()
        Path(os.path.join(self.checkpoint_path, 'test_images')).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(self.checkpoint_path, 'test_images', image_name[0] + ".png"), dpi=100)
        plt.close(fig)

        return {"image_name": image_name[0], "dice_loss": loss_val.cpu().numpy(), "iou": iou_val.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_iou = torch.stack(
            [x["val_iou"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss, prog_bar=True)
        self.log("ptl/val_iou", avg_iou, prog_bar=True)
        self.log("hp_metric", avg_iou, prog_bar=False)

    def test_epoch_end(self, outputs):
        avg_loss = np.stack(
            [x["dice_loss"] for x in outputs]).mean()
        avg_iou = np.stack(
            [x["iou"] for x in outputs]).mean()
        df = pd.DataFrame(outputs)
        df.to_csv(os.path.join(self.checkpoint_path, "test_res.csv"))
        return {"iou": avg_iou, "dice_loss": avg_loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.model_cfg["LR"], weight_decay=self.model_cfg["WD"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, verbose=True)
        mon = 'val_loss'
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': mon
        }
