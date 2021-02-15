import pytorch_lightning as pl
from LesionSegDataSet import LesionSegDataSet
from torchvision import transforms
from torch.utils import data


class LesionDataModule(pl.LightningDataModule):

    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        if self.data_cfg['AP']==1:
            self.AD=(0,0)
            self.HF = 0
            self.VF = 0
        elif self.data_cfg['AP']==2:
            self.AD=(-10,10)
            self.HF = 0.3
            self.VF = 0.3
        elif self.data_cfg['AP'] == 3:
            self.AD=(-20,20)
            self.HF = 0.5
            self.VF = 0.5

    def setup(self, stage=None):
        # see more aug in https://pytorch.org/docs/stable/torchvision/transforms.html
        aug_trans = transforms.Compose([
            transforms.RandomRotation(degrees=self.AD),
            transforms.RandomHorizontalFlip(p=self.HF),
            transforms.RandomVerticalFlip(p=self.VF),

        ])
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            aug_trans,
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.lesion_dataset_train = LesionSegDataSet(self.data_cfg['root_dir'], self.data_cfg['csv_file'], data_set='train',
                                                     transform=self.train_transform,out_chan=self.data_cfg['OC'])
        self.lesion_dataset_validation = LesionSegDataSet(self.data_cfg['root_dir'], self.data_cfg['csv_file'],
                                                          data_set='validation', transform=self.val_transform,out_chan=self.data_cfg['OC'])
        self.lesion_dataset_test = LesionSegDataSet(self.data_cfg['root_dir'], self.data_cfg['csv_file'], data_set='test',
                                                    transform=self.val_transform,out_chan=self.data_cfg['OC'] )

    def train_dataloader(self):
        return data.DataLoader(self.lesion_dataset_train, batch_size=self.data_cfg['BS'])

    def val_dataloader(self):
        return data.DataLoader(self.lesion_dataset_validation, batch_size=self.data_cfg['BS'])

    def test_dataloader(self):
        return data.DataLoader(self.lesion_dataset_test, batch_size=1)
