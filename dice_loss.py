import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # https: // www.mathworks.com / help / vision / ref / nnet.cnn.layer.dicepixelclassificationlayer.html
        inputs = torch.sigmoid(inputs)
        xy_dim = [2,3]
        chan_dim=1
        obs_dim=0
        batch_size = inputs.shape[obs_dim]
        w = 1/(targets.sum(dim=xy_dim)**2+1e-10)
        # flatten label and prediction tensors
        intersection = (inputs*targets).sum(dim=xy_dim)
        union = (inputs**2 + targets**2).sum(dim=xy_dim)
        numer = 2 * (w*intersection).sum(dim=chan_dim)
        denom = (w*union).sum(dim=chan_dim)


        dice = ((numer) / (denom + 1e-10)).sum()/batch_size

        return 1 - dice
