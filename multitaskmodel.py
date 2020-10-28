import torch.nn as nn
from confnets.models.unet import UNet

class MultitaskModel(nn.Module):
    def __init__(self, out_channels, fmaps):
        """UNet with three output branches for overlap probability, object probability and star distances.

        Parameters:
        out_channels -- number of output channels of the UNet backbone
        fmaps -- tuple with number of channels per resolution level in UNet
        """

        super(MultitaskModel, self).__init__()
        self.backbone = UNet(
            in_channels=1,
            out_channels=out_channels,
            fmaps=fmaps,
            dims=2,
            depth=len(fmaps) - 1,
            final_activation=nn.ReLU(),
            norm_type="BatchNorm2d"
            )

        # output branches
        self.overlap_output = nn.Conv2d(out_channels, 1, 3, 1, (1, 1))
        self.stardist_output = nn.Conv2d(out_channels, 32, 3, 1, (1, 1))
        self.objprob_output = nn.Conv2d(out_channels, 1, 3, 1, (1, 1))

    def forward(self, x):
        x = self.backbone(x)
        pred_stardist = nn.ReLU()(self.stardist_output(x))
        pred_overlap = self.overlap_output(x)
        pred_objprob = self.objprob_output(x)

        return pred_overlap, pred_stardist, pred_objprob  