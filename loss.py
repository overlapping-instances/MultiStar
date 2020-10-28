import torch
import torch.nn as nn

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model):
        """Wrapper for homoscedastic uncertainty loss (https://arxiv.org/abs/1705.07115).
        In contrast to the paper losses are added with the same weights, irrespective of
        whether they are regression or classification losses.
        
        Parameters:
        model -- multitaskmodel.MultitaskModel instance
        """

        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, image, overlap, stardist, objprob):
        pred_overlap, pred_stardist, pred_objprob = self.model(image)

        prec_overlap = torch.exp(-self.log_vars[0])
        prec_stardist = torch.exp(-self.log_vars[1])
        prec_objprob = torch.exp(-self.log_vars[2])
        
        loss_overlap = nn.functional.binary_cross_entropy_with_logits(pred_overlap, overlap)
        loss_stardist = StardistancesLoss()(pred_stardist, stardist, overlap, objprob)
        loss_objprob = nn.functional.binary_cross_entropy_with_logits(pred_objprob[overlap < 0.5], objprob[overlap < 0.5])

        loss = (
            prec_overlap * loss_overlap + self.log_vars[0] +
            prec_stardist * loss_stardist + self.log_vars[1] +
            prec_objprob * loss_objprob + self.log_vars[2]
        )

        return loss, loss_overlap, loss_stardist, loss_objprob, prec_overlap, prec_stardist, prec_objprob 

class StardistancesLoss(nn.Module):
    def __init__(self):
        """The loss is defined as the mean absolute distance between ray lengths, weighted by the true object probability.
        Overlap areas are excluded.
        """

        super(StardistancesLoss, self).__init__()
    
    def forward(self, pred_stardist, target_stardist, target_overlap, target_objprob):
            loss = None
            
            # extend number of channels of object probabilites and overlap by repeating values of single channel
            target_objprob = target_objprob.repeat(1, target_stardist.shape[1], 1, 1)
            target_overlap = target_overlap.repeat(1, target_stardist.shape[1], 1, 1)
            
            # create mask for pixels without overlap
            mask = (target_overlap < 0.5).int().float()

            # numerical stability
            target_objprob[target_objprob < 0] = 1e-6

            # weighted absolute error at non-overlap pixels
            loss = torch.abs(pred_stardist - target_stardist) * mask * target_objprob
            
            # average over all pixels and rays
            loss = loss.sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3])
            
            # average over minibatch
            loss = torch.mean(loss)

            return loss


