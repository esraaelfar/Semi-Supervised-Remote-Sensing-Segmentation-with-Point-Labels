import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=0, reduction='mean'):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) - contains class indices, with 0 (or ignore_index) for unlabeled.
        """
        mask = (targets != self.ignore_index).float()
        
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        
        if self.reduction == 'sum':
            return loss.sum()
        else:
            num_labeled = mask.sum()
            if num_labeled > 0:
                return loss.sum() / num_labeled
            else:
                return 0.0 * loss.sum() 

