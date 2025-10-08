import torch.nn.functional as F

def bce(recon, x, mask):
    loss = F.binary_cross_entropy_with_logits(recon, x, reduction='none')
    loss_masked = (loss * mask).sum() / mask.sum()
    return loss_masked