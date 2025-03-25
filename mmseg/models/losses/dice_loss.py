import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
from .utils import weighted_loss
from .cross_entropy_loss import cross_entropy, _expand_onehot_labels, binary_cross_entropy, mask_cross_entropy

def my_loss(pred, label, weight=None, class_weight=None, reduction='mean',avg_factor=None,
ignore_index=-100):
    # alpha = 0.25
    # gamma = 2
    # pred= torch.nn.functional.softmax(pred, dim=1)   
    # # print(66666666666)
    # # print(target.shape)
    # # 开始计算
    # loss_y1 = -1 * alpha * \
    #     torch.pow((1 - pred[:, 1, :, :] ), gamma) * \
    #     torch.log(pred[:, 1, :, :] + eps) * target
    # loss_y0 = -1 * (1 - alpha) * torch.pow(1-pred[:, 0, :, :],
    #                                             gamma) * torch.log(pred[:, 0, :, :]  + eps) * (1 - target)
    # loss = loss_y0 + loss_y1
    # # print("###################################################")
    # # print("loss: ", loss.shape)
    # # print("###################################################")
    # return loss
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    # if weight is not None:
    #     weight = weight.float()
    # loss = weight_reduce_loss(
    #     loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def my_loss2(pred, target, eps=1e-8):
    alpha = 0.1
    gamma = 2
    pred= torch.nn.functional.softmax(pred, dim=1)   
    # print(66666666666)
    # print(target.shape)
    # 开始计算
    loss_y1 = -1 * torch.log(pred[:, 1, :, :] + eps) * target
    loss_y0 = -1 * torch.log(pred[:, 0, :, :]  + eps) * (1 - target)
    loss = loss_y0 + loss_y1
    # print("###################################################")
    # print("loss: ", loss.shape)
    # print("###################################################")
    return loss

@LOSSES.register_module()
class Dice_loss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(Dice_loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.class_weight = get_class_weight(class_weight)
        self.alpha = 0.25
        self.gamma = 2
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.debug = False
        self.debug_output = None

    def forward(self,
                pred,
                target,
                top_50_percent_mask,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = 'mean'
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None


        if weight is not None:
            weight = weight.float()
            final = my_loss(pred, target)
        else:
            final = my_loss(pred, target)
        final = weight_reduce_loss(
            final, target, top_50_percent_mask, weight=weight, reduction=reduction, avg_factor=avg_factor)

        if self.debug:
            self.debug_output = {
                'Seg. Pred.': pred.detach().cpu().numpy(),
                'Seg. GT': target.detach().cpu().numpy()
            }
        return final