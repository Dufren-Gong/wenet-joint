import torch
from torch import nn
import torch.nn.functional as F
    
#or
#通常，α \alphaα 的值在 [0,1] 之间，表示正负样本的权重比例。对于目标检测任务，α \alphaα可以设为正样本和负样本的比例  0.16
#焦点因子 γ \gammaγ 通常设为2，但可以根据具体问题调整。更大的 γ \gammaγ 会使得模型更加专注于难分类样本
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=[1, 1], gamma=2, ignore_index=-1, reduction='mean'):
#         """
#         Focal Loss for multi-class classification.

#         Args:
#             alpha (float, optional): Weighting factor in balanced cross entropy. Default: 0.25.
#             gamma (float, optional): Focusing parameter for modulating factor (1-p). Default: 2.
#             ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -1.
#             reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
#         """
#         super(FocalLoss, self).__init__()
#         self.first_flag = True
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.loss_fcn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

#     def forward(self, inputs, targets):
#         """
#         Forward pass for focal loss.

#         Args:
#             inputs (Tensor): Logits tensor of shape (batch_size, num_classes, ...).
#             targets (Tensor): Target tensor of shape (batch_size, ...).

#         Returns:
#             Tensor: Computed focal loss.
#         """
#         # Compute the cross entropy loss without reduction
#         if self.first_flag:
#             self.first_flag = False
#             self.alpha = torch.tensor(self.alpha, device=targets.device)  # alpha now a Tensor
#             self.gamma = torch.tensor(self.gamma, device=targets.device)
#         ce_loss = self.loss_fcn(inputs, targets)
#         print(inputs)
#         print(targets)
#         print(ce_loss)
#         exit(1)
#         alpha = self._get_alpha(targets, inputs.shape[-1])

#         # Get the probabilities corresponding to the target class
#         p_t = torch.exp(-ce_loss)

        
#         # Compute the focal loss factor
#         focal_factor = (1 - p_t) ** self.gamma
        
#         # Apply the alpha weighting factor
#         focal_loss = alpha * focal_factor * ce_loss
#         # Reduce the loss based on the reduction mode
#         if self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
        
#     def _get_alpha(self, targets, number_class=2):
#         targets = torch.mul(targets, targets == 1)
#         target_one_hot = nn.functional.one_hot(targets, num_classes=number_class)
#         alpha = (target_one_hot * self.alpha).sum(dim=-1)
#         return alpha

#new
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-1, reduction='none'):
        super(FocalLoss, self).__init__()
        self.first_flag = True
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.tensor([1, 1])
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = torch.tensor(gamma)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        if self.first_flag:
            self.first_flag = False
            self.alpha = self.alpha.to(targets.device)  # alpha now a Tensor
            self.gamma = self.gamma.to(targets.device)
        # 过滤掉 ignore_index 的部分
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        # 获取每个类的概率
        inputs_soft = F.softmax(inputs, dim=1)
        # 获取目标类的概率
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (inputs_soft * targets_one_hot).sum(dim=1)

        # 获取对应的 alpha 值
        at = self.alpha.gather(0, targets)

        # 计算 Focal Loss
        log_pt = torch.log(pt)
        focal_loss = -at * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
