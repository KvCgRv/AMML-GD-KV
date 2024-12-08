from torch.nn import Module, Parameter
import torch
import torch.nn.functional as F
import math
import torch.nn as nn

# 处理结果(M,N) ---> 角度间隔+约束后的(M,N)
class FC(Module):

    def __init__(self, fc_type='MV-AM', margin=0.25, t=0.2, scale=32, embedding_size=8, num_class=2,
                 easy_margin=True):
        super(FC, self).__init__()
        self.weight = Parameter(torch.Tensor(embedding_size, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.t = t
        self.easy_margin = easy_margin
        self.scale = scale
        self.fc_type = fc_type
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # duplication formula
        self.iter = 0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x, label):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        # 参数矩阵归一化
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)  # wx+b 求和后就是未施加约束的分母，即softmax中的分母
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        batch_size = label.size(0)

        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # ground truth score：即分子
        # (batch,1)
        if self.fc_type == 'FC':  # 普通的网络
            final_gt = gt   # (batch,1)，即每个样本相对于真实标签的预测概率，loss = sum(1-gt)

        # 待解决
        elif self.fc_type == 'SphereFace':
            self.iter += 1

            # 5,10000,0.0001,1,-2
            self.cur_lambda = max(self.lambda_min, self.base * (1 + self.alpha * self.iter) ** (-1 * self.power))
            # self.margin = 0.35
            cos_theta_m = self.margin_formula[int(self.margin)](gt)  # cos(margin * gt)
            theta = gt.data.acos()
            k = ((self.margin * theta) / math.pi).floor()
            phi_theta = ((-1.0) ** k) * cos_theta_m - 2 * k
            final_gt = (self.cur_lambda * gt + phi_theta) / (1 + self.cur_lambda)

        elif self.fc_type == 'AM':  # cosface
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin

        elif self.fc_type == 'Arc':  # arcface
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m

        elif self.fc_type == 'MV-AM':
            # gt(batch,1)  cos_theta(batch,N)
            mask = cos_theta > (gt - self.margin)  # 找出难样本hard，如果是>gt 那么就是semi-hard
            hard_vector = cos_theta[mask]    #
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin

        elif self.fc_type == 'MV-Arc':
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)

        else:
            raise Exception('unknown fc type!')
        # 替换cos_theta中的值，即对应的gt加约束后的值
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        # cos_theta *= self.scale
        return cos_theta  # 这里的costheta大小为(batch,N)，可以直接进行损失的计算了


# Loss functions
def loss_final(pred, label, loss_type, criteria, save_rate=0.9, gamma=2.0):
    if loss_type == 'Softmax':
        criteria = nn.CrossEntropyLoss()
        loss_final = criteria(pred,label)
    elif loss_type == 'FocalLoss':
        assert (gamma >= 0)
        input = F.cross_entropy(pred, label, reduce=False)
        pt = torch.exp(-input)
        loss = (1 - pt) ** gamma * input
        loss_final = loss.mean()
    elif loss_type == 'HardMining':
        batch_size = pred.shape[0]
        loss = F.cross_entropy(pred, label, reduce=False)
        ind_sorted = torch.argsort(-loss) # from big to small
        num_saved = int(save_rate * batch_size)
        ind_update = ind_sorted[:num_saved]
        loss_final = torch.sum(F.cross_entropy(pred[ind_update], label[ind_update]))
    else:
        raise Exception('unknown loss type!!')

    return loss_final


class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)