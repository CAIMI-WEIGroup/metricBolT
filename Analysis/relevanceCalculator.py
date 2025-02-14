
# hacked from https://github.com/hila-chefer/Transformer-MM-Explainability

import torch
import numpy as np
from pytorch_metric_learning import distances

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    """
        得到的是Rel
    """
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def normalizeCam(cam, nW):
    cam = cam / torch.max(cam)#torch.max(cam[:nW, :nW])
    return cam

def normalizeR(R):
    R_ = R - torch.eye(R.shape[0])
    R_ /= R.sum(dim=1, keepdim=True)
    return R_ + torch.eye(R.shape[0])

def generate_relevance(model, input):

    # 获取模型的设备
    device = next(model.parameters()).device

    # 运行模型并启用分析模式，得到输出和分类 token (CLS)
    cls = model(input, analysis=True)

    # construct the initial relevance matrix
    # 构建初始相关性矩阵
    # 获取模型的窗口移动大小
    shiftSize = model.shiftSize
    # 获取模型的窗口大小
    windowSize = model.hyperParams.windowSize
    # 获取输入的时间长度（BOLD token 的数量）
    T = input.shape[-1] # number of bold tokens
    # 动态长度：根据 shiftSize 和 windowSize 计算
    dynamicLength = ((T - windowSize) // shiftSize) * shiftSize + windowSize
    # 获取模型最后一层的窗口数量（CLS tokens 的数量）
    nW = model.last_numberOfWindows # number of cls tokens
    # 计算总的 token 数量（包括 BOLD tokens 和 CLS tokens）
    num_tokens = dynamicLength + nW
    # 初始化相关性矩阵为单位矩阵
    R = torch.eye(num_tokens, num_tokens)

    # now pass the relevance matrix through the blocks
    # 现在将相关性矩阵通过模型的各个块进行处理
    # 遍历模型的每个块
    for block in model.blocks:
        # 获取块中的注意力权重（CAM）获取每个块的全局注意力图A_G[m] m代表块的索引
        cam = block.getJuiceFlow().cpu()
        # 根据注意力权重更新相关性矩阵
        R += apply_self_attention_rules(R, cam)

    # R.shape = (dynamicLength + nW, dynamicLength + nW)

    # get the part that the window cls tokens are interested in
    # here we have relevance of each window cls token to the bold tokens
    # 提取窗口 CLS tokens 对 BOLD tokens 的相关性
    # 这里我们得到每个窗口 CLS token 对应的 BOLD token 的相关性
    # 这个应该是最终的W_imp
    # CLS token 自身的注意力: R[:nW, :nW]
    # CLS token 对 BOLD token 的注意力: R[:nW, nW:]
    # BOLD token 对 CLS token 的注意力: R[nW:, :nW]
    # BOLD token 自身的注意力: R[nW:, nW:]
    inputToken_relevances = R[:nW, nW:] # of shape (nW, dynamicLength)


    return inputToken_relevances # (nW, dynamicLength)
