import torch
from einops import rearrange, repeat
import numpy as np

def windowBoldSignal(boldSignal, windowLength, stride):
    
    """
        对输入的 BOLD token信号进行窗口化处理
        输入：
            boldSignal : (batchSize, N, T)
        输出：
            windowedBoldSignals : (batchSize, (T-windowLength) // stride, N, windowLength )
            windowedBoldSignals里面包含每个窗口的BOLD tokens，batch_size可以理解为人数，每个人被划分为nW个窗口，每个窗口包含的BOLD tokens的大小为N*windowLength
            samplingEndPoints：list,包含每个窗口的结束位置（时间索引）
        output : (batchSize, (T-windowLength) // stride, N, windowLength )
                即(batchSize, nW, N, windowLength )

    """

    # 获取输入 BOLD 信号的时间维度长度 T
    T = boldSignal.shape[2]

    # NOW WINDOWING
    # 初始化用于存储窗口化的信号和采样终点的列表
    windowedBoldSignals = []
    samplingEndPoints = []

    # 计算窗口的数量 (T - windowLength) // stride + 1 并进行迭代
    for windowIndex in range((T - windowLength)//stride + 1):

        # 提取当前窗口的信号
        sampledWindow = boldSignal[:, :, windowIndex * stride  : windowIndex * stride + windowLength]
        # 记录当前窗口的结束位置
        samplingEndPoints.append(windowIndex * stride + windowLength)
        # 在维度 1 上增加一个新维度
        sampledWindow = torch.unsqueeze(sampledWindow, dim=1)
        # 将当前窗口的信号添加到列表中
        windowedBoldSignals.append(sampledWindow)

    # 将所有窗口的信号在维度 1 上拼接起来
    windowedBoldSignals = torch.cat(windowedBoldSignals, dim=1)
    # 返回窗口化的信号和采样终点
    return windowedBoldSignals, samplingEndPoints

def get_loss(embeddings, labels):
    """
    计算同类样本之间的余弦相似度损失
    Args:
        embeddings: shape为(2*batch_size, embedding_dim)的张量
        labels: shape为(2*batch_size,)的标签张量
    Returns:
        loss_value: 标量损失值
    """
    import torch
    import torch.nn.functional as F
    
    batch_size = len(labels) // 2
    # 归一化embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    loss_value = 0.0
    # 计算每对同类样本之间的损失
    for i in range(batch_size):
        # 找到第i个样本对应的另一个同类样本的索引
        current_label = labels[i]
        pair_idx = i + batch_size
        
        # 计算余弦相似度（由于已经归一化，直接点积即可）
        cos_sim = torch.dot(embeddings[i], embeddings[pair_idx])
        
        # 计算损失：使用平方损失，目标是使相似度为0
        loss_value += cos_sim.pow(2)
    
    # 对损失取平均
    loss_value = loss_value / batch_size
    
    return loss_value
