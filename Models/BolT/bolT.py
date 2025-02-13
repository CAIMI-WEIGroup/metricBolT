import torch
from torch import nn

import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# import transformers

from Models.BolT.bolTransformerBlock import BolTransformerBlock

# 定义BolT类，继承自nn.Module
class BolT(nn.Module):
    def __init__(self, hyperParams, details):

        super().__init__()

        # 输入数据的维度
        dim = hyperParams.dim

        # 存储超参数
        self.hyperParams = hyperParams
        # 输入层的LayerNorm标准化
        self.inputNorm = nn.LayerNorm(dim)

        # 定义一个类别标记，用于表示序列的全局信息
        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        # 用于存储Transformer块的列表
        self.blocks = []

        # 定义窗口移动的大小 步长
        shiftSize = int(hyperParams.windowSize * hyperParams.shiftCoeff)
        # 存储窗口移动的大小
        self.shiftSize = shiftSize
        # 用于存储每层的接收域大小
        self.receptiveSizes = []

        # 初始化每个Transformer块
        for i, layer in enumerate(range(hyperParams.nOfLayers)):

            # 根据focalRule来计算接收域的大小
            if(hyperParams.focalRule == "expand"):
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * i * hyperParams.fringeCoeff * (1-hyperParams.shiftCoeff))
            elif(hyperParams.focalRule == "fixed"):
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * 1 * hyperParams.fringeCoeff * (1-hyperParams.shiftCoeff))

            print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            # 存储接收域大小
            self.receptiveSizes.append(receptiveSize)

            # 创建并添加BolTransformerBlock
            self.blocks.append(BolTransformerBlock(
                dim = hyperParams.dim,
                numHeads = hyperParams.numHeads,
                headDim= hyperParams.headDim,
                windowSize = hyperParams.windowSize,
                receptiveSize = receptiveSize,
                shiftSize = shiftSize,
                mlpRatio = hyperParams.mlpRatio,
                attentionBias = hyperParams.attentionBias,
                drop = hyperParams.drop,
                attnDrop = hyperParams.attnDrop
            ))

        # 将blocks列表转换为ModuleList，以便它们能被正确注册
        self.blocks = nn.ModuleList(self.blocks)

        # Encoder后的LayerNorm标准化
        self.encoder_postNorm = nn.LayerNorm(dim)

        # 定义分类器头
        if hyperParams.method == "concat":
            # 分类的类别数
            nOfClasses = details.nOfClasses
            self.classifierHead = nn.Linear(dim, nOfClasses)

        # for token painting # 用于token painting的变量，暂未使用
        self.last_numberOfWindows = None

        # for analysis only # 仅用于分析的tokens列表，暂未使用
        self.tokens = []

        # 调用初始化权重的方法
        self.initializeWeights()

    # 初始化权重的方法
    def initializeWeights(self):
        # a bit arbitrary
        # 初始化clsToken的权重
        torch.nn.init.normal_(self.clsToken, std=1.0)

    # 计算模型浮点操作数（FLOPs）的方法
    # T: 输入信号的时间长度
    def calculateFlops(self, T):

        windowSize = self.hyperParams.windowSize # 窗口大小
        shiftSize = self.shiftSize # 窗口移动的步长
        focalSizes = self.focalSizes  # 各层的焦点大小列表
 
        macs = [] # 存储每层的乘加操作数（MACs）

        nW = (T-windowSize) // shiftSize  + 1 # 计算窗口的数量

        C = 400 # for schaefer atlas # Schaefer图谱的区域数量，用于脑连接性数据
        H = self.hyperParams.numHeads  # 注意力头的数量
        D = self.hyperParams.headDim  # 每个头的维度

        for l, focalSize in enumerate(focalSizes):

            # 对于每层，计算其MACs
            mac = 0

            # MACS from attention calculation
            # 从注意力计算中得到的MACs
            # 投影输入
                # projection in
            mac += nW * (1+windowSize) * C * H * D * 3

                # attention, softmax is omitted
            # 注意力计算（忽略softmax）
            mac += 2 * nW * H * D * (1+windowSize) * (1+focalSize)

                # projection out
            # 投影输出
            mac += nW * (1+windowSize) * C * H * D


            # MACS from MLP layer (2 layers with expand ratio = 1)
            # 来自MLP层的MACs（两层，扩展比率=1）
            mac += 2 * (T+nW) * C * C
            # 将计算的MACs添加到列表中
            macs.append(mac)

        # 返回每层的MACs和总的FLOPs（FLOPs = 2 * MAC）
        return macs, np.sum(macs) * 2 # FLOPS = 2 * MAC


    def forward(self, roiSignals, analysis=False):
        
        """
            Input : 
            
                roiSignals : (batchSize, N, dynamicLength)

                analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 
            
            Output:

                logits : (batchSize, #ofClasses)

            模型的前向传播方法。

            输入：
                roiSignals:应该就是BOLD序列 区域兴趣信号，维度为(batchSize, N, dynamicLength)，其中N为区域数量，dynamicLength为时间长度。
                analysis: 布尔值，仅当需要分析模型时设置为True。

            输出：
                logits: 模型输出的类别得分，维度为(batchSize, #ofClasses)。

        """

        # roiSignals应该是BOLD tokens
        # 调整roiSignals的维度以适配模型结构，从(batchSize, N, dynamicLength)变为(batchSize, dynamicLength, N)
        roiSignals = roiSignals.permute((0,2,1))

        batchSize = roiSignals.shape[0]
        # 时间序列长度
        T = roiSignals.shape[1] # dynamicLength

        # 计算需要的窗口数量
        nW = (T-self.hyperParams.windowSize) // self.shiftSize  + 1
        # 复制类别标记以匹配窗口数量
        cls = self.clsToken.repeat(batchSize, nW, 1) # (batchSize, #windows, C)
        
        # record nW and dynamicLength, need in case you want to paint those tokens later
        # 记录窗口数量和时间长度，以备后续使用
        self.last_numberOfWindows = nW
        
        if(analysis):
            self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        for block in self.blocks:
            # 对每个BolTransformerBlock执行前向传播
            roiSignals, cls = block(roiSignals, cls, analysis)
            
            if(analysis):
                # 如果进行分析，记录每一步的cls和roiSignals
                self.tokens.append(torch.cat([cls, roiSignals], dim=1))
        
        """
            roiSignals : (batchSize, dynamicLength, featureDim)
            cls : (batchSize, nW, featureDim)
            在经过所有Transformer块后：
            roiSignals的维度为(batchSize, dynamicLength, featureDim)
            cls的维度为(batchSize, nW, featureDim)
        """

        # 对cls应用LayerNorm
        cls = self.encoder_postNorm(cls)

        if self.hyperParams.method == "concat":
            # 根据池化策略选择分类器头的输入
            if(self.hyperParams.pooling == "cls"):
                # 如果使用cls池化，取cls的均值作为输入
                logits = self.classifierHead(cls.mean(dim=1)) # (batchSize, #ofClasses)
            elif(self.hyperParams.pooling == "gmp"):
                # 如果使用全局平均池化，取roiSignals的均值作为输入
                logits = self.classifierHead(roiSignals.mean(dim=1))

            torch.cuda.empty_cache()

            # 返回模型输出和更新的cls
            return logits, cls

        else:
            cls_ = cls.mean(dim=1)

            torch.cuda.empty_cache()
            return cls_

