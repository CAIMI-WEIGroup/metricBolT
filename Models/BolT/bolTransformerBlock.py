
import torch
from torch import nn

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from Models.BolT.util import windowBoldSignal


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 1,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):
    '''
        WindowAttention 类的主要作用是：

        局部注意力计算： 将输入序列分成多个窗口，在每个窗口内独立计算注意力。
        降低计算复杂度： 避免全局注意力机制的高计算复杂度，特别是对于长序列输入。
        提高模型效率： 在保证模型性能的同时，提高计算效率和内存利用率。
    '''

    def __init__(self, dim, windowSize, receptiveSize, numHeads, headDim=20, attentionBias=True, qkvBias=True, attnDrop=0., projDrop=0.):

        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim ** -0.5

        self.attentionBias = attentionBias

        # define a parameter table of relative position bias
        # 定义相对位置偏置参数表
        maxDisparity = windowSize - 1 + (receptiveSize - windowSize)//2

        # 定义 cls token 的偏置参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*maxDisparity+1, numHeads))  # maxDisparity, nH

        self.cls_bias_sequence_up = nn.Parameter(torch.zeros((1, numHeads, 1, receptiveSize)))
        self.cls_bias_sequence_down = nn.Parameter(torch.zeros(1, numHeads, windowSize, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, numHeads, 1, 1)))

        # get pair-wise relative position index for each token inside the window
        # 获取窗口内每个 token 的相对位置索引
        coords_x = torch.arange(self.windowSize) # N
        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize)//2 # M
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # N, M
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index", relative_position_index)

        # 定义 Q 和 KV 线性变换层
        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)
        # 定义 dropout 层
        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)


        self.projDrop = nn.Dropout(projDrop)

        # prep the biases
        # 初始化偏置参数
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.cls_bias_sequence_up, std=.02)
        trunc_normal_(self.cls_bias_sequence_down, std=.02)
        trunc_normal_(self.cls_bias_self, std=.02)
        
        self.softmax = nn.Softmax(dim=-1)


        # for token painting
        # 用于 token painting 的注意力图和梯度
        self.attentionMaps = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionGradients = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.nW = None

    # 保存注意力图
    def save_attention_maps(self, attentionMaps):
        self.attentionMaps = attentionMaps

    # 保存注意力梯度
    def save_attention_gradients(self, grads):
        self.attentionGradients = grads

    # Juice" 代表了通过注意力机制传播的信息量或影响力。具体来说，它是注意力图（attention map）和对应的注意力梯度（attention gradients）的结合
    # 注意力图表示的是模型在不同位置之间的注意力权重，而注意力梯度则反映了这些权重对模型输出结果的影响程度
    # 通过将注意力图和注意力梯度逐元素相乘，可以得到一个新的矩阵，这个矩阵代表了每个位置对整体注意力机制的贡献


    # 计算跨所有注意力头部的平均 Juice 值。这个函数将注意力图（cam）和注意力梯度（grad）逐元素相乘，
    # 然后对结果进行处理，以生成一个综合的 Juice 矩阵。
    # def averageJuiceAcrossHeads(self, cam, grad):
    def averageJuiceAcrossHeads(self, cam):

        """
            Hacked from the original paper git repo ref: https://github.com/hila-chefer/Transformer-MM-Explainability
            cam : (numberOfHeads, n, m)
            grad : (numberOfHeads, n, m)
        """

        #cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        #grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        # 逐元素相乘 grad 和 cam
        # cam = grad * cam
        # 对 cam 进行 clamping，使其最小值为 0，然后在最后一个维度上求平均
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    # 注意：此函数假设只有一个对象需要分析。如果要继续使用此实现，请为每个对象分别生成相关性图
    # 计算全局 Juice 矩阵。它将多个局部窗口内的 Juice 信息聚合到一个全局矩阵中，用于分析整个序列中的注意力机制。
    def getJuiceFlow(self, shiftSize): # NOTE THAT, this functions assumes there is only one subject to analyze. So if you want to keep using this implementation, generate relevancy maps one by one for each subject

        # infer the dynamic length
        # 推断动态长度
        dynamicLength = self.windowSize + (self.nW - 1) * shiftSize

        # 获取目标注意力图和注意力梯度
        targetAttentionMaps = self.attentionMaps # (nW, h, n, m) 
        # targetAttentionGradients = self.attentionGradients #self.attentionGradients # (nW h n m)

        # 初始化全局 juice 矩阵和归一化矩阵
        globalJuiceMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)
        normalizerMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)


        # aggregate(by averaging) the juice from all the windows
        # 通过平均化从所有窗口聚合 juice
        for i in range(self.nW):

            # average the juices across heads # 在头部间平均 juice
            # window_averageJuice = self.averageJuiceAcrossHeads(targetAttentionMaps[i], targetAttentionGradients[i]) # of shape (1+windowSize, 1+receptiveSize)
            window_averageJuice = self.averageJuiceAcrossHeads(targetAttentionMaps[i])

            # now broadcast the juice to the global juice matrix.
            # 现在将 juice 广播到全局 juice 矩阵中
            # set boundaries for overflowing focal attentions
            # 设置焦点注意力的溢出边界
            L = (self.receptiveSize-self.windowSize)//2

            overflow_left = abs(min(i*shiftSize - L, 0))
            overflow_right = max(i*shiftSize + self.windowSize + L - dynamicLength, 0)

            leftMarker_global = i*shiftSize - L + overflow_left
            rightMarker_global = i*shiftSize + self.windowSize + L - overflow_right
            
            leftMarker_window = overflow_left
            rightMarker_window = self.receptiveSize - overflow_right

            # first the cls it self  # 首先是 cls 自身
            globalJuiceMatrix[i, i] += window_averageJuice[0,0]
            normalizerMatrix[i, i] += 1
            # cls to bold tokens # cls 到 bold tokens
            globalJuiceMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window])
            # bold tokens to cls # bold tokens 到 cls
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += window_averageJuice[1:, 0]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += torch.ones_like(window_averageJuice[1:, 0])
            # bold tokens to bold tokens # bold tokens 到 bold tokens
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window])

        # to prevent divide by zero for those non-existent attention connections
        # 防止除以零的情况，对于那些不存在的注意力连接
        normalizerMatrix[normalizerMatrix == 0] = 1
        # 将全局 juice 矩阵除以归一化矩阵
        globalJuiceMatrix = globalJuiceMatrix / normalizerMatrix

        return globalJuiceMatrix

    def forward(self, x, x_, mask, nW, analysis=False):
        """
            Input:

            x: base BOLD tokens with shape of (B*num_windows, 1+windowSize, C), the first one is cls token
            x_: receptive BOLD tokens with shape of (B*num_windows, 1+receptiveSize, C), again the first one is cls token
            mask: (mask_left, mask_right) with shape (maskCount, 1+windowSize, 1+receptiveSize)
            nW: number of windows
            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 

            Output:

            transX : attended BOLD tokens from the base of the window, shape = (B*num_windows, 1+windowSize, C), the first one is cls token

        """


        B_, N, C = x.shape
        _, M, _ = x_.shape

        # 减去 cls token 的位置
        N = N-1  # 窗口数量
        M = M-1  # 感受野大小

        # 计算批量大小
        B = B_ // nW

        # 解包 mask
        mask_left, mask_right = mask

        # linear mapping
        q = self.q(x) # (batchSize * #windows, 1+N, C)
        k, v = self.kv(x_).chunk(2, dim=-1) # (batchSize * #windows, 1+M, C)

        # head seperation
        # 分离多头 n=1+N(CLSToken+WindowTokens), m=1+M(CLSToken+ReceptiveTokens), d=C
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)

        # 计算注意力得分 h注意力头数，也就是窗口数量
        attn = torch.matmul(q , k.transpose(-1, -2)) * self.scale # (batchSize*#windows, h, n, m)

        # 获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, M, -1)  # N, M, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, M

        # 如果有注意力偏置，则加上
        if(self.attentionBias):
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up
            attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down
        
        # mask the not matching queries and tokens here
        # 获取 mask 的数量
        maskCount = mask_left.shape[0]
        # repate masks for batch and heads
        # 为 batch 和 head 重复 mask
        mask_left = repeat(mask_left, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        mask_right = repeat(mask_right, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        # 设置 mask 的值为负无穷大
        mask_value = max_neg_value(attn)

        # 调整注意力矩阵的形状
        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW = nW)        
        
        # make sure masks do not overflow
        # 确保 mask 不会溢出
        maskCount = min(maskCount, attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]

        # 应用 mask
        attn[:, :maskCount].masked_fill_(mask_left==1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right==1, mask_value)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")

        # 对注意力得分进行 softmax 归一化
        attn = self.softmax(attn) # (b, h, n, m)

        # 如果进行分析，则保存注意力图和梯度
        if(analysis):
            # 保存注意力图（detached 表示不参与梯度计算）
            self.save_attention_maps(attn.detach()) # save attention
            # 用于保存注意力图的梯度
            # handle = attn.register_hook(self.save_attention_gradients) # save it's gradient
            self.nW = nW
            # self.handle = handle

        attn = self.attnDrop(attn)
        # 应用注意力矩阵于值向量 v，得到加权后的输出
        x = torch.matmul(attn, v) # of shape (b_, h, n, d)
        # 将多头注意力的输出重新排列并合并
        x = rearrange(x, 'b h n d -> b n (h d)')
        # 应用输出投影
        x = self.proj(x)
        # 应用投影后的 dropout
        x = self.projDrop(x)
        
        return x



class FusedWindowTransformer(nn.Module):

    def __init__(self, dim, windowSize, shiftSize, receptiveSize, numHeads, headDim, mlpRatio, attentionBias, drop, attnDrop):
        
        super().__init__()


        self.attention = WindowAttention(dim=dim, windowSize=windowSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, attentionBias=attentionBias, attnDrop=attnDrop, projDrop=drop)
        
        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.shiftSize = shiftSize

    def getJuiceFlow(self):  
        return self.attention.getJuiceFlow(self.shiftSize)

    def forward(self, x, cls, windowX, windowX_, mask, nW, analysis=False):
        """

            Input: 

            x : (B, T, C)
            cls : (B, nW, C)
            windowX: (B, 1+windowSize, C)
            windowX_ (B, 1+windowReceptiveSize, C)
            mask : (B, 1+windowSize, 1+windowReceptiveSize)
            nW : number of windows

            analysis : Boolean, it is set True only when you want to analyze the model, otherwise not important 

            Output:

            xTrans : (B, T, C)
            clsTrans : (B, nW, C)

        """

        # WINDOW ATTENTION  windowXTrans：attended BOLD tokens from the base of the window, shape = (B*num_windows, 1+windowSize, C), the first one is cls token
        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW, analysis=analysis) # (B*nW, 1+windowSize, C)
        # 提取 cls token 的转换结果
        clsTrans = windowXTrans[:,:1] # (B*nW, 1, C)
        # 提取窗口化后的 x 的转换结果
        xTrans = windowXTrans[:,1:] # (B*nW, windowSize, C)
        # 重新排列 clsTrans 的形状
        clsTrans = rearrange(clsTrans, "(b nW) l c -> b (nW l) c", nW=nW)
        # 重新排列 xTrans 的形状
        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)
        # FUSION # 融合窗口化后的 xTrans
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)
        
        # residual connections
        # 残差连接
        clsTrans = clsTrans + cls
        xTrans = xTrans + x

        # MLP layers
        # 通过前馈网络层
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))
        clsTrans = clsTrans + self.mlp(self.mlp_norm(clsTrans))

        return xTrans, clsTrans

    # 定义 gatherWindows 方法，用于将窗口化的 X 聚合到动态长度的张量中 相当与Token Fuser
    def gatherWindows(self, windowedX, dynamicLength, shiftSize):
        
        """
        Input:
            windowedX : (batchSize, nW, windowLength, C)
            scatterWeights : (windowLength, )
        
        Output:
            destination: (batchSize, dynamicLength, C)
        
        """

        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]
        
        device = windowedX.device


        destination = torch.zeros((batchSize, dynamicLength,  C)).to(device)
        scalerDestination = torch.zeros((batchSize, dynamicLength, C)).to(device)

        # 生成索引矩阵，用于在后续步骤中进行 scatter_add 操作
        indexes = torch.tensor([[j+(i*shiftSize) for j in range(windowLength)] for i in range(nW)]).to(device)
        # 将索引矩阵扩展为 (batchSize, nW, windowLength, 1) 的形状，并沿着最后一维复制 C 次
        indexes = indexes[None, :, :, None].repeat((batchSize, 1, 1, C)) # (batchSize, nW, windowSize, featureDim)
        # 重新排列 windowedX 以便于 scatter 操作
        src = rearrange(windowedX, "b n w c -> b (n w) c")
        # 重新排列 indexes 以便于 scatter 操作
        indexes = rearrange(indexes, "b n w c -> b (n w) c")
        # 使用 scatter_add_ 将窗口化的张量聚合到目标张量中
        destination.scatter_add_(dim=1, index=indexes, src=src)

        # 创建缩放源张量，用于计算每个位置的加权和
        scalerSrc = torch.ones((windowLength)).to(device)[None, None, :, None].repeat(batchSize, nW, 1, C) # (batchSize, nW, windowLength, featureDim)
        # 重新排列缩放源张量
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")
        #  使用 scatter_add_ 将缩放源张量聚合到缩放目标张量中
        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)
        # 对目标张量进行缩放，以确保每个位置的值是正确的平均值
        destination = destination / scalerDestination


        return destination

    

class BolTransformerBlock(nn.Module):

    def __init__(self, dim, numHeads, headDim, windowSize, receptiveSize, shiftSize, mlpRatio=1.0, drop=0.0, attnDrop=0.0, attentionBias=True):

        # 确保感受野大小与窗口大小的差是偶数
        assert((receptiveSize-windowSize)%2 == 0)

        super().__init__()
        self.transformer = FusedWindowTransformer(dim=dim, windowSize=windowSize, shiftSize=shiftSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, mlpRatio=mlpRatio, attentionBias=attentionBias, drop=drop, attnDrop=attnDrop)

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize

        # 计算余量，即感受野大小和窗口大小之差的一半 感觉应该是双边区域的大小
        self.remainder = (self.receptiveSize - self.windowSize) // 2

        # create mask here for non matching query and key pairs
        # 创建 mask，用于过滤不匹配的 query 和 key 对
        maskCount = self.remainder // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)
        mask_right = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)

        # 为每个 mask 填充合适的位置
        for i in range(maskCount):
            if(self.remainder > 0):
                mask_left[i, :, 1:1+self.remainder-shiftSize*i] = 1
                if(-self.remainder+shiftSize*i > 0):
                    mask_right[maskCount-1-i, :, -self.remainder+shiftSize*i:] = 1

        # 将 mask 注册为 buffer，确保在模型保存和加载时它们也会被保存和加载
        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)


    def getJuiceFlow(self):
        return self.transformer.getJuiceFlow()
    
    def forward(self, x, cls, analysis=False):
        """
        Input:
            x : (batchSize, dynamicLength, c) BOLD tokens
            cls : (batchSize, nW, c)
        
            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 


        Output:
            fusedX_trans : (batchSize, dynamicLength, c)
            cls_trans : (batchSize, nW, c)

        """

        B, Z, C = x.shape
        device = x.device

        #update z, incase some are dropped during windowing # 更新 z，以防在窗口化过程中丢掉一些
        # 更新 Z，以防在窗口化过程中丢掉一些    步长shiftSize = windowSize * shiftCoeff
        Z = self.windowSize + self.shiftSize * (cls.shape[1]-1)
        x = x[:, :Z]

        # form the padded x to be used for focal keys and values
        # 构建填充后的 x，用于 focal keys 和 values
        x_ = torch.cat([torch.zeros((B, self.remainder,C),device=device), x, torch.zeros((B, self.remainder,C), device=device)], dim=1) # (B, remainder+Z+remainder, C) 

        # window the sequences
        # 对序列进行窗口化
        windowedX, _ = windowBoldSignal(x.transpose(2,1), self.windowSize, self.shiftSize) # (B, nW, C, windowSize)         
        # windowedX = (batchSize, (T-windowLength) // stride, N, windowLength )即(batchSize, nW, N/C, windowLength )
        # batch_size可以理解为人数，每个人被划分为nW个窗口，每个窗口包含的BOLD tokens的大小为N*windowLength
        windowedX = windowedX.transpose(2,3) # (B, nW, windowSize, C)

        windowedX_, _ = windowBoldSignal(x_.transpose(2,1), self.receptiveSize, self.shiftSize) # (B, nW, C, receptiveSize)
        windowedX_ = windowedX_.transpose(2,3) # (B, nW, receptiveSize, C)

        # 获取窗口数量
        nW = windowedX.shape[1] # number of windows

        # 拼接 cls token 和窗口化后的输入
        xcls = torch.cat([cls.unsqueeze(dim=2), windowedX], dim = 2) # (B, nW, 1+windowSize, C)
        xcls = rearrange(xcls, "b nw l c -> (b nw) l c") # (B*nW, 1+windowSize, C) 
       
        xcls_ = torch.cat([cls.unsqueeze(dim=2), windowedX_], dim=2) # (B, nw, 1+receptiveSize, C)
        xcls_ = rearrange(xcls_, "b nw l c -> (b nw) l c") # (B*nW, 1+receptiveSize, C)

        masks = [self.mask_left, self.mask_right]

        # pass to fused window transformer
        fusedX_trans, cls_trans = self.transformer(x, cls, xcls, xcls_, masks, nW, analysis) # (B*nW, 1+windowSize, C)


        return fusedX_trans, cls_trans





