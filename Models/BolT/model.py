from Models.BolT.bolT import BolT
import torch
from pytorch_metric_learning import losses,distances
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f


class Model():

    # 定义模型类，初始化方法接收超参数hyperParams和细节details作为输入
    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details

        # 根据传入的超参数和细节创建BolT模型实例
        self.model = BolT(hyperParams, details)

        # load model into gpu
        # 将模型移动到指定的设备（通常是GPU）上
        self.model = self.model.to(details.device)

        # self.miner = MultiSimilarityMiner(epsilon=0.00001)

        # set criterion
        # 设置损失函数，这里是交叉熵损失，带有可选的标签平滑处理
        # self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)#, weight = classWeights)
        self.distance = distances.CosineSimilarity()
        if hyperParams.loss == "TripletMarginLoss":
            self.criterion = losses.TripletMarginLoss(margin=0.7,distance=self.distance, swap=True)
        elif hyperParams.loss == "MultiSimilarityLoss":
            self.criterion = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.7)
        elif hyperParams.loss == "CircleLoss":
            self.criterion = losses.CircleLoss(m=0.1, gamma=60)
        # set optimizer
        # 设置优化器，使用Adam算法，学习率由hyperParams.lr决定，并加入权重衰减项
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = hyperParams.lr, weight_decay = hyperParams.weightDecay)

        # set scheduler
        # 设置学习率调度器，这里采用OneCycleLR策略
        # 计算每轮训练的步数（根据总训练样本数量和批次大小）
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))


        # 设置学习率从最大值线性下降至初始值，然后继续下降至最小值
        # 计算最大学习率 (hyperParams.maxLr) 与初始学习率 (hyperParams.lr) 的比值
        divFactor = hyperParams.maxLr / hyperParams.lr
        # 初始学习率与最小学习率的比值
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        #  OneCycleLR 调度器的一个参数，用于控制学习率的最终衰减
        #  在一个周期内先增加后减少学习率，以便模型更好地训练
        #  pct_start=0.3  # 在周期的 30% 时达到最大学习率
        # 学习率先上升到一个峰值，然后缓慢下降到一个很低的值。这种调度策略有助于模型更好地收敛，并且通常可以在较少的 epoch 内获得更好的性能
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)

    def metric_test_step(self,x):

        # PREPARE INPUTS
        # 准备输入数据 这个函数将输入数据和标签移动到指定的设备（通常是GPU）
        inputs= (x.to(self.details.device),)

        self.model.eval()

        # 前向传播计算预测结果和分类得分
        cls = self.model(*inputs)

        torch.cuda.empty_cache()

        return cls

    def step(self, timeseries, labels, loss_name, train=True):
        # PREPARE INPUTS
        # 准备输入数据 这个函数将输入数据和标签移动到指定的设备（通常是GPU）
        timeseries = timeseries.to(self.details.device)
        labels = labels.to(self.details.device)

        timeseries = (timeseries,)

        # DEFAULT TRAIN ROUTINE
        # 根据train标志选择训练模式或验证模式
        if (train):
            self.model.train()
        else:
            self.model.eval()

        # 前向传播计算预测结果和分类得分
        timeseries_cls = self.model(*timeseries)
        if loss_name == "TripletMarginLoss":
            positive_distance_avg, negative_distance_avg, acc = self.get_Metrics_all("TripletMargin Loss", timeseries_cls, labels)
        else:
            positive_distance_avg, negative_distance_avg, positive_loss_avg, negative_loss_avg, acc = self.get_Metrics_all(loss_name, timeseries_cls, labels)

        # hard_pairs = self.miner(timeseries_cls, labels)

        loss = self.criterion(timeseries_cls, labels)

        # 如果处于训练阶段，则执行反向传播、优化器更新操作以及学习率调度器的step()
        if (train):

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 调用学习率调度器
            if (not isinstance(self.scheduler, type(None))):
                self.scheduler.step()

        # 将损失、预测结果、概率分布和真实标签转换到CPU，并清空CUDA缓存
        loss = loss.detach().to("cpu")

        # del hard_pairs
        del labels
        del timeseries
        del timeseries_cls

        torch.cuda.empty_cache()

        # 返回损失、预测结果、概率分布和真实标签criterion
        if loss_name == "TripletMarginLoss":
            return positive_distance_avg, negative_distance_avg, acc, loss
        else:
            return positive_distance_avg, negative_distance_avg, positive_loss_avg, negative_loss_avg, acc, loss


    def get_Metrics_all(self, loss_name, embeddings, labels):

        # 假设batch_size设为8，每个batch里面有8个类别，每个类别2个数据，计算准确率时把16个数据分成2个8，分别set1={1,2,3,4,5,6,7,8},set2={1,2,3,4,5,6,7,8}
        # 然后从set1里面挑出一个，分别与set2里面的值挨个计算，找到距离最小或者说相似性最大的，查看是否属于统一类别，是则加一，得到的总数除以；反过来set2也是如此，
        # 将set1得到的值与set2得到的值加和除以，得到这个batch的平均准确率，所有batch的平均准确率加和除以batch的个数，得到一个epoch的平均准确率，以作为一个衡量标准。
        # 如果准确率很高很高接近于1，那就说明每个epoch里面的每个batch的准确率都很高，也就说明基本分类正确（因为batch随机的）

        # 将数据分成查询集和参考集
        len_query = int(len(embeddings) / 2)
        query_embeddings = embeddings[:len_query]
        query_labels = labels[:len_query]
        reference_embeddings = embeddings[len_query:]
        reference_labels = labels[len_query:]

        # 创建一个 AccuracyCalculator 的实例
        acc_calc = AccuracyCalculator(include=(["precision_at_1"]))

        # 计算准确率
        metrics1 = acc_calc.get_accuracy(
            query=query_embeddings,
            reference=reference_embeddings,
            query_labels=query_labels,
            reference_labels=reference_labels,
            ref_includes_query=False
        )

        metrics2 = acc_calc.get_accuracy(
            query=reference_embeddings,
            reference=query_embeddings,
            query_labels=reference_labels,
            reference_labels=query_labels,
            ref_includes_query=False
        )
        acc = (metrics1["precision_at_1"] + metrics2["precision_at_1"]) / 2

        del query_embeddings
        del query_labels
        del reference_embeddings
        del reference_labels
        del acc_calc
        del metrics1
        del metrics2
        torch.cuda.empty_cache()

        # 余弦相似度矩阵
        similarity_matrix = self.distance(embeddings, embeddings)
        # 计算距离矩阵（余弦距离=1-余弦相似度）
        distance_matrix = 1 - similarity_matrix

        # 获取正负样本掩码
        label = labels.unsqueeze(1)
        # 正样本掩码：形状为 (batch_size, batch_size)，其中同类样本对的位置为 True，其他位置为 False。
        pos_mask = label == label.t()
        # 负样本掩码：形状为 (batch_size, batch_size)，其中不同类样本对的位置为 True，其他位置为 False
        neg_mask = label != label.t()

        # 排除对角线元素（即样本与自身的距离）
        diag_mask = torch.eye(distance_matrix.size(0), dtype=torch.bool)
        diag_mask = diag_mask.to(self.details.device)
        positive_mask = pos_mask & ~diag_mask
        negative_mask = neg_mask & ~diag_mask

        # 计算正例对和负例对的数量
        num_positive_pairs = positive_mask.sum().item()
        num_negative_pairs = negative_mask.sum().item()

        # 计算正例距离和负例距离
        positive_distance_sum = distance_matrix[positive_mask].sum().item()
        negative_distance_sum = distance_matrix[negative_mask].sum().item()

        # 在每个batch内，所有正例对的距离加在一起除以正例对的数量，得到一个batch内的一个正例对的平均距离，
        positive_distance_avg = positive_distance_sum / num_positive_pairs
        # 在每个batch内，所有负例对的距离加在一起除以负例对的数量，得到一个batch内的一个负例对的平均距离
        negative_distance_avg = negative_distance_sum / num_negative_pairs

        if loss_name == "Circle Loss":

            anchor_positive = similarity_matrix[positive_mask]
            anchor_negative = similarity_matrix[negative_mask]
            new_mat = torch.zeros_like(similarity_matrix)

            new_mat[positive_mask] = (
                    -self.criterion.gamma
                    * torch.relu(self.criterion.op - anchor_positive.detach())
                    * (anchor_positive - self.criterion.delta_p)
            )
            new_mat[negative_mask] = (
                    self.criterion.gamma
                    * torch.relu(anchor_negative.detach() - self.criterion.on)
                    * (anchor_negative - self.criterion.delta_n)
            )

            logsumexp_pos = lmu.logsumexp(
                new_mat, keep_mask=positive_mask, add_one=False, dim=1
            )
            logsumexp_neg = lmu.logsumexp(
                new_mat, keep_mask=negative_mask, add_one=False, dim=1
            )
            soft_plus = torch.nn.Softplus(beta=1)

            pos_losses = soft_plus(logsumexp_pos)
            neg_losses = soft_plus(logsumexp_neg)

            positive_loss_sum = pos_losses.sum().item()
            negative_loss_sum = neg_losses.sum().item()

            # 在每个batch内，所有正例对的损失加在一起除以正例对的数量，得到一个batch内的一个正例对的平均损失，
            positive_loss_avg = positive_loss_sum / num_positive_pairs
            # 在每个batch内，所有负例对的损失加在一起除以负例对的数量，得到一个batch内的一个负例对的平均损失
            negative_loss_avg = negative_loss_sum / num_negative_pairs

            del anchor_positive
            del anchor_negative
            del new_mat
            del logsumexp_pos
            del logsumexp_neg
            del soft_plus
            del pos_losses
            del neg_losses
            del positive_loss_sum
            del negative_loss_sum
            torch.cuda.empty_cache()

        elif loss_name == "MultiSimilarity Loss":

            # 计算正样本对的损失矩阵
            # 正样本损失矩阵中的每个值表示每个样本与其所有正样本的相似度分数的损失。
            pos_exp = torch.exp(self.criterion.alpha * (similarity_matrix - self.criterion.base))
            pos_loss_matrix = torch.log1p(torch.sum(pos_exp * pos_mask.float(), dim=1)) / self.criterion.alpha
            pos_loss_sum = torch.sum(pos_loss_matrix)

            # 计算负样本对的损失矩阵
            # 负样本损失矩阵中的每个值表示每个样本与其所有负样本的相似度分数的损失
            neg_exp = torch.exp(self.criterion.beta * (self.criterion.base - similarity_matrix))
            neg_loss_matrix = torch.log1p(torch.sum(neg_exp * neg_mask.float(), dim=1)) / self.criterion.beta
            neg_loss_sum = torch.sum(neg_loss_matrix)

            positive_loss_sum = pos_loss_sum.detach().to("cpu")
            negative_loss_sum = neg_loss_sum.detach().to("cpu")

            # 在每个batch内，所有正例对的损失加在一起除以正例对的数量，得到一个batch内的一个正例对的平均损失，
            positive_loss_avg = positive_loss_sum / num_positive_pairs
            # 在每个batch内，所有负例对的损失加在一起除以负例对的数量，得到一个batch内的一个负例对的平均损失
            negative_loss_avg = negative_loss_sum / num_negative_pairs

            del pos_exp
            del pos_loss_matrix
            del pos_loss_sum
            del neg_exp
            del neg_loss_matrix
            del neg_loss_sum
            del positive_loss_sum
            del negative_loss_sum
            torch.cuda.empty_cache()

        elif loss_name == "TripletMargin Loss":
            del embeddings
            del labels
            del similarity_matrix
            del distance_matrix
            del label
            del pos_mask
            del neg_mask
            del diag_mask
            del positive_mask
            del negative_mask
            del num_positive_pairs
            del num_negative_pairs
            del positive_distance_sum
            del negative_distance_sum
            torch.cuda.empty_cache()
            return positive_distance_avg, negative_distance_avg, acc

        del embeddings
        del labels
        del similarity_matrix
        del distance_matrix
        del label
        del pos_mask
        del neg_mask
        del diag_mask
        del positive_mask
        del negative_mask
        del num_positive_pairs
        del num_negative_pairs
        del positive_distance_sum
        del negative_distance_sum
        torch.cuda.empty_cache()

        return positive_distance_avg, negative_distance_avg, positive_loss_avg, negative_loss_avg, acc

    # HELPER FUNCTIONS HERE
    # 辅助函数：准备输入数据
    def prepareInput(self, x, y):

        """
            x = (batchSize, N, T)
            y = (batchSize, )
            x: 输入数据，形状为(batchSize, N, T)
            y: 真实标签，形状为(batchSize, )

        """
        # to gpu now
        # 将输入数据和标签移动到指定的设备（通常是GPU）
        x = x.to(self.details.device)
        y = y.to(self.details.device)

        # 返回处理后的输入数据和标签
        return (x, ), y

    # 辅助函数：计算损失
    def getLoss(self, yHat, y, cls):
        
        # cls.shape = (batchSize, #windows, featureDim)
        # 对分类得分cls计算类均值一致性损失
        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))
        # 计算交叉熵损失
        y = y.long()
        cross_entropy_loss = self.criterion(yHat, y)
        # 结合两类损失并乘以lambdaCons系数得到总损失
        return cross_entropy_loss + clsLoss * self.hyperParams.lambdaCons


