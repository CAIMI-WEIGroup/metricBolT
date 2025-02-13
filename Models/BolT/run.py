from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option

from Models.BolT.model import Model
from Dataset.dataset import getDataset
import pandas as pd

def train_Metric(model, dataset, epoch):

    dataLoader = dataset.getIdx(train=True)

    losses = []
    pos_distances = []
    neg_distances = []
    if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":
        pos_losses = []
        neg_losses = []
    accuracy = []


    for i, data in enumerate(tqdm(dataLoader, ncols=120, desc=f'training epoch:{epoch}')):
        # 提取批次中的时间序列特征数据（形状：[批次大小, N, 动态长度]）N是脑区数目
        timeseries = data["timeseries"]  # (batchSize, N, dynamicLength)
        # 提取批次中的标签数据（形状：[批次大小]）
        label = data["label"]  # (batchSize, N, dynamicLength)

        num_tensors = 2 * timeseries[0].shape[0]
        tensor_shape = timeseries[0][0].shape
        timeseries_ = torch.zeros((num_tensors, *tensor_shape))
        labels_ = torch.zeros((num_tensors))
        for i in range(timeseries[0].shape[0]):
            timeseries_[i] = timeseries[0][i]
            labels_[i] = label[0][i]
            timeseries_[i + timeseries[0].shape[0]] = timeseries[1][i]
            labels_[i + timeseries[0].shape[0]] = label[1][i]

        timeseries = timeseries_
        labels = labels_

        # NOTE: xTrain and yTrain are still on "cpu" at this point
        # 注意：此时xTrain和yTrain仍在CPU上

        # model.step(timeseries, labels, loss_name=model.hyperParams.loss, train=True)

        # # 执行模型的一个训练步骤，返回训练损失、预测结果、概率分布及真实的标签
        if model.hyperParams.loss == "TripletMarginLoss":
            positive_distance_avg, negative_distance_avg, acc, loss = model.step(timeseries, labels, loss_name=model.hyperParams.loss, train=True)
        else:
            positive_distance_avg, negative_distance_avg, positive_loss_avg, negative_loss_avg, acc, loss = model.step(timeseries, labels, loss_name=model.hyperParams.loss, train=True)
        losses.append(loss)
        pos_distances.append(positive_distance_avg)
        neg_distances.append(negative_distance_avg)
        if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":
            pos_losses.append(positive_loss_avg)
            neg_losses.append(negative_loss_avg)
        accuracy.append(acc)


    if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":

        return sum(losses), sum(pos_distances) / len(pos_distances), sum(neg_distances) / len(neg_distances), sum(pos_losses) / len(pos_losses), sum(neg_losses) / len(neg_losses), sum(accuracy) / len(accuracy)

    # 返回整个训练过程中这一轮的所有迭代过程的正样本距离、负样本距离和损失值，都分别是一个list,每个元素代表一个迭代过程产生的损失值
    return sum(losses), sum(pos_distances) / len(pos_distances), sum(neg_distances) / len(neg_distances), sum(accuracy) / len(accuracy)

def val_Metric(model, dataset, epoch):

    dataLoader = dataset.getIdx(train=False)

    losses = []
    pos_distances = []
    neg_distances = []
    if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":
        pos_losses = []
        neg_losses = []
    accuracy = []

    for i, data in enumerate(tqdm(dataLoader, ncols=120, desc=f'validing epoch:{epoch}')):
        # 提取批次中的时间序列特征数据（形状：[批次大小, N, 动态长度]）N是脑区数目
        timeseries = data["timeseries"]  # (batchSize, N, dynamicLength)
        # 提取批次中的标签数据（形状：[批次大小]）
        label = data["label"]  # (batchSize, N, dynamicLength)

        num_tensors = 2 * timeseries[0].shape[0]
        tensor_shape = timeseries[0][0].shape
        timeseries_ = torch.zeros((num_tensors, *tensor_shape))
        labels_ = torch.zeros((num_tensors))
        for i in range(timeseries[0].shape[0]):
            timeseries_[i] = timeseries[0][i]
            labels_[i] = label[0][i]
            timeseries_[i + timeseries[0].shape[0]] = timeseries[1][i]
            labels_[i + timeseries[0].shape[0]] = label[1][i]

        timeseries = timeseries_
        labels = labels_

        # NOTE: xTrain and yTrain are still on "cpu" at this point
        # 注意：此时xTrain和yTrain仍在CPU上

        # # 执行模型的一个训练步骤，返回训练损失、预测结果、概率分布及真实的标签
        if model.hyperParams.loss == "TripletMarginLoss":
            positive_distance_avg, negative_distance_avg, acc, loss = model.step(timeseries, labels,loss_name=model.hyperParams.loss,train=True)
        else:
            positive_distance_avg, negative_distance_avg, positive_loss_avg, negative_loss_avg, acc, loss = model.step(timeseries, labels, loss_name=model.hyperParams.loss, train=True)
        losses.append(loss)
        pos_distances.append(positive_distance_avg)
        neg_distances.append(negative_distance_avg)
        if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":
            pos_losses.append(positive_loss_avg)
            neg_losses.append(negative_loss_avg)
        accuracy.append(acc)

        pos_distances.append(positive_distance_avg)
        neg_distances.append(negative_distance_avg)
        if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":
            pos_losses.append(positive_loss_avg)
            neg_losses.append(negative_loss_avg)
        accuracy.append(acc)
        losses.append(loss)

    if model.hyperParams.loss == "CircleLoss" or model.hyperParams.loss == "MultiSimilarityLoss":
        return sum(losses), sum(pos_distances) / len(pos_distances), sum(neg_distances) / len(neg_distances),sum(pos_losses)/len(pos_losses),sum(neg_losses)/len(neg_losses), sum(accuracy) / len(accuracy)

    # 返回整个训练过程中这一轮的所有迭代过程的正样本距离、负样本距离和损失值，都分别是一个list,每个元素代表一个迭代过程产生的损失值
    return sum(losses), sum(pos_distances)/len(pos_distances), sum(neg_distances)/len(neg_distances), sum(accuracy)/len(accuracy)

def run_Metric(hyperParams, datasetDetails, device="cuda:3", analysis=True, timespan=None):

    # extract datasetDetails
    nOfEpochs = datasetDetails.nOfEpochs


    # 根据数据集详情构建数据集实例
    train_dataset = getDataset(datasetDetails,"train")
    # valid_dataset = getDataset(datasetDetails,"val")

    details = Option({
        "device" : device,
        "nOfTrains" : train_dataset.get_nOfTrains(),
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs
    })


    results = []

    model = Model(hyperParams, details)

    for epoch in range(nOfEpochs):

        if hyperParams.loss == "CircleLoss" or hyperParams.loss == "MultiSimilarityLoss":
            train_losses,train_pos_dist,train_neg_dist,train_pos_loss,train_neg_loss,train_acc = train_Metric(model, train_dataset, epoch)
            print("epoch:", epoch, "train_losses: ", train_losses, "train_pos_dist: ", train_pos_dist, "train_neg_dist: ", train_neg_dist, "train_pos_loss: ", train_pos_loss, "train_neg_loss: ", train_neg_loss, "train_acc: ", train_acc)
            # val_losses, val_pos_dist, val_neg_dist, val_pos_loss, val_neg_loss, val_acc = val_Metric(model, valid_dataset, epoch)
            # print("epoch:", epoch, "val_losses: ", val_losses, "val_pos_dist: ", val_pos_dist,"val_neg_dist: ", val_neg_dist, "val_pos_loss: ", val_pos_loss, "val_neg_loss: ",val_neg_loss, "val_acc: ", val_acc)
        else:
            train_losses, train_pos_dist, train_neg_dist, train_acc = train_Metric(model, train_dataset, epoch)
            print("epoch:", epoch, "train_losses: ", train_losses, "train_pos_dist: ", train_pos_dist, "train_neg_dist: ",train_neg_dist, "train_acc: ",train_acc)
            # val_losses, val_pos_dist, val_neg_dist, val_acc = val_Metric(model, valid_dataset, epoch)
            # print("epoch:", epoch, "val_losses: ", val_losses, "val_pos_dist: ", val_pos_dist, "val_neg_dist: ",val_neg_dist, "val_pos_loss: ", "val_acc: ", val_acc)

        if(analysis and epoch % 25 == 0):
                targetSaveDir = "./Analysis/TargetSavedModels/{}/".format(timespan )
                os.makedirs(targetSaveDir, exist_ok=True)
                torch.save(model, targetSaveDir + "/model_epoch_{}.save".format(epoch))

    return results
