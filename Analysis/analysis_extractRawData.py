'''
    这个文件用于指定数量的受试者的每个时间序列中每个时间点的相关性大小
    并把归一化后的token、标签、相关性图、所有层的注意力图、相对位置偏置表都保存下来
    保存在"./Analysis/Data/{}".format(argv.datasetspan)目录下
'''

import argparse
from genericpath import exists
import sys
import os
from sklearn.manifold import TSNE
import gc
import os

# print("cwd = {}".format(os.getcwd()))

# os.chdir("..")
sys.path.append("./")
print("cwd = {}".format(os.getcwd()))

import numpy as np
import torch
from tqdm import tqdm

from Analysis.relevanceCalculator import generate_relevance
from Dataset.DataLoaders.abcdLoader import Metric_abcdLoader

def extractData(sub_span, num, isTrain, ModelPath, targetDumpFolder_):
    """
        这个函数用于提取指定数据集的数据，包括标签、相关性图、tokens、注意力图和相对位置偏置表。
        sub_path: str, 数据集时间跨度。
        num: int, 提取的数据数量。全部提取占用内存太大
        isTrain: bool, 是否是训练集。
        ModelPath: str, 目标模型文件路径。
        targetDumpFolder: str, 目标保存文件夹路径。
    """
    if isTrain:
        train = "train"
    else:
        train = "test"
    timeseries, labels, subjIds= Metric_abcdLoader(sub_span, train)

    # 取10个人，人太多占用内存过大
    if num == "all":
        num = len(labels)
    else:
        num = int(num)

    for i, subjId in enumerate(tqdm(subjIds, ncols=60)):

        device = torch.device("cuda:{}".format(argv.device))
        # 将数据转换为张量并移动到设备上
        label = torch.tensor(labels[i]).long().to(device)

        for j in range(len(timeseries[i])):

            targetDumpFolder = targetDumpFolder_

            modell = torch.load(ModelPath, map_location="cpu")

            model = modell.model.to("cuda:{}".format(argv.device))

            torch.cuda.empty_cache()

            # 设置模型为评估模式
            model.eval()

            # 规范化时间序列数据
            timeseries_ = timeseries[i][j]
            timeseries_ = torch.tensor(timeseries_).float().to(device)
            timeseries_ = (timeseries_ - timeseries_.mean(dim=1, keepdims=True)) / timeseries_.std(dim=1, keepdims=True)
            timeseries_ = timeseries_[None, :, :]
            # 生成输入token的相关性图
            inputToken_relevances = generate_relevance(model, timeseries_)  # label) # (nW, T)


            if (isTrain):
                targetDumpFolder += "/TRAIN/{}/timepoint{}".format(subjId[j],j+1)
            else:
                targetDumpFolder += "/TEST/{}/timepoint{}".format(subjId[j],j+1)
            os.makedirs(targetDumpFolder, exist_ok=True)

            # 保存标签和相关性图到目标文件夹
            np.save(targetDumpFolder + "/label.npy", label[j].cpu().numpy())
            np.save(targetDumpFolder + "/clsRelevancyMap.npy", inputToken_relevances.detach().cpu().numpy())


            # 保存tokens
            # 首先保存输入本身
            token_0 = timeseries_.detach().cpu().numpy()[0].T
            np.save(targetDumpFolder + "/token_layerIn.npy", token_0)

            # 获取模型中的层数
            layerCount = len(model.blocks)

            # 遍历所有层
            for layer in range(layerCount):
                # 保存每层的token
                token_layer = model.tokens[layer][0].cpu().detach().numpy()
                # 保存每层的注意力图
                attentionMaps = model.blocks[layer].transformer.attention.attentionMaps.cpu().detach().numpy()
                np.save(targetDumpFolder + "/attentionMaps_layer{}.npy".format(layer), attentionMaps)
                # 保存每层的相对位置偏置表
                relative_position_bias_table = model.blocks[
                    layer].transformer.attention.relative_position_bias_table.cpu().detach().numpy()
                np.save(targetDumpFolder + "/relative_position_bias_table_layer{}.npy".format(layer),
                        relative_position_bias_table)

            # clean previous caches values
            for token in model.tokens:
                del token
            del model.tokens
            model.tokens = []
            for k in range(len(model.blocks)):
                # model.blocks[i].transformer.attention.handle.remove()
                # del model.blocks[i].transformer.attention.attentionGradients
                del model.blocks[k].transformer.attention.attentionMaps

            del token_0
            del attentionMaps
            del targetDumpFolder
            del relative_position_bias_table
            del inputToken_relevances
            del model
            torch.cuda.empty_cache()


        if i == num:
            break

    del timeseries
    del label
    torch.cuda.empty_cache()

import argparse
# parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasetspan", type=str, default="only_baseyear")
parser.add_argument("-p", "--model_epoch", type=str, default="400")

parser.add_argument("-c", "--device", type=str, default="0")
parser.add_argument("-t", "--train", type=str, default="test")
parser.add_argument("-n", "--number", type=str, default="all")

print("cwd = {}".format(os.getcwd()))

argv = parser.parse_args()


if argv.train == "train":
    isTrain = True
else:
    isTrain = False
ModelPath = "./Analysis/TargetSavedModels/{}/model_epoch_{}.save".format(argv.datasetspan,argv.model_epoch)
print("ModelPath = {}".format(ModelPath))
targetDumpFolder = "./Analysis/Data/{}".format(argv.datasetspan)
print("targetDumpFolder = {}".format(targetDumpFolder))
extractData(argv.datasetspan, argv.number, isTrain, ModelPath, targetDumpFolder)