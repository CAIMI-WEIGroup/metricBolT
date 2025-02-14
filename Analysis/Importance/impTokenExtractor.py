'''
    这个文件用于提取每个人的每个时间的前topK个最重要的时间点
    并把这topK个最重要的，时间点的点数、token、相关性得分、标签、受试者id保存起来
    保存在"{}/Analysis/DataExtracted/seed_{}/{}/TRAIN/{}/timeseries{}".format(os.getcwd(),seed, targetDataset, subjId, i+1)
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm

import os

from subjectReader import getSubjects, readSubject


def tokenExtractor(dataset, seed, K, startK, train):
    """
        提取指定数据集中的BOLD token，并根据其相关性进行排序，选择前topK个最重要的token和随机选择的token，分别存储在训练集和测试集中。

        参数：
        dataset: str
            数据集名称。
        seed: int
            随机种子。
        topK: int
            选择的最重要的BOLD token的数量。
        startK: int
            选择的重要token的起始位置，用于偏移。
        """

    targetDataset = dataset

    if train == "train":

        # 获取训练和测试集的受试者目录 返回的是路径列表,['../Data/ABCD/seed_0/TRAIN/subject1', '../Data/ABCD/seed_0/TRAIN/subject2', ...]
        trainSubjectDirs = getSubjects(targetDataset, seed, True)
        print("Extracting train subjects...")

        # 提取训练集受试者的token和相关性得分
        for subjectDir in tqdm(trainSubjectDirs, ncols=60):

            subjId = subjectDir.split("/")[-1]

            for i in range(2):

                # 初始化各种变量，用于存储提取的token、标签和相关性得分
                top_train_static_subjIds = []
                top_x_train_static = []
                top_x_train_static_relevancyScore = []
                top_y_train_static = []
                top_label = []
                bottom_train_static_subjIds = []
                bottom_x_train_static = []
                bottom_x_train_static_relevancyScore = []
                bottom_y_train_static = []
                bottom_label = []



                # 保存训练和测试数据的目录
                saveFolder_train = "{}/Analysis/DataExtracted/seed_{}/{}/{}/timeseries{}".format(os.getcwd(),seed, targetDataset, subjId, i+1)
                # if(os.path.exists(saveFolder_train + "/x_train_static.npy") and os.path.exists(saveFolder_test + "/x_test_static.npy")):
                #     continue
                # 创建保存目录
                os.makedirs(saveFolder_train, exist_ok=True)

                # # 读取受试者数据
                clsRelevancyMap, label, inputTokens = readSubject(subjectDir+ "/timepoint{}".format(i+1))
                # attentionMaps, clsRelevancyMap, label, inputTokens = readSubject(subjectDir+ "/timepoint{}".format(i+1))
                # attentionMap = attentionMaps[-1].mean(axis=1)

                # 计算相关性得分的平均值
                clsRelevancyMap = clsRelevancyMap.mean(axis=0)

                # 选择前topK个最重要的token的索引
                if (startK + K <= clsRelevancyMap.shape[0]):

                    if (startK != 0):
                        # np.argsort 函数返回一个数组，这个数组包含的是对 clsRelevancyMap 进行排序后各元素的索引。
                        # 具体来说，如果 clsRelevancyMap 是一个数组，那么 np.argsort(clsRelevancyMap) 返回的是一个包含 clsRelevancyMap 元素从小到大排序的索引的数组。
                        # -startK：从数组末尾向前数 startK 个位置
                        # -startK - topK：从数组末尾向前数 startK + topK 个位置
                        # 从索引 -startK - topK 到索引 -startK（不包括 -startK 位置）截取子数组
                        top_target_ind_static = np.argsort(clsRelevancyMap)[-startK - K: -startK]
                    else:
                        top_target_ind_static = np.argsort(clsRelevancyMap)[-K:]

                else:

                    top_target_ind_static = np.argsort(clsRelevancyMap)[0:K]
                
                bottom_target_ind_static = np.argsort(clsRelevancyMap)[0:K]

                top_averageRelScore = np.mean(clsRelevancyMap[top_target_ind_static]) / np.min(clsRelevancyMap)
                bottom_averageRelScore = np.mean(clsRelevancyMap[bottom_target_ind_static]) / np.min(clsRelevancyMap)

                # STATIC
                # 提取重要的token
                top_targetTokens = inputTokens[top_target_ind_static]
                bottom_targetTokens = inputTokens[bottom_target_ind_static]

                # 将提取的token、标签和相关性得分加入相应列表
                for token in top_targetTokens:
                    # x_train_static 存放重要的token
                    top_x_train_static.append(token)
                    # y_train_static 存放标签
                    top_y_train_static.append(label)
                    # x_train_static_relevancyScore 存放相关性得分
                    top_x_train_static_relevancyScore.append(top_averageRelScore)
                    # train_static_subjIds 存放受试者ID
                    top_train_static_subjIds.append(subjId)
                    top_label.append(1)

                for token in bottom_targetTokens:
                    # x_train_static 存放重要的token
                    bottom_x_train_static.append(token)
                    # y_train_static 存放标签
                    bottom_y_train_static.append(label)
                    # x_train_static_relevancyScore 存放相关性得分
                    bottom_x_train_static_relevancyScore.append(bottom_averageRelScore)
                    # train_static_subjIds 存放受试者ID
                    bottom_train_static_subjIds.append(subjId)
                    bottom_label.append(0)

                np.save(saveFolder_train + "/top_target_ind_static.npy", top_target_ind_static)
                np.save(saveFolder_train + "/top_x_train_label.npy", top_label)
                np.save(saveFolder_train + "/top_x_train_static.npy", top_x_train_static)
                np.save(saveFolder_train + "/top_x_train_static_relevancyScore.npy", top_x_train_static_relevancyScore)
                np.save(saveFolder_train + "/top_y_train_static.npy", top_y_train_static)
                np.save(saveFolder_train + "/top_train_static_subjIds.npy", top_train_static_subjIds)

                np.save(saveFolder_train + "/bottom_target_ind_static.npy", bottom_target_ind_static)
                np.save(saveFolder_train + "/bottom_x_train_label.npy", bottom_label)
                np.save(saveFolder_train + "/bottom_x_train_static.npy", bottom_x_train_static)
                np.save(saveFolder_train + "/bottom_x_train_static_relevancyScore.npy", bottom_x_train_static_relevancyScore)
                np.save(saveFolder_train + "/bottom_y_train_static.npy", bottom_y_train_static)
                np.save(saveFolder_train + "/bottom_train_static_subjIds.npy", bottom_train_static_subjIds)


    else:

        testSubjectDirs = getSubjects(targetDataset, seed, False)

        print("Extracting test subjects...")
        # 提取测试集受试者的token和相关性得分
        for subjectDir in tqdm(testSubjectDirs, ncols=60):

            subjId = subjectDir.split("/")[-1]

            for i in range(2):

                # 初始化各种变量，用于存储提取的token、标签和相关性得分
                top_test_static_subjIds = []
                top_x_test_static = []
                top_x_test_static_relevancyScore = []
                top_y_test_static = []
                top_label = []
                bottom_test_static_subjIds = []
                bottom_x_test_static = []
                bottom_x_test_static_relevancyScore = []
                bottom_y_test_static = []
                bottom_label = []



                # 保存训练和测试数据的目录
                saveFolder_test = "{}/Analysis/DataExtracted/seed_{}/{}/{}/timeseries{}".format(os.getcwd(),seed, targetDataset, subjId, i+1)
                # if(os.path.exists(saveFolder_train + "/x_train_static.npy") and os.path.exists(saveFolder_test + "/x_test_static.npy")):
                #     continue
                # 创建保存目录
                os.makedirs(saveFolder_test, exist_ok=True)

                # # 读取受试者数据
                clsRelevancyMap, label, inputTokens = readSubject(subjectDir+ "/timepoint{}".format(i+1))
                # attentionMaps, clsRelevancyMap, label, inputTokens = readSubject(subjectDir+ "/timepoint{}".format(i+1))
                # attentionMap = attentionMaps[-1].mean(axis=1)

                # 计算相关性得分的平均值
                clsRelevancyMap = clsRelevancyMap.mean(axis=0)

                # 选择前topK个最重要的token的索引
                if (startK + K <= clsRelevancyMap.shape[0]):

                    if (startK != 0):
                        # np.argsort 函数返回一个数组，这个数组包含的是对 clsRelevancyMap 进行排序后各元素的索引。
                        # 具体来说，如果 clsRelevancyMap 是一个数组，那么 np.argsort(clsRelevancyMap) 返回的是一个包含 clsRelevancyMap 元素从小到大排序的索引的数组。
                        # -startK：从数组末尾向前数 startK 个位置
                        # -startK - topK：从数组末尾向前数 startK + topK 个位置
                        # 从索引 -startK - topK 到索引 -startK（不包括 -startK 位置）截取子数组
                        top_target_ind_static = np.argsort(clsRelevancyMap)[-startK - K: -startK]
                    else:
                        top_target_ind_static = np.argsort(clsRelevancyMap)[-K:]

                else:

                    top_target_ind_static = np.argsort(clsRelevancyMap)[0:K]
                
                bottom_target_ind_static = np.argsort(clsRelevancyMap)[0:K]

                top_averageRelScore = np.mean(clsRelevancyMap[top_target_ind_static]) / np.min(clsRelevancyMap)
                bottom_averageRelScore = np.mean(clsRelevancyMap[bottom_target_ind_static]) / np.min(clsRelevancyMap)

                # STATIC
                # 提取重要的token
                top_targetTokens = inputTokens[top_target_ind_static]
                bottom_targetTokens = inputTokens[bottom_target_ind_static]

                # 将提取的token、标签和相关性得分加入相应列表
                for token in top_targetTokens:
                    # x_train_static 存放重要的token
                    top_x_test_static.append(token)
                    # y_train_static 存放标签
                    top_y_test_static.append(label)
                    # x_train_static_relevancyScore 存放相关性得分
                    top_x_test_static_relevancyScore.append(top_averageRelScore)
                    # train_static_subjIds 存放受试者ID
                    top_test_static_subjIds.append(subjId)
                    top_label.append(1)

                for token in bottom_targetTokens:
                    # x_train_static 存放重要的token
                    bottom_x_test_static.append(token)
                    # y_train_static 存放标签
                    bottom_y_test_static.append(label)
                    # x_train_static_relevancyScore 存放相关性得分
                    bottom_x_test_static_relevancyScore.append(bottom_averageRelScore)
                    # train_static_subjIds 存放受试者ID
                    bottom_test_static_subjIds.append(subjId)
                    bottom_label.append(0)

                np.save(saveFolder_test + "/top_target_ind_static.npy", top_target_ind_static)
                np.save(saveFolder_test + "/top_x_test_label.npy", top_label)
                np.save(saveFolder_test + "/top_x_test_static.npy", top_x_test_static)
                np.save(saveFolder_test + "/top_x_test_static_relevancyScore.npy", top_x_test_static_relevancyScore)
                np.save(saveFolder_test + "/top_y_test_static.npy", top_y_test_static)
                np.save(saveFolder_test + "/top_test_static_subjIds.npy", top_test_static_subjIds)

                np.save(saveFolder_test + "/bottom_target_ind_static.npy", bottom_target_ind_static)
                np.save(saveFolder_test + "/bottom_x_test_label.npy", bottom_label)
                np.save(saveFolder_test + "/bottom_x_test_static.npy", bottom_x_test_static)
                np.save(saveFolder_test + "/bottom_x_test_static_relevancyScore.npy", bottom_x_test_static_relevancyScore)
                np.save(saveFolder_test + "/bottom_y_test_static.npy", bottom_y_test_static)
                np.save(saveFolder_test + "/bottom_test_static_subjIds.npy", bottom_test_static_subjIds)


import argparse
# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-d", "--datasetspan", type=str, default="only_fouryear")
parser.add_argument("-k", "--K", type=int, default=5)
parser.add_argument("--train", type=str, default="test")

argv = parser.parse_args()

tokenExtractor(argv.datasetspan, argv.seed, argv.K, 0, argv.train)