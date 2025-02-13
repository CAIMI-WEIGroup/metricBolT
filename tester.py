import torch
from tqdm import tqdm
from utils import Option, calculateMetric
from Dataset.dataset import getDataset
from sklearn import metrics as skmetr
import pandas as pd
from scipy import io
import numpy as np
import torch.nn.functional as F
import os
from pytorch_metric_learning import distances
import csv
from Dataset.DataLoaders.abcdLoader import Metric_abcdLoader


def Metric_test(epoch_all, datasetspan):

    x, y, subjectIds = Metric_abcdLoader(datasetspan, "test")

    distance = distances.CosineSimilarity()
  
    epoch_ = [int(num) for num in epoch_all.split(",")]

    for epoch in epoch_:

        acc = 0
        MaxSimilarity = []
        subject_dict = {} #记录识别成功的受试者,成功为1,不成功为0
        subject_dict['src_subject_id'] = []
        subject_dict['identification'] = []

        model = torch.load("./Analysis/TargetSavedModels/" + datasetspan + "/model_epoch_" + str(epoch) + ".save",map_location="cpu")
        model.details.device = "cpu"
        for i in tqdm(range(len(subjectIds)), ncols=120, desc=f'Testing model_epoch_{epoch}'):

            subject_dict['src_subject_id'].append(subjectIds[i][0][4:8]+"_"+subjectIds[i][0][8:19])
            timeseries = x[i][0]
            timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1,keepdims=True)
            xTest = torch.tensor(timeseries.astype(np.float32)).unsqueeze(0)

            xTest_cls = model.metric_test_step(xTest)
            maxsim = -100

            for j in range(len(subjectIds)):
                timeseries = x[j][1]
                timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)
                xTest_2 = torch.tensor(timeseries.astype(np.float32)).unsqueeze(0)
                xTest_2_cls = model.metric_test_step(xTest_2)
                if maxsim < distance(xTest_cls, xTest_2_cls)[0][0]:
                    maxsim = distance(xTest_cls, xTest_2_cls)[0][0]
                    subject = subjectIds[j][1]
                del xTest_2
                del xTest_2_cls
                torch.cuda.empty_cache()
            if subject == subjectIds[i][0]:
                MaxSimilarity.append(maxsim.detach().to("cpu").item())
                acc = acc + 1
                subject_dict['identification'].append(1)
            else:
                subject_dict['identification'].append(0)

        print("\nmodel_epoch_{} Test accuracy : {}".format(epoch,acc/len(subjectIds)))
        # print("\nMax Similarity for everysubject: {}".format(MaxSimilarity))
        # print("\nAverage Max Similarity: {}".format(sum(MaxSimilarity)/len(MaxSimilarity)))
        # print("len(subject_dict[]):",len(subject_dict['src_subject_id']))
        # print("len(subject_dict['identification']):",len(subject_dict['identification']))


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasetspan", type=str, default="only_fouryear")
parser.add_argument("--epoch", type=str, default="350")

argv = parser.parse_args()

print("datasetspan is {}".format(argv.datasetspan))
Metric_test(argv.epoch, argv.datasetspan)
