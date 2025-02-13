import torch
import numpy as np
import os
import sys
from scipy import io
import pandas as pd
from tqdm import tqdm


def read_available(path1, path2):
    try:
        return io.loadmat(path1)
    except FileNotFoundError:
        return io.loadmat(path2)

def Metric_abcdLoader(span, train, DynamicLength=375):

    # print("当前工作目录:", os.getcwd())
    # present = os.getcwd()
    # print(os.listdir(present))
    path = "/Dataset/spilt_subjects/"+span+"/"
    if(train == "train"):
        path = path + "train.txt"
    elif(train == "val"):
        path = path + "val.txt"
    else:
        path = path + "test.txt"

    path = os.getcwd()+path
    print("Loading data from path:",path)

    x = []  # 存放每个受试者的ROI时间序列特征
    y = []  # 存放每个受试者的标签（例如：疾病分类）
    subjectIds = []  # 存放每个受试者的唯一标识符

    # 导入数据的路径
    sub_path = "Dataset/Data"

    f = open(path, "r")
    lines = f.readlines()

    for line in tqdm(lines, desc=f'Loading data:'):
        x_ = []
        y_ = []
        subjectIds_ = []
        if("only_baseyear" in path):
            data1 = io.loadmat(sub_path + "/" + line[0:19]+"/noGSR/run01_ses-baselineYear1Arm1_time_series_0.01-0.1_lausanne250.mat")
            data2 = io.loadmat(sub_path + "/" + line[0:19]+"/noGSR/run02_ses-baselineYear1Arm1_time_series_0.01-0.1_lausanne250.mat")
        elif ("only_twoyear" in path):
            data1 = io.loadmat(sub_path + "/" + line[0:19] + "/noGSR/run01_ses-2YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat")
            data2 = io.loadmat(sub_path + "/" + line[0:19] + "/noGSR/run02_ses-2YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat")
        elif ("only_fouryear" in path):
            data1 = io.loadmat(sub_path + "/" + line[0:19] + "/noGSR/run01_ses-4YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat")
            data2 = io.loadmat(sub_path + "/" + line[0:19] + "/noGSR/run02_ses-4YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat")
        elif ("base_two" in path):
            path1_1 = sub_path + "/" + line[0:19] + "/noGSR/run01_ses-baselineYear1Arm1_time_series_0.01-0.1_lausanne250.mat"
            path1_2 = sub_path + "/" + line[0:19] + "/noGSR/run02_ses-baselineYear1Arm1_time_series_0.01-0.1_lausanne250.mat"
            path2_1 = sub_path + "/" + line[0:19] + "/noGSR/run01_ses-2YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            path2_2 = sub_path + "/" + line[0:19] + "/noGSR/run02_ses-2YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            data1 = read_available(path1_1, path1_2)
            data2 = read_available(path2_1, path2_2)
        elif ("two_four" in path):
            path2_1 = sub_path + "/" + line[0:19] + "/noGSR/run01_ses-2YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            path2_2 = sub_path + "/" + line[0:19] + "/noGSR/run02_ses-2YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            path4_1 = sub_path + "/" + line[0:19] + "/noGSR/run01_ses-4YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            path4_2 = sub_path + "/" + line[0:19] + "/noGSR/run02_ses-4YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            data1 = read_available(path2_1, path2_2)
            data2 = read_available(path4_1, path4_2)
        elif ("base_four" in path):
            path1_1 = sub_path + "/" + line[0:19] + "/noGSR/run01_ses-baselineYear1Arm1_time_series_0.01-0.1_lausanne250.mat"
            path1_2 = sub_path + "/" + line[0:19] + "/noGSR/run02_ses-baselineYear1Arm1_time_series_0.01-0.1_lausanne250.mat"
            path4_1 = sub_path + "/" + line[0:19] + "/noGSR/run01_ses-4YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            path4_2 = sub_path + "/" + line[0:19] + "/noGSR/run02_ses-4YearFollowUpYArm1_time_series_0.01-0.1_lausanne250.mat"
            data1 = read_available(path1_1, path1_2)
            data2 = read_available(path4_1, path4_2)

        timeseries1 = data1["averageTimeSeries"]
        timeseries2 = data2["averageTimeSeries"]

        label1 = data1["label"][0][0]
        x_.append(timeseries1)
        y_.append(label1)
        subjectIds_.append(line[0:19])

        label2 = data2["label"][0][0]
        x_.append(timeseries2)
        y_.append(label2)
        subjectIds_.append(line[0:19])

        x.append(x_)
        y.append(y_)
        subjectIds.append(subjectIds_)


    print("Data loaded successfully!")

    return x, y, subjectIds



if __name__ == "__main__":

    # 改变工作目录至上两级
    os.chdir("../../")
    Metric_abcdLoader("base_two","test")