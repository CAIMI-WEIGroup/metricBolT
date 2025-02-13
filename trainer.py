'''
    这段代码主要是用于执行一个机器学习或深度学习模型在特定数据集上的测试任务。
    通过命令行参数，用户可以选择模型类型、数据集以及其它相关设置（如是否开启分析模式）。
    并针对指定数据集进行多次随机试验（每次试验使用不同的随机种子）。
    最后，它会汇总所有试验结果并计算各项评估指标的均值和标准差，并将结果输出和持久化存储
'''

import Models.BolT.run
print(dir(Models.BolT.run))

import os

import torch

from utils import Option
from Dataset.datasetDetails import datasetDetailsDict
# import model runners
from Models.BolT.run import run_Metric
# import hyper param fetchers
from Models.BolT.hyperparams import getHyper_bolT
import os


import argparse
# parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datasetspan", type=str, default="only_fouryear")
parser.add_argument("--method", type=str, default="Metric")
parser.add_argument("-l", "--loss", type=str, default="TripletMarginLoss")
parser.add_argument("-m", "--model", type=str, default="bolT")
parser.add_argument("-a", "--analysis", type=bool, default=True)
parser.add_argument("-e","--nOfEpochs", type=int, default=500)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")

argv = parser.parse_args()
print("argv.analysis",argv.analysis)
pattern = [["Metric", "TripletMarginLoss"],["Metric", "MultiSimilarityLoss"], ["Metric", "CircleLoss"],["concat", "CrossEntropyLoss"]]
xx = [argv.method,argv.loss]
if [argv.method,argv.loss] not in pattern:
    print("Invalid method")
    exit()


# 创建一个字典，键为模型名，值为对应的超参数获取函数
hyperParamDict = {

        "bolT" : getHyper_bolT,

}

# 创建一个字典，键为模型名，值为对应的模型运行函数
modelDict = {

        "bolT" : run_Metric,
}

# 根据命令行参数中的模型名，获取对应的超参数获取函数和模型运行函数
getHyper = hyperParamDict[argv.model]
runModel = modelDict[argv.model]

timespan = argv.datasetspan


print("\nTrain model is {}".format(argv.model),"\ntrain data is {}".format(timespan),"\nMethod is {}".format(argv.method),"\nLoss is {}".format(argv.loss))


datasetName = "ABCD"
# 从datasetDetailsDict字典中获取对应数据集的详细信息
datasetDetails = datasetDetailsDict[datasetName]

datasetDetails["nOfEpochs"] = argv.nOfEpochs

datasetDetails["method"] = argv.method

datasetDetails["span"] = argv.datasetspan

if argv.method == "Metric":
    datasetDetails["nOfClasses"] = 0
else:
    datasetDetails["nOfClasses"] = 2

# 调用相应模型的超参数获取函数获取超参数设置
hyperParams = getHyper()
hyperParams.loss = argv.loss
hyperParams.method = argv.method

print("Dataset details : {}".format(datasetDetails))
# test


results = run_Metric(hyperParams, Option({**datasetDetails}),device="cuda:{}".format(argv.device) if torch.cuda.is_available() else "cpu", analysis=argv.analysis, timespan = timespan)

print(results)

