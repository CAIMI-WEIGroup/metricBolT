'''
    定义了一个名为SupervisedDataset的类，
    它是专门为神经网络模型训练设计的一个自定义数据集类。
    SupervisedDataset类主要用于处理有监督学习任务的数据加载和预处理过程
    支持交叉验证拆分及动态采样长度
    根据提供的数据集名字从loaderMapper中找到相应的加载器函数来加载数据（特征、标签和受试者ID）
'''
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .DataLoaders.abcdLoader import Metric_abcdLoader

# 定义一个字典，用于映射不同的数据集名到对应的加载器函数
loaderMapper = {
    "ABCDMetric" : Metric_abcdLoader,
}


# 定义一个用于获取数据集实例的函数
def getDataset(options,path):
    return SupervisedDataset(options,path)

# 定义一个继承自torch.utils.data.Dataset的监督学习数据集类
class SupervisedDataset(Dataset):
    
    def __init__(self, datasetDetails, train="train"):

        # 设置数据集的相关参数
        self.batchSize = datasetDetails.batchSize
        # 动态采样长度
        self.dynamicLength = datasetDetails.dynamicLength

        # 根据数据集名从loaderMapper字典中获取对应的加载器函数
        loader = loaderMapper[datasetDetails.datasetName+datasetDetails.method]

        self.k = None

        # 调用加载器函数加载数据集，获取特征数据、标签数据和受试者ID
        self.data, self.labels, self.subjectIds = loader(datasetDetails.span, train, self.dynamicLength)

        # 初始化目标数据、目标标签、目标受试者ID变量为空
        self.targetData = None
        self.targetLabel = None
        self.targetSubjIds = None

        # 初始化随机范围变量
        self.randomRanges = None

        # 初始化训练和测试索引集合为空
        self.trainIdx = None
        self.testIdx = None

    # 重写父类__len__方法，返回数据集大小
    def __len__(self):
        return len(self.data) if isinstance(self.targetData, type(None)) else len(self.targetData)

    def get_nOfTrains(self):
        return len(self.data)

    # 设置当前工作折叠以及是否为训练阶段
    def setIdx(self, train=True):

        self.train = train

        if not train:
            testIdx = list(range(len(self.data)))
            self.testIdx = testIdx
        else: 
            trainIdx = list(range(len(self.data)))
            self.trainIdx = trainIdx


        # 根据当前阶段（训练或测试）更新目标数据、目标标签和目标受试者ID
        self.targetData = [self.data[idx] for idx in trainIdx] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [self.labels[idx] for idx in trainIdx] if train else [self.labels[idx] for idx in testIdx]
        self.targetSubjIds = [self.subjectIds[idx] for idx in trainIdx] if train else [self.subjectIds[idx] for idx in testIdx]


    def getIdx(self, train=True):

        self.setIdx(train)

        return DataLoader(self, batch_size=self.batchSize, shuffle=True, drop_last=True)

    # 重写父类__getitem__方法，返回处理后的数据样本
    def __getitem__(self, idx):

        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]

        subjects = []

        # 对时序数据进行归一化处理
        # normalize timeseries
        for timeseries in subject:
            timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1,
                                                                                            keepdims=True)
            subjects.append(timeseries)

        subject = subjects
        subjects = []

        # 返回经过处理后的数据样本
        return {"timeseries": subject, "label": label, "subjId": subjId}





