'''
    这个文件用于生成脑区重要性
    在impTokenExtractor.py保存每个时间序列最重要的和最不重要的时间序列之后
    训练集进行训练回归
    测试集进行测试
'''

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from brainRegressor import generateImportanceFromCoefs
import matplotlib.pyplot as plt
from brainRegressor import getSubjectwiseAccuracy
from tqdm import tqdm
from scipy import io as scio
import shap
import argparse
# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datasetspan", type=str, default="base_four")

argv = parser.parse_args()
train_path = "{}/Analysis/DataExtracted/{}/TRAIN/".format(os.getcwd(),argv.datasetspan)
subjects = os.listdir(train_path)
train_tokens = []
train_labels = []
train_subjIds = []
j = 0
for sub in tqdm(subjects,desc="Extracting train data:"):
    for i in range(2):
        train_tokens.append(np.concatenate((np.load(train_path+sub+"/timeseries"+str(i+1)+"/top_x_train_static.npy"),np.load(train_path+sub+"/timeseries"+str(i+1)+"/bottom_x_train_static.npy")),axis = 0))
        train_labels.append(np.concatenate((np.load(train_path+sub+"/timeseries"+str(i+1)+"/top_x_train_label.npy"),np.load(train_path+sub+"/timeseries"+str(i+1)+"/bottom_x_train_label.npy")),axis=0))
        train_subjIds.append(np.concatenate((np.load(train_path +sub+"/timeseries"+str(i+1) + "/top_train_static_subjIds.npy"),np.load(train_path +sub+"/timeseries"+str(i+1) + "/bottom_train_static_subjIds.npy")),axis=0))
        # train_tokens.append(np.load(train_path+sub+"/timeseries"+str(i+1)+"/bottom_x_train_static.npy"))
        # train_labels.append(np.load(train_path+sub+"/timeseries"+str(i+1)+"/bottom_x_train_label.npy"))
        # train_subjIds.append(np.load(train_path +sub+"/timeseries"+str(i+1) + "/bottom_train_static_subjIds.npy"))
    # j = j + 1
    # if j == 30:
    #     break

train_tokens = np.array(train_tokens)
train_tokens = np.vstack(train_tokens)
len1 = len(train_labels)
len2 = len(train_labels[0])
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(len1*len2)
train_subjIds = np.array(train_subjIds)
train_subjIds = train_subjIds.reshape(len1*len2)

print("1的数量：{}".format(np.sum(train_labels)))
print("0的数量：{}".format(len1*len2-np.sum(train_labels)))
# 训练和评估随机重要性模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_tokens, train_labels)


# 生成并保存重要性结果
# 每个系数对应一个输入特征 系数的绝对值大小表示该特征对预测结果的重要性。绝对值越大，表示该特征越重要。
# 正系数表示该特征的增加会增加预测为正类的概率，负系数则相反。
# 使用这些系数来确定哪些特征对分类任务最重要
importance = rf.feature_importances_

saveFolderName = "{}/Results/{}".format(os.getcwd(),argv.datasetspan)
print(saveFolderName)
os.makedirs(saveFolderName, exist_ok=True)

np.save(saveFolderName + "/importance.npy", importance )
# 生成并保存重要性权重的折线图
plt.figure()
plt.plot(importance)
plt.savefig(saveFolderName + "/importance.png")
data = {argv.datasetspan:importance}
# scio.savemat("../Simple-Brain-Plot-main/{}.mat".format(argv.datasetspan),data)

test_path = "{}/Analysis/DataExtracted/{}/TEST/".format(os.getcwd(),argv.datasetspan)
subjects = os.listdir(test_path)
test_tokens = []
test_labels = []
test_subjIds = []
for sub in tqdm(subjects,desc="Extracting test data:"):
    for i in range(2):
        test_tokens.append(np.concatenate((np.load(test_path+sub+"/timeseries"+str(i+1)+"/top_x_test_static.npy"),np.load(test_path+sub+"/timeseries"+str(i+1)+"/bottom_x_test_static.npy")),axis = 0))
        test_labels.append(np.concatenate((np.load(test_path+sub+"/timeseries"+str(i+1)+"/top_x_test_label.npy"),np.load(test_path+sub+"/timeseries"+str(i+1)+"/bottom_x_test_label.npy")),axis = 0))
        test_subjIds.append(np.concatenate((np.load(test_path + sub +"/timeseries"+str(i+1)+ "/top_test_static_subjIds.npy"),np.load(test_path + sub +"/timeseries"+str(i+1)+ "/bottom_test_static_subjIds.npy")),axis = 0))
        # test_tokens.append(np.load(test_path+sub+"/timeseries"+str(i+1)+"/bottom_x_test_static.npy"))
        # test_labels.append(np.load(test_path+sub+"/timeseries"+str(i+1)+"/bottom_x_test_label.npy"))
        # test_subjIds.append(np.load(test_path + sub +"/timeseries"+str(i+1)+ "/bottom_test_static_subjIds.npy"))

test_tokens = np.array(test_tokens)
test_tokens = np.vstack(test_tokens)
len1 = len(test_labels)
len2 = len(test_labels[0])
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(len1*len2)
test_subjIds = np.array(test_subjIds)
test_subjIds = test_subjIds.reshape(len1*len2)

# 评估模型（token-wise）：计算模型在训练和测试数据上的准确率，并存储结果。
train_pred = rf.predict(train_tokens)
test_pred = rf.predict(test_tokens)
accuracy_train = accuracy_score(train_labels, train_pred)
accuracy_test = accuracy_score(test_labels, test_pred)
print("训练集准确率（token-wise）：", accuracy_train)
print("测试集准确率（token-wise）：", accuracy_test)
