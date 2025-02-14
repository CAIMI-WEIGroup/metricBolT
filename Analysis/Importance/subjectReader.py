from glob import glob
import numpy as np
import os
# import sys
# sys.path.append("./Analysis")

def getSubjects(dataset, seed, isTrain):

    targetFolder = "{}/Analysis/Data/seed_{}/{}".format(os.getcwd(),seed,dataset)

    if(isTrain):
        targetFolder += "/TRAIN"
    else:
        targetFolder += "/TEST"

    print("targetFolder is {}".format(targetFolder))

    subjects = glob(targetFolder + "/*")

    # 返回的是路径列表,['../Data/ABCD/seed_0/TRAIN/subject1', '../Data/ABCD/seed_0/TRAIN/subject2', ...]
    return subjects

def readSubject(folder):
    # attentionMaps = []
    # for i in range(4):
    #     attentionMaps.append(np.load(folder + "/attentionMaps_layer{}.npy".format(i)))
    clsRelevancyMap = np.load(folder + "/clsRelevancyMap.npy")
    label = np.load(folder + "/label.npy")
    inputTokens = np.load(folder + "/token_layerIn.npy")

    # return attentionMaps, clsRelevancyMap, label, inputTokens
    return clsRelevancyMap, label, inputTokens

