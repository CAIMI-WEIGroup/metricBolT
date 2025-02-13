
from utils import Option

def getHyper_bolT():

    hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : 219, # 应该是N,也就是BOLD token的维度

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 25,
            "shiftCoeff" : 2.0/5.0,  # alpha
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)  fringeCoeff就是beta,2是窗口两边
            "focalRule" : "expand",

            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.1,
            "attnDrop" : 0.1,
            "lambdaCons" : 1,

            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                

    }

    return Option(hyperDict)

