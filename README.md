# MetricBolT
- [Overview](#overview)
- [Python Dependencies](#Python-Dependencies)
- [Dataset](#Dataset)
- [Train and Testing](#Train-and-Testing)
- [Interpretability](#Interpretability)

# Overview

``MetricBolT`` is an innovative deep learning framework designed to extract unique neural fingerprints from fMRI time series data. Leveraging deep metric learning techniques, it learns to map temporal brain activity patterns into a distinctive feature space where representations from the same subject are pulled closer together while those from different subjects are pushed apart. This approach effectively captures the uniqueness of individual brain activity patterns, creating a robust neural fingerprint that can serve as a reliable biomarker for subject identification and neuroimaging.



# Python Dependencies

all packages and their versions used in `MetricBolT` are as follows:

```
aiohappyeyeballs        2.4.0
aiohttp                 3.10.5
aiosignal               1.3.1
annotated-types         0.7.0
asttokens               2.0.5
async-timeout           4.0.3
attrs                   24.2.0
autocommand             2.2.2
backcall                0.2.0
backports.tarfile       1.2.0
bctpy                   0.6.1
Brotli                  1.0.9
certifi                 2024.8.30
charset-normalizer      3.3.2
click                   8.1.7
cloudpickle             3.0.0
colorama                0.4.6
comm                    0.2.1
contourpy               1.1.1
cycler                  0.12.1
debugpy                 1.6.7
decorator               5.1.1
docker-pycreds          0.4.0
einops                  0.8.0
enigmatoolbox           2.0.3
et-xmlfile              1.1.0
eval_type_backport      0.2.0
executing               0.8.3
faiss                   1.8.0
filelock                3.13.1
fonttools               4.53.1
frozenlist              1.4.1
fsspec                  2024.6.1
gitdb                   4.0.11
GitPython               3.1.43
gmpy2                   2.1.2
h5py                    3.11.0
huggingface-hub         0.24.5
idna                    3.7
importlib-metadata      7.0.1
importlib_resources     6.4.3
inflect                 7.3.1
ipykernel               6.28.0
ipython                 8.12.2
jaraco.context          5.3.0
jaraco.functools        4.0.1
jaraco.text             3.12.1
jedi                    0.19.1
Jinja2                  3.1.4
joblib                  1.4.2
jupyter_client          8.6.0
jupyter_core            5.7.2
kiwisolver              1.4.5
llvmlite                0.41.1
lxml                    5.3.0
MarkupSafe              2.1.3
matplotlib              3.7.5
matplotlib-inline       0.1.6
mkl-fft                 1.3.8
mkl-random              1.2.4
mkl-service             2.4.0
more-itertools          10.3.0
mpmath                  1.3.0
msgpack                 1.1.0
multidict               6.1.0
nest-asyncio            1.6.0
netneurotools           0.2.5
networkx                3.1
neuromaps               0.0.5
nibabel                 5.2.1
nilearn                 0.10.4
numba                   0.58.1
numpy                   1.24.3
openpyxl                3.1.5
ordered-set             4.1.0
packaging               24.1
pandas                  2.0.3
parso                   0.8.3
patsy                   0.5.6
pickleshare             0.7.5
pillow                  10.4.0
pip                     24.2
platformdirs            3.10.0
prompt-toolkit          3.0.43
protobuf                5.29.1
psutil                  5.9.0
pure-eval               0.2.2
pydantic                2.10.3
pydantic_core           2.27.1
Pygments                2.15.1
pyparsing               3.1.2
PySocks                 1.7.1
python-dateutil         2.9.0.post0
pytorch-metric-learning 2.6.0
pytz                    2024.1
pywin32                 305.1
PyYAML                  6.0.1
pyzmq                   25.1.2
requests                2.32.3
safetensors             0.4.4
scikit-learn            1.3.2
scipy                   1.10.1
seaborn                 0.13.2
sentry-sdk              2.19.2
setproctitle            1.3.4
setuptools              72.1.0
shap                    0.44.1
six                     1.16.0
slicer                  0.0.7
smmap                   5.0.1
stack-data              0.2.0
statsmodels             0.14.1
sympy                   1.12
threadpoolctl           3.5.0
timm                    1.0.8
tomli                   2.0.1
torch                   2.4.0
torchaudio              2.4.0
torchvision             0.19.0
tornado                 6.4.1
tqdm                    4.66.5
traitlets               5.14.3
typeguard               4.3.0
typing_extensions       4.12.2
tzdata                  2024.1
urllib3                 2.2.2
vtk                     9.1.0
wcwidth                 0.2.5
wheel                   0.43.0
win-inet-pton           1.1.0
wordcloud               1.9.4
wslink                  2.2.1
yarl                    1.12.1
zipp                    3.17.0
```

# Dataset

In this study, we evaluated our method using both the Adolescent Brain Cognitive Development (ABCD) Study dataset and the Human Connectome Project (HCP) dataset. For demonstration purposes, we provide a subset of the ABCD dataset consisting of 16 subjects with fMRI time series data collected across three time points: baseline, second-year follow-up, and fourth-year follow-up. The data can be found in the `MetricBolT/Dataset/Data` directory.To facilitate method evaluation, this subset has been pre-split into training and testing sets, which are available under MetricBolT/Dataset/spilt_subjects/

# Train and Testing

Our model requires at least two time points of fMRI data per subject for both training and testing, with equal time intervals between scans. The model can generalize to new subjects, meaning subjects in the test set need not be present in the training set.

```
python train.py --datasetspan base_four --nOfEpochs 500 --analysis True
```

```
python test.py --datasetspan base_four --epoch 350
```



# Interpretability 

To analyze brain region importance, we need to compute both relevancy maps and token importance scores.

First, extract the importance of BOLD tokens and the CLS tokens, that is, the CLS token and relevancy maps.

```
cd Analysis
python analysis_extractRawData.py --datasetspan base_four --model_epoch 350 --train test 
```

For each time series, the top 5 and bottom 5 features based on importance are selected and saved, along with their corresponding class labels (0 and 1).

```
python impTokenExtractor.py --datasetspan base_four --K 5 
```

The saved features and labels are fed into a random forest model to obtain the importance (contribution scores) of each brain region.

```
python ROIImportancerandomforest.py --datasetspan base_four
```





