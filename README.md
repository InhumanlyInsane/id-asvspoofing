# Indonesian Speech Spoofing
- Vincent Suhardi (2206082505)
- Muh. Kemal Lathif Galih Putra (2206081225)
- Bryan Jeshua (2206027021)

With the advancement of artificial intelligence technology, the ability to produce synthetic voices that are almost indistinguishable from genuine human voices is increasing. This poses serious challenges in detecting spoofing, which refers to **acts of mimicking** or **falsifying voice identity** of someone for specific purposes. In the Indonesian language context, spoofing detection becomes crucial for maintaining security during conversations. Various Neural Network models have been used to detect spoofing, but their effectiveness can vary. Therefore, it is important to conduct a comparative study of different types of Neural Network models to determine the most optimal method for **detecting spoofing in the Indonesian language** and **generating the spoofing dataset itself**. This repository will serve as the collection of papers, journal, and codes that will be used to fulfill our goals.
# If Using Colab:
We can clone the github from https://github.com/clovaai/aasist into the cell in notebook google colab

```
!git clone https://github.com/clovaai/aasist
```
# Getting started
requirements.txt must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible.

# Installing dependencies
```cmd
!pip install -r requirements.txt
```
Our environment (for GPU training)
Based on a docker image: pytorch:1.6.0-cuda10.1-cudnn7-runtime
GPU: 1 NVIDIA Tesla V100
About 16GB is required to train AASIST using a batch size of 24
gpu-driver: 418.67

# Data preparation
We train/validate/evaluate AASIST using the ASVspoof 2019 logical access dataset [4].

```
!python ./download_dataset.py
```
(Alternative) Manual preparation is available via

ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
Download LA.zip and unzip it
Set your dataset directory in the configuration file
Training
The main.py includes train/validation/evaluation.

### To train AASIST [1]:

```
!python main.py --config ./config/AASIST.conf
```

### To train AASIST-L [1]:

```
!python main.py --config ./config/AASIST-L.conf
```
# Training baselines
We additionally enabled the training of RawNet2[2] and RawGAT-ST[3].

### To Train RawNet2 [2]:

```
!python main.py --config ./config/RawNet2_baseline.conf
```

### To train RawGAT-ST [3]:

```
!python main.py --config ./config/RawGATST_baseline.conf
```
Here, we compare the EER metric from each training model across several epochs performed.