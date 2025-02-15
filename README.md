# Text-Style-Transfer with CycleGAN: Decoder-only Models and Italian Datasets
This repository contains the code for the final project of the Deep Natural Language Processing course for the 2024/2025 academic year at Politecnico di Torino.

This work is based on the the paper [Self-supervised Text Style Transfer using Cycle-Consistent Adversarial Networks](https://dl.acm.org/doi/10.1145/3678179), published in ACM Transactions on Intelligent Systems and Technology.

It includes the Python package to train and test the CycleGAN architecture for Text Style Transfer described in the paper and all the extensions presented in this project.

## Installation
The following command will clone the project:
```
git clone https://github.com/danielercole/DNLP_project.git
```

To install the required libraries and dependencies, you can refer to the `env.yml` file.

Before experimenting, you can create a virtual environment for the project using Conda.
```
conda create -f env.yml -n cyclegan_tst 
conda activate cyclegan_tst
```

The installation should also cover all the dependencies..


## Data
### Verse to prose
We take the original text of the *Divina Commedia* from [Wikisource](https://it.wikisource.org/wiki/Divina_Commedia)
and the corresponding prose interpretation from this [website](https://www.orlandofurioso.com/parafrasi-dei-canti-dellinferno-prima-cantica-del-poema-divina-commedia/). The final dataset can be found in `data/dante` and the files name follow the rules of the original work, so are of the form `[train|dev|test].[0|1].txt`, where 0 is for the original verses and 1 is for the prose interpretation.

### Formality transfer
According to the dataset license, you can request access to the [XFORMAL](https://arxiv.org/abs/2104.04108) dataset following the steps described in its official [repository](https://github.com/Elbria/xformal-FoST).


Once you have gained access, put it into the `family_relationships` directory for the *Family & Relationships* domains, under the `data/XFORMAL` folder. Please name the files as `[train|dev|test].[0|1].txt`, where 0 is for informal style and 1 is for formal style.


### Sentiment transfer
We use the [Yelp](https://papers.nips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html) dataset following the same splits as in [Li et al.](https://aclanthology.org/N18-1169/) available in the official [repository](https://github.com/lijuncen/Sentiment-and-Style-Transfer). Put it into the `data/yelp` folder and please name the files as `[train|dev|test].[0|1].txt`, where 0 is for negative sentiment and 1 is for positive sentiment.

## Training and testing
You can train and test all the extensions to the orginal work simply running the notebooks presented in the `notebooks` directory.
Every jupiter notebook is though to be runned on Google Colab and can be easily customized to change the training setup. For the full explaination of the several command line arguments link here the GitHub folder of the [original project](https://github.com/gallipoligiuseppe/TST-CycleGAN/tree/main).

Note: all the files the end with *DO* are related to the extension in which we substitute the generator with a Decoder-Only model like GPT2-Instruct. All the files that end with *CTRL*, instead, are related to the Text Style Transfer done with Salesforce’s CTRL.


## Authors
Alessandro Arneodo, 
[Daniele Ercole](https://github.com/danielercole), 
[Davide Fassio](https://github.com/Davidefassio), 
Alessia Manni