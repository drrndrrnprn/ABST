# Aspect Base Sentiment Transfer

## Setup

### 1. Install Dependencies

```shell
# install fairseq
cd ~
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### 2. Download required pre-trained models

download pre-trained BART

```shell
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xvzf bart.base.tar.gz
```
### 3. 

## Training

### Preprocess data

```shell
bash preprocess.sh
```

### Fine-tune mlm BART encoder 

```shell
bash train_bart_e_mlm.sh
```

### Fine-tune denoising BART

```shell
bash train_bart_abst_denoising.sh
```