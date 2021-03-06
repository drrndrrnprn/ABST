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

```shell
# download pre-trained BART
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xvzf bart.base.tar.gz
```
### 3. Setup Dataset

```
# 0. concatenate json datum
concat_json.ipynb

# 1. separate json file to train, dev and test
sep_json.ipynb

# 2. extract aos and make text files
sep_json.ipynb

# 3. make various aos files (reversed, reversed first) 
make_aos.ipynb

# 4. analyze triplets in aos file
count.ipynb
```



## Training

<!-- ### Preprocess data

```shell
bash preprocess.sh
``` -->

### Fine-tune mlm BART encoder 

```shell
bash train_bart_e_mlm.sh
```

### Fine-tune denoising BART

```shell
bash train_bart_abst_denoising.sh
```

## Evaluation

```shell
bash inference.sh
```
### BARTABSA

```shell
bash eval.sh
```
