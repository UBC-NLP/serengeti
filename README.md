<p align="center">
    <br>
    <img src="./images/serengeti_logo.png"/>
    <br>
<p>

<p align="center">
<a href="https://huggingface.co/UBC-NLP/serengeti">
        <img alt="Documentation" src="https://img.shields.io/website.svg?down_color=red&down_message=offline&up_message=online&url=https://huggingface.co/UBC-NLP/serengeti">
    </a>
<a href="https://github.com/UBC-NLP/serengeti/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/UBC-NLP/serengeti"></a>
<a href="https://github.com/UBC-NLP/serengeti/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/UBC-NLP/serengeti"></a>

</p>
 
<img src="./images/serengati_languages.jpg" width="55%" height="55%" align="right">
<div style='text-align: justify;'>
Multilingual pretrained language models (mPLMs) acquire valuable, generalizable linguistic information during pretraining and have advanced the state of the art on task-specific finetuning. 
<br><br>
To date, only ~31 out of 2,000 African languages are covered in existing language models. We ameliorate this limitation by developing <b>SERENGETI</b>, a set of massively multilingual language model that covers 517 African languages and language varieties. We evaluate our novel models on eight natural language understanding tasks across 20 datasets, comparing to 4 mPLMs that cover 4-23 African languages. 
<br><br>
<b>SERENGETI</b> outperforms other models on 11 datasets across eights tasks, achieving 82.27 average F<sub>1</sub>-score. We also perform analyses of errors from our models, which allows us to investigate the influence of language genealogy and linguistic similarity when the models are applied under zero-shot settings. We will publicly release our models for research.
</div>

## Table of Contents
- [1 Our Language Models](#1-Our-Language-Models)
  - [1.1 Training Data](#11-training-data)
  - [1.2 Models Architecture](#12-models-architecture)
  - [1.3 Serengeti Models](#13-serengeti-models)
- [2. AfroNLU Benchmark and Evaluation](#2-our-benchmark-AfroNLU)
  - [2.1 Named Entity Recognition](#21-named-entity-recognition)
  - [2.2 Phrase Chuncking](#22-phrase-chuncking)
  - [2.3 Part of Speech Tagging](#23-part-of-speech-tagging)
  - [2.4 News Classification](#24-news-classification)
  - [2.5 Sentiment Analysis](#25-sentiment-analysis)
  - [2.6 Topic Classification](#26-topic-classification)
  - [2.7 Question Answering](#27-question-answering)
  - [2.8 Language Identification](#28-language-identification) 
- [3. How to use AraT5 model](#3-how-to-use-arat5-model)
- [4. Ethics](#4-ethics)
- [5. AraT5 Models Checkpoints](#5-arat5-models-checkpoints)
- [6. Citation](#6-citation)
- [7. Acknowledgments](#7-acknowledgments)

## 1. Our Language Models
## 1.1 Training Data

* **Serengeti Training Data**: SERENGETI is pretrained using 42GB of data comprising a multi-domain, multi-script collection. The multi-domain dataset comprises texts from religious, news, government documents, health documents, and existing corpora written in five scripts from the set {Arabic, Coptic, Ethiopic, Latin, and Vai}. 
* Religious Domain. Our religious data is taken from online Bibles, Qurans, and data crawled from the Jehovah’s witness website. We also include religious texts from the book of Mormon.
* News Domain. We collect data from online newspapers [(Adebara and Abdul-Mageed, 2022)](https://aclanthology.org/2022.acl-long.265/) and news sites such as [(Voice of America)](https://www.voanews.com/navigation/allsites), [(Voice of Nigeria)](https://von.gov.ng/), [(BBC)](https://www.bbc.com/news), [(Global voices)](https://globalvoices.org/), and [(DW)](https://www.dw.com/en/top-stories/s-9097) news sites. We collect local newspapers from 27 languages from across Africa.
* Government Documents. We collect government documents South African Centre for Digital Language Resources [(SADiLaR)](https://www.sadilar.org/), and the Universal Declaration of human rights [(UDHR)](https://www.un.org/en/about-us/universal-declaration-of-human-rights) in multiple languages.
* Health Documents. We collect multiple health documents from the Department of Health, State Government of Victoria, Australia. We collect documents in Amharic, Dinka, Harari, Oromo, Somali, Swahili, and Tigrinya.
* Existing Corpora. We collect corpora available on the web for different African languages, including from Project Gutenberg for Afrikaans, South African News data. for Sepedi and Setswana, OSCAR (Abadji et al., 2021) for Afrikaans, Amharic, Somali, Swahili, Oromo, Malagasy, and Yoruba. We also used Tatoeba for Afrikaans, Amharic, Bemba, Igbo, Kanuri, Kongo, Luganda, Malagasy, Sepedi, Ndebele, Kinyarwanda, Somali, Swahili, Tsonga, Xhosa, Yoruba, and Zulu; Swahili Language Modelling Data for Swahili; Ijdutse corpus for Hausa; Data4Good corpora for Luganda, CC-100 for Amharic, Fulah, Igbo, Yoruba, Hausa, Tswana, Lingala, Luganada, Afrikaans, Somali, Swahili, Swati, North Sotho, Oromo, Wolof, Xhosa, and Zulu; Afriberta-Corpus for Afaan / Oromo, Amharic, Gahuza, Hausa, Igbo, Pidgin, Somali, Swahili, Tigrinya and Yoruba; mC4 for Afrikaans, Amharic, Hausa, Igbo, Malagasy, Chichewa, Shona, Somali, Sepedi, Swahili, Xhosa, Yoruba and Zulu.

## 1.2 Models Architecture

To train our Serengeti, we use the same architecture as ```Electra``` [(Chi etal, 2022)](https://aclanthology.org/2022.acl-long.427/) and  ```XLMR``` [(Conneau etal, 2020)](https://aclanthology.org/2020.acl-main.747/). We experiment with different vocabulary sizes for the Electra models and name them Serengeti-E110 and Serengeti-E250 with 110K and 250K respectively. Each of these models has 12 layers and 12 attention heads. We pretrain each model for 40 epochs with a sequence length of 512, a learning rate of 2e − 4 and a batch size of 216 and 104 for the SERENGETI-E110 and SERENGETI-E250, respectively. We train the XLMR-base model, which we refer to henceforth as Serengeti with a 250K vocabulary size for 20 epochs. This model has 12 layers and 12 attention heads, a sequence length of 512 and a batch size of 8. Serengeti outperforms both Electra models. 

## 1.3 Serengeti Models 
*  **Serengeti-E100**: Electra with 100k vocabulary size
*  **Serengeti-E250**: Electra with 250k vocabulary size, 
*  **Serengeti**:  XLMR-base model.


## 2. AfroNLU Benchmark and Evaluation
AfroNLU is composed of seven different tasks, covering both token and sentence level tasks, across 18 different datasets. The benchmark covers a total of 32 different languages and language varieties. n addition we evaluate our best model (SERENGETI) on an African language identification (LID) task covering all the 517 languages in our pretraining collection. For LID, we use two datasets to test SERENGETI. This puts AfroNLU at a total of 20 different datasets and eight different tasks.
AfroNLU includes the following tasks: ```named entity recognition```,  ```phrase chuncking```,  ```part of speech tagging```, ```news classification```, ```sentiment analysis```,  ```topic classification```,  ```question answering``` and ```language identification```.

### 2.1 
#### 2.1  Named Entity Recognition

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  MasakaNER-v1 [Ifeoluwa Adelani et al., 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00416/107614/MasakhaNER-Named-Entity-Recognition-for-African)                   |81.41±0.26 |78.57±0.53 |84.16±0.45 |81.42±0.30 |81.23±0.32 |81.54±0.68 |**84.53±0.56** |
|  MasakaNER-v2 [Ifeoluwa Adelani et al., 2022](https://aclanthology.org/2022.emnlp-main.298/)    |87.17±0.18 |84.82±0.96  |88.69±0.12 |86.22±0.06  |86.57±0.27 |86.69±0.29 |**88.86±0.25** |      
|  MasakaNER-east    [Ifeoluwa Adelani et al., 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00416/107614/MasakhaNER-Named-Entity-Recognition-for-African)                                     | 80.38±0.56 | 78.33±1.25 |  83.02±0.31 |  79.31±0.92 | 80.53±0.71 | 81.26±0.68 | **83.75±0.26** |       
|  MasakaNER-eastwest  [Ifeoluwa Adelani et al., 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00416/107614/MasakhaNER-Named-Entity-Recognition-for-African) | 82.85±0.38 | 82.37±0.90 | **86.31±0.30**  | 82.98±0.44 |  82.90±0.49 | 83.67±0.44 | 85.94±0.27 |      
|  MasakaNER-west⋆   [Ifeoluwa Adelani et al., 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00416/107614/MasakhaNER-Named-Entity-Recognition-for-African) | 82.85±0.79 | 83.99±0.39  | **86.78±0.44** | 84.08±0.32  | 82.06±0.67 | 83.45±0.81 | 86.27±0.94 |      
|  NCHLT-NER     [(SADiLaR)](https://www.sadilar.org/)| 71.41±0.07 | 70.58±0.26  | 72.27±0.14  | 68.74±0.29  | 64.46±0.37 | 64.42±0.24 | **73.18±0.24** |       
|  Yoruba-Twi-NER  [Alabi et al., 2020](https://aclanthology.org/2020.lrec-1.335/)     | 61.18±2.19 |  70.37±0.61  | 58.48±1.85  | 69.24±3.05 | 61.77±1.24 | 57.99±2.61 | **71.25±1.73** | 
|  WikiAnn  [(Pan et al.2017; Rahimi et al., 2019)](https://aclanthology.org/P19-1015/)     | 83.82±0.39 | 82.65±0.77 | 86.01±0.83  | 83.05±0.20 | 83.17±0.54 | 84.85±0.53 | **85.83±0.94** | 

Metric is F1. 

#### 2.2  Phrase Chuncking

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  Phrase-Chunk [(SADiLaR)](https://www.sadilar.org/)                  | 88.86±0.18 | 88.65±0.06 | 90.12±0.12 | 87.86±0.20 | 90.39±0.21 | 89.93±0.33 | **90.51±0.04** |

Metric is F1. 

#### 2.3  Part of Speech Tagging

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  POS-tagging [(Onyenwe et al., 2018,2019)](https://dl.acm.org/doi/10.1145/3146387)                 | 85.50±0.08 |  85.42±0.13 | 85.39±0.21 | 85.43±0.05 | 85.50±0.16 | **85.61±0.13** | 85.54±0.08 |
     
Metric is F1. 

#### 2.4  News Classification

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  Amharic News [(Azime and Mohammed, 2021)](https://arxiv.org/abs/2103.05639)    | 84.97±0.55 |  59.01±1.47 | 86.18±0.85 | 86.54±1.20 |  86.50±0.71 | 86.34±0.30 | **86.82±0.72** |
|  Kinnews [(Niyongabo et al., 2020)](https://aclanthology.org/2020.coling-main.480/)     | 76.58±0.70 |  77.45±0.43 | 79.13±0.53 | 80.40±1.50 | **81.43±1.02** | 80.38±1.36 | 79.80±0.68 |
|  Kirnews [(Niyongabo et al., 2020)](https://aclanthology.org/2020.coling-main.480/)     | 57.18±3.44 |  74.71±2.56 | 87.67±0.92 | **89.59±0.27** | 78.75±3.24 | 86.60±1.28 |  87.53±2.31 |
|  Swahili News V.0.2 [(David, 2020a,b)](https://dl.acm.org/doi/10.1145/3146387)     | 87.50±0.91 |  85.12±0.93 | 87.49±1.26 | 87.91±0.36 | 87.33±0.28 |  86.12±1.30 |  **88.24±0.99** |

Metric is F1

#### 2.5  Sentiment Analysis

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  Bambara-V1 [(Diallo et al., 2021)](https://arxiv.org/abs/2108.02524)    | 47.17±1.83 |  64.56±1.71 | 59.40±0.56 | 65.06±2.08 |  65.07±2.59 | **65.76±2.02** | 63.36±3.31 |
|  Pidgin Tweet [(Oyewusi et al., 2020)](https://arxiv.org/abs/2003.12450v1)     | 70.42±0.68 |  68.59±0.47 | **71.40±0.51** | 69.19±0.97 | 71.06±0.39 | 70.46±1.02 | 69.74±0.92 |
|  YOSM [(Shode et al., 2022)](https://arxiv.org/abs/2204.09711)     | 85.57±1.09 |  85.25±0.25 | 87.46±0.42 | 88.66±0.23 | 86.86±0.95 | 85.58±1.51 | **87.86±0.81** |

Metric is F1

#### 2.6  Topic Classification

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  Hausa-Topic [(Hedderich et al., 2020)](https://aclanthology.org/2020.emnlp-main.204/)    | 85.80±1.45 |  81.38±0.42 | 88.67±0.30 | **92.59±0.69**|  88.52±1.31 | 89.07±0.95 | 89.93±0.49 |
|  Yoruba-Topic [(Hedderich et al., 2020)](https://aclanthology.org/2020.emnlp-main.204/)   | 54.69±2.89 |  71.79±1.43 | 75.13±1.40 | **81.79±0.66** | 65.22±4.72 | 66.34±4.09 | 79.87±1.61 |     
      
Metric is F1

#### 2.7  Question Answering

| **Dataset**  |  **XLMR** | **mBERT** | **Afro-XLMR** | **AfriBERTa** |  **SERENGETI-E110** | **SERENGETI-E250** |  **SERENGETI** | 
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|
|  QA-Swahili [ (Clark et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00317/96451/TyDi-QA-A-Benchmark-for-Information-Seeking)    | 82.79±1.93 |  **83.40±0.78** | 79.94±0.39 | 57.3±1.8 |  79.76±0.52 | 81.25±1.33 | 80.01±0.78 |

Metric is F1

#### 2.8  Language Identification
We evaluate only Serengeti on the language identification datasets listed below and compare the results with AfroLID:

| **Dataset**  |  **AfroLID** | **Serengeti** | 
|----------------|:---------:|:-------------:|
|  AfroLID [(Adebara et al., 2022b)](https://aclanthology.org/2022.emnlp-main.128.pdf)    | 96.14 | **97.64±0.02** |

| **Dataset**  |  **Split** | **AfroLID** | **Serengeti** | 
|----------------|:---------:|:-------------:|:---------:|
|  AfriSenti [(Muhammad et al., 2022; Yimam et al., 2020)](https://arxiv.org/abs/2201.08277) | Amharic (amh) | 97.00 |  **99.50±0.01** |
|  AfriSenti [(Muhammad et al., 2022; Yimam et al., 2020)](https://arxiv.org/abs/2201.08277) | Hausa (hau)  | 89.00 |  **98.09±0.02** |
|  AfriSenti [(Muhammad et al., 2022; Yimam et al., 2020)](https://arxiv.org/abs/2201.08277) | Igbo  (ibo) | 46.00 | **95.28±0.00** |
|  AfriSenti [(Muhammad et al., 2022; Yimam et al., 2020)](https://arxiv.org/abs/2201.08277) | Nigerian Pidgin (pcm)   | 56.00 |  **77.73±0.01** |
|  AfriSenti [(Muhammad et al., 2022; Yimam et al., 2020)](https://arxiv.org/abs/2201.08277) | Swahili (swh)  | 96.00 |  **98.66±0.02** |
|  AfriSenti [(Muhammad et al., 2022; Yimam et al., 2020)](https://arxiv.org/abs/2201.08277) | Yoruba (yor) | 82.00 |  **98.96±0.00** |
Metric is F1

#  3. How to use AraT5 model

Below is an example for fine-tuning **AraT5-base** for News Title Generation on the Aranews dataset 
``` bash
!python run_trainier_seq2seq_huggingface.py \
        --learning_rate 5e-5 \
        --max_target_length 128 --max_source_length 128 \
        --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
        --model_name_or_path "UBC-NLP/AraT5-base" \
        --output_dir "/content/AraT5_FT_title_generation" --overwrite_output_dir \
        --num_train_epochs 3 \
        --train_file "/content/ARGEn_title_genration_sample_train.tsv" \
        --validation_file "/content/ARGEn_title_genration_sample_valid.tsv" \
        --task "title_generation" --text_column "document" --summary_column "title" \
        --load_best_model_at_end --metric_for_best_model "eval_bleu" --greater_is_better True --evaluation_strategy epoch --logging_strategy epoch --predict_with_generate\
        --do_train --do_eval
```
For the more details about the fine-tuning example, please read this notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/UBC-NLP/araT5/blob/main/examples/Fine_tuning_AraT5.ipynb) 

In addition, we release the fine-tuned checkpoint of the News Title Generation (NGT) which is described in the paper. The model available at Huggingface ([UBC-NLP/AraT5-base-title-generation](https://huggingface.co/UBC-NLP/AraT5-base-title-generation)).

## 4. Ethics

Our models are developed using data from the public domain. 
We provide access to our models to accelerate scientific research with no liability on our part.
Please use our models and benchmark only ethically.
This includes, for example, respect and protection of people's privacy.
We encourage all researchers who decide to use our models to adhere to the highest standards.
For example, if you apply our models on Twitter data, we encourage you to review Twitter policy at [Twitter policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy). For example, Twitter provides the following policy around use of [sensitive information](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases): 

### Sensitive information

You should be careful about using Twitter data to derive or infer potentially sensitive characteristics about Twitter users. Never derive or infer, or store derived or inferred, information about a Twitter user’s:

- Health (including pregnancy)
- Negative financial status or condition
- Political affiliation or beliefs
- Racial or ethnic origin
- Religious or philosophical affiliation or beliefs
- Sex life or sexual orientation
- Trade union membership
- Alleged or actual commission of a crime
- Aggregate analysis of Twitter content that does not store any personal data (for example, user IDs, usernames, and other identifiers) is permitted, provided that the analysis also complies with applicable laws and all parts of the Developer Agreement and Policy.

# 5.  AraT5 Models Checkpoints 

AraT5 Pytorch and Tenserflow checkpoints are available on Huggingface website for direct download and use ```exclusively for research```. `For commercial use, please contact the authors via email @ (*muhammad.mageed[at]ubc[dot]ca*).`

| **Model**   | **Link** | 
|---------|:------------------:|
|  **AraT5-base** |     [https://huggingface.co/UBC-NLP/AraT5-base](https://huggingface.co/UBC-NLP/AraT5-base)       | 
| **AraT5-msa-base**  |     [https://huggingface.co/UBC-NLP/AraT5-msa-base](https://huggingface.co/UBC-NLP/AraT5-msa-base)     |     
| **AraT5-tweet-base**  |   [https://huggingface.co/UBC-NLP/AraT5-tweet-base](https://huggingface.co/UBC-NLP/AraT5-tweet-base)    |      
| **AraT5-msa-small** |     [https://huggingface.co/UBC-NLP/AraT5-msa-small](https://huggingface.co/UBC-NLP/AraT5-msa-small)   |     
| **AraT5-tweet-small**|    [https://huggingface.co/UBC-NLP/AraT5-tweet-small](https://huggingface.co/UBC-NLP/AraT5-tweet-small) |  
| **Title generation model**|    [https://huggingface.co/UBC-NLP/AraT5-base-title-generation](https://huggingface.co/UBC-NLP/AraT5-base-title-generation) | 
|🔥**AraT5v2-base-1024**🔥| [https://huggingface.co/UBC-NLP/AraT5v2-base-1024](https://huggingface.co/UBC-NLP/AraT5v2-base-1024) |


## Citation
If you use the pre-trained model (Serengeti) for your scientific publication, or if you find the resources in this repository useful, please cite our paper as follows (to be updated):
```
@inproceedings{adebara-etal-2022-serengeti,
  title={SERENGETI: Massively Multilingual Language Models for Africa},
  author={Adebara, Ife and Elmadany, AbdelRahim and Abdul-Mageed, Muhammad and Inciarte, Alcides Alcoba},
  journal={arXiv preprint arXiv:2212.10785},
  year={2022}
}

```



## Acknowledgments
We gratefully acknowledges support from Canada Research Chairs (CRC), the Natural Sciences and Engineering Research Council of Canada (NSERC; RGPIN-2018-04267), the Social Sciences and Humanities Research Council of Canada (SSHRC; 435-2018-0576; 895-2020-1004; 895-2021-1008), Canadian Foundation for Innovation (CFI; 37771), [Digital Research Alliance of Canada](https://alliancecan.ca), [UBC ARC-Sockeye](https://arc.ubc.ca/ubc-arc-sockeye), Advanced Micro Devices, Inc. (AMD), and Google. Any opinions, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of CRC, NSERC, SSHRC, CFI, the Alliance, AMD, Google, or UBC ARC-Sockeye.
