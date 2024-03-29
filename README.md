# CAiRE-COVID
A machine learning-based system that uses state-of-the-art natural language processing (NLP) question answering (QA) techniques combined with summarization for mining the available scientific literature

<img src="img/tensorflow.png" width="12%"> <img src="img/pytorch-logo-dark.png" width="12%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


<img align="right" src="img/HKUST.jpg" width="12%">

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{su2020caire,
  title={CAiRE-COVID: A Question Answering and Query-focused Multi-Document Summarization System for COVID-19 Scholarly Information Management},
  author={Su, Dan and Xu, Yan and Yu, Tiezheng and Siddique, Farhad Bin and Barezi, Elham and Fung, Pascale},
  booktitle={Proceedings of the 1st Workshop on NLP for COVID-19 (Part 2) at EMNLP 2020},
  year={2020}
}
</pre>

## Abstract

We present CAiRE-COVID, a real-time question answering (QA) and multi-document summarization system, which won one of the 10 tasks in the Kaggle COVID-19 Open Research Dataset Challenge, judged by medical experts. Our system aims to tackle the recent challenge of mining the numerous scientific articles being published on COVID-19 by answering high priority questions from the community and summarizing salient question-related information. It combines information extraction with state-of-the-art QA and query-focused multi-document summarization techniques, selecting and highlighting evidence snippets from existing literature given a query. We also propose query-focused abstractive and extractive multi-document summarization methods, to provide more relevant information related to the question. We further conduct quantitative experiments that show consistent improvements on various metrics for each module. We have launched our website CAiRE-COVID for broader use by the medical community, and have open-sourced the code for our system, to bootstrap further study by other researches.

## System Online!
Currently the CAiRE-COVID system has already been launched online. Please access the system by [http://caire.ust.hk/covid](http://caire.ust.hk/covid).
## Kaggle CORD-19 Task Winner
We are honored to be informed that our submission has won as the best response for the task [What has been published about information sharing and inter-sectoral collaboration?](https://www.kaggle.com/sudansudan/caire-cord-task10)

## Install
1. You can install the requirements by:
```
pip install -r requirements.txt
```
2. In addition, you also need to install [pytorch](https://pytorch.org/).

## System Modules Usage
If you are interested in trying out the system modules yourself, you can utilize the system module by the following methods:
### Document Retriever
**1. Query Paraphrasing**  
For this part, you can implement your own methods or skip this step if your queries are relatively short and simple or you don't persuit SOTA performance. 
**2. Search Engine** 
**2.1 install Python dependencies and pre-built index**  
Following the lucene+answerini information retrieval as described in: [https://github.com/castorini/anserini/blob/master/docs/experiments-covid.md](https://github.com/castorini/anserini/blob/master/docs/experiments-covid.md), set up JAVA sdk 11 first:
```
curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz
update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1
update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java
```
```python
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"
```

**2.2 Get the pyserini library, which is anserini wrapped with python:**
```
pip install pyserini==0.8.1.0
```
We can build the lucene index of the COVID-19 dataset from scratch, or get one of the pre-built indexes. Using the paragraph indexing which indexes each paragraph of an article (already uploaded the index as a dataset to use), can be downloaded from: [link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dsu_connect_ust_hk/EXGGMqssOiJEjAi8BYGMmHwBHBewM5V38-A41Qw7tBbn8Q).

```python
from pyserini.search import pysearch
COVID_INDEX = 'the directory name of the index you downloaded from the above link'
```
The indexing is done based on each paragraph merged with the title and abstract. Given an article with id doc_id, the index will be as follows:
+ doc_id : title + abstract
+ doc_id.00001 : title + abstract + 1st paragraph
+ docid.00002: title + abstract + 2nd paragraph
+ docid.00003: title + abstract + 3rd paragraph

**2.3 Try the example!**  
```
python project/retrieval.py
```

### Relevent Snippet Selection
You can use our package by install with ```pip``` or use the source code.
```
pip install caireCovid
```
#### Question Answering System
In this system, we build QA modules by a ensemble of two QA models, which are [BioBERT](https://github.com/dmis-lab/bioasq-biobert) model which fine-tuned on SQuAD, and MRQA model which is our submission to MRQA@EMNLP 2019. 

The MRQA model and the exported BioBERT model that are utilized in this project can bo downloaded by this [link](https://drive.google.com/drive/folders/1yjzYN_KCz8uLobqaUddftBGPAZ6uSDDj?usp=sharing).

If you want to use our MRQA model in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{su2019generalizing,
  title={Generalizing Question Answering System with Pre-trained Language Model Fine-tuning},
  author={Su, Dan and Xu, Yan and Winata, Genta Indra and Xu, Peng and Kim, Hyeondey and Liu, Zihan and Fung, Pascale},
  booktitle={Proceedings of the 2nd Workshop on Machine Reading for Question Answering},
  pages={203--211},
  year={2019}
}
</pre>

We provide the example script, while you need to change the paths to the QA models in ```project/qa.py```. Note that the final output is already re-ranked based on re-ranking score.
```
python project/qa.py
```

#### Hightlighting
Keyword highlighting is mainly implemented by term matching, of which the code could be found in ```src/covidQA/highlights.py```.

### Summarization
You can use our package by install with ```pip``` or use the source code.
```
pip install covidSumm
```

We provide the example scripts for both abstractive and extractive summarization.
```
python project/abstractive_summarization.py
```
```
python project/extractive_summarization.py
```
