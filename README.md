# CAiRE-COVID
A machine learning-based system that uses state-of-the-art natural language processing (NLP) question answering (QA) techniques combined with summarization for mining the available scientific literature

<img src="img/tensorflow.png" width="12%"> <img src="img/pytorch-logo-dark.png" width="12%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


<img align="right" src="img/HKUST.jpg" width="12%">

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
We can build the lucene index of the COVID-19 dataset from scratch, or get one of the pre-built indexes. Using the paragraph indexing which indexes each paragraph of an article (already uploaded the index as a dataset to use), can be downloaded from: [https://www.dropbox.com/s/xg2b4aapjvmx3ve/lucene-index-cord19-paragraph-2020-04-24.tar.gz](https://www.dropbox.com/s/xg2b4aapjvmx3ve/lucene-index-cord19-paragraph-2020-04-24.tar.gz).
```python
from pyserini.search import pysearch
COVID_INDEX = '../input/luceneindexcovidparagraph20200410/lucene-index-covid-paragraph-2020-04-24'
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
