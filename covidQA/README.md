# caireCovid system
<img src="img/tensorflow.png" width="12%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img align="right" src="img/HKUST.jpg" width="15%">
Kaggle system for covid 19

## requirements
<pre>
tensorflow(-gpu)==1.13.1
</pre>

## Install
You can use our package by install with ```pip```
```
pip install caireCovid
```
## Question Answering System
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
