'''
This is the example of using extractive summarization model
'''

from extractive import Summarizer
import argparse
import sys
import pandas as pd
import csv
import requests
import torch
import os
import scipy
import numpy as np


class extractive_summarizer(object):
    def __init__(self, cuda = '0'):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        # print(article)
        self.model = Summarizer(
            model='albert-large-v1',
            hidden= -2,
            reduce_option= 'mean',
            device = device
        )
        print("success load model")

    
    def get_QA_answer_api(self, query):
        url = "http://eez114.ece.ust.hk:5000/query_qa"
        payload = "{\n\t\"text\": \""+query+"\"\n}"
        headers = {
            'Content-Type': "application/json",
            'cache-control': "no-cache",
            'Postman-Token': "696fa512-5fed-45ca-bbe7-b7a1b4d19fe4"
        }
        response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
        response = response.json()
        return response

    def extractive_summary(self, query):
        QA_result = self.get_QA_answer_api(query)
        QA_span = ""
        for i in range(5):
            QA_span =QA_span + " " + QA_result[i]['answer']
        sentences, n_clusters, labels, hidden_args, content, hidden = self.model(QA_span)
        _,_,_,_,_,query_hidden = self.model(query)
        distances = scipy.spatial.distance.cdist(query_hidden, hidden, "cosine")[0].tolist()
        rank = np.argsort(distances)
        extractive_summary = ""
        for i in range(3):
            if len(rank) < 3:
                return ""
            idx = int(np.argwhere(rank == i))
            extractive_summary = extractive_summary + " " + content[idx]
        return extractive_summary.strip()
    
    def choose_QA_result(self, QA_result, query):
        QA_span = ""
        for i in range(5):
            QA_span =QA_span + " " + QA_result[i]['answer']
        sentences, n_clusters, labels, hidden_args, content, hidden = self.model(QA_span)
        _,_,_,_,_,query_hidden = self.model(query)
        distances = scipy.spatial.distance.cdist(query_hidden, hidden, "cosine")[0].tolist()
        rank = np.argsort(distances)
        extractive_summary = ""
        for i in range(3):
            if len(rank) < 3:
                return ""
            idx = int(np.argwhere(rank == i))
            extractive_summary = extractive_summary + " " + content[idx]
        return extractive_summary.strip()
    
summarizar = extractive_summarizer('0')

query = "chloroquine cure covid-19"
QA_result = summarizar.get_QA_answer_api(query)
# extractive_summary = summarizar.extractive_summary(query)
extractive_summary = summarizar.choose_QA_result(QA_result, query)
print(extractive_summary)

