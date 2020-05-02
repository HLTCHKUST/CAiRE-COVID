import json
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

import torch
import numpy
from tqdm import tqdm
# from transformers import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')

COVID_INDEX_PARA = '../../covid19data/lucene-index-covid-paragraph-2020-03-20/'
COVID_INDEX_TITLE = '../../covid19data/lucene-index-covid-2020-03-20/'

from pyserini.search import pysearch

searcher_para = pysearch.SimpleSearcher(COVID_INDEX_PARA)
searcher_title = pysearch.SimpleSearcher(COVID_INDEX_TITLE)

def printResults():
    query = "risk factors in environment"
    #searcher = pysearch.SimpleSearcher('lucene-index-covid-2020-03-20/')
    hits = searcher_para.search(query)
    print(len(hits))
    # Prints the first 10 hits
    for i in range(0, 10):
        print(f'{i+1} {hits[i].docid} {hits[i].score} {hits[i].lucene_document.get("title")} {hits[i].score} {hits[i].lucene_document.get("doi")}')
        #print(len(hits[i].contents.split('\n')))
        #print(json.loads(hits[i].raw).keys())
    #print(hits[0].raw)
    #print(json.loads(hits[0].raw)['paper_id'])

# printResults()

def get_para_results(query):
    hits = searcher_para.search(query,50)
    #print(len(hits))
    temp = {}
    i = 0
    j = 0
    k = 0
    output = []
    while i<len(hits) and i<50 and j<10:
        #print(i,j)
        outJson = {}
        outJson['rank'] = i+1
        #outJson['score'] = hits[i].score
        #print(hits[i].docid)
        if '.' in hits[i].docid:
            doi = hits[i].lucene_document.get('doi')
            paragraph = {}
            paragraph['score'] = hits[i].score
            paragraph['text'] = hits[i].contents.split('\n')[-1]
            #print(paragraph['text'])
            #print()
            if doi in temp:
                if len(output[temp[doi]]['paragraphs']) < 3:
                    output[temp[doi]]['paragraphs'].append(paragraph)
                    #print(paragraph['text'])
                    k+=1
                i+=1
                continue
            else:
                outJson['paragraphs'] = []
                outJson['paragraphs'].append(paragraph)    
                temp[doi] = j
                #print(paragraph['text'])
                k+=1
        else:
            outJson['score'] = hits[i].score
            #print('no para')
        outJson['title'] = hits[i].lucene_document.get('title')
        #print(outJson['title'])
        outJson['abstract'] = hits[i].lucene_document.get('abstract')
        outJson['sha'] = hits[i].lucene_document.get('sha')
        outJson['doi'] = hits[i].lucene_document.get('doi')
        #print(outJson['doi'])
        output.append(outJson)
        #print()
        #print()
        i+=1
        j+=1
    #print(i,j,k)
    return output

# def get_para_results(query):
#     hits = searcher_para.search(query)
#     #print(len(hits))
#     i = 0
#     output = []
#     while i<10:
#         #print(i,j)
#         outJson = {}
#         outJson['rank'] = i+1
#         outJson['score'] = hits[i].score
#         #print(hits[i].docid)
#         if '.' in hits[i].docid:
#             outJson['paragraph']=hits[i].contents.split('\n')[-1]
#         outJson['title'] = hits[i].lucene_document.get('title')
#         outJson['abstract'] = hits[i].lucene_document.get('abstract')
#         outJson['sha'] = hits[i].lucene_document.get('sha')
#         outJson['doi'] = hits[i].lucene_document.get('doi')
#         output.append(outJson)
#         i+=1

#     return output

# output = get_para_results('incubation period of COVID-19 in humans')
# for item in output: 
#     if 'metadata' in item['data']:
#         print(item['data']['paper_id'])
#         print(item['data']['metadata']['title'])
#     else:
#         print(item['data']['title'])
#     if 'paragraphs' in item:
#         print(item['paragraphs'])
#     print()
get_para_results('chlorine covid-19')

def get_results(query):
    hits = searcher_title.search(query)
    output = []
    for i in range(0, 10):
        outJson = {}
        outJson['rank'] = i+1
        outJson['score'] = hits[i].score
        outJson['data'] = json.loads(hits[i].raw)
        output.append(outJson)

    return output