import os
import sys

import json
from retrieval import information_retrieval
from caireCovid import QaModule, print_answers_in_file

import requests
import json
import os
import argparse

from abstractive_utils import get_ir_result, result_to_json, get_qa_result
from abstractive_model import abstractive_summary_model
from abstractive_config import set_config
from abstractive_bart_model import *


args = set_config()

def get_summary_list(article_list, abstractive_model):
    summary_list = []
    for i in range(len(article_list)):
        article = article_list[i]
        summary_results = abstractive_model.generate_summary(article)
        result = ""
        for item in summary_results:
            result += item.replace('\n', ' ')
        summary_list.append(result)
    return summary_list

def get_answer_summary(query, abstractive_model):
    paragraphs_list = get_qa_result(query, topk = 3)
    answer_summary_list = abstractive_model.generate_summary(paragraphs_list)
    answer_summary = ""
    for item in answer_summary_list:
        answer_summary += item.replace('\n', ' ')
    answer_summary_json = {}
    answer_summary_json['summary'] = answer_summary
    answer_summary_json['question'] = query
    print("the summary is =====")
    print(answer_summary_json) 
    return answer_summary_json

def get_article_summary(query, abstractive_summary_model):
    article_list, meta_info_list = get_ir_result(query, topk = 1)   # list of list of dictionary {'src':"", 'tgt':""}
    summary_list = get_summary_list(article_list, abstractive_summary_model)
    summary_list_json = []
    
    with open('summary.output', 'w') as fout:
        for i in range(len(summary_list)):
            json_summary = {}
            json_summary = result_to_json(meta_info_list[i], summary_list[i])
            summary_list_json.append(json_summary)
            json.dump(json_summary, fout)
            fout.write('\n')

    return summary_list_json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--query', help='input query', required=False, default='What is the range of incubation periods for COVID-19 in humans?')
    args_query = vars(parser.parse_args())
    
    query = 'What is the range of incubation periods for COVID-19 in humans'
    abstractive_summary_model = abstractive_summary_model(config = args)
    model_path = '/home/sudan/Kaggle/bart/pretrained_model/bart.large.cnn'

    bart_model = Bart_model(model_path)

    # option 1:
    # get a list of summarizations for each article using unilm-fine-tuned model, after the search engine
    # summary_list_json = get_article_summary(query, abstractive_summary_model)

    # Option 2:
    # get a summary from the top 3 paragraphs after the QA model (using unilm-fine-tuned model)
    # answer_summary_json = get_answer_summary(query, abstractive_summary_model)
    # print(answer_summary_json)

    # Option 3:
    # get article summary from bart model
    bart_summary_list_json = get_bart_article_summary(query, bart_model)
    print(bart_summary_list_json)
    # bart_summary_list_json = bart_model.bart_generate_summary(query, bart_model)


    # Option 4: 
    # get a summary from the top 3 paragraphs after the QA model (using Bart model)
    bart_answer_summary_json = get_bart_answer_summary(query, bart_model)
    print(bart_answer_summary_json)