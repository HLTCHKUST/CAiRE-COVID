import os
import sys
import pprint

import json
from .retrieval import information_retrieval
from ..src.covidQA import QaModule, print_answers_in_file, rankAnswers, rankAnswersList

pp = pprint.PrettyPrinter(indent=4)

all_results, data_for_qa = information_retrieval("./dummy_data/task1_question.json")
qa_model = QaModule(["mrqa", "biobert"], ["PATH TO MRQA MODEL", "PATH TO BIOBERT MODEL"], \
    "./MRQA_FOLDER/spiece.model", "./BIOBERT_FOLDER/bert_config.json", "./BIOBERT_FOLDER/vocab.txt")

print("Get Answers...")
answers = qa_model.getAnswers(data_for_qa)
format_answer = qa_model.makeFormatAnswersList(answers)
ranked_answers = rankAnswersList(format_answer)

'''
Final output for synthesis
List [{
        "question": "xxxx",
        "data": 
        {
            "answer": ["answer1", "answer2", ...],
            "confidence": [confidence1, confidence2, ...],
            "title": [title1, title2, ...],
            "doi": [doi1, doi2, ...]
            "sha": [sha1, sha2, ...]
        }
}]
'''