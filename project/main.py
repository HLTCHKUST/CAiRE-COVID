import os
import sys
import pprint

import json
from retrieval import information_retrieval
from ..src.covidQA import QaModule, print_answers_in_file, rankAnswers, rankAnswersList

pp = pprint.PrettyPrinter(indent=4)

all_results, data_for_qa = information_retrieval("question_generation/task1_question.json")
# tf 1.15.2  ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1586435240", "/home/xuyan/kaggle/bioasq-biobert/model/1586435317"]
# tf 1.13.1  ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1564469515", "/home/xuyan/kaggle/bioasq-biobert/model/1585470591"]
qa_model = QaModule(["mrqa", "biobert"], ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1586435240", "/home/xuyan/kaggle/bioasq-biobert/model/1586435317"], \
    "./mrqa/model/spiece.model", "./biobert/model/bert_config.json", "./biobert/model/vocab.txt")
print("Get Answers...")
answers = qa_model.getAnswers(data_for_qa)
format_answer = qa_model.makeFormatAnswersList(answers)
ranked_answers = rankAnswersList(format_answer)

for ranked_answer in ranked_answers:
    for i, answer in enumerate(ranked_answer):
        pp.pprint(answer)
        input()

        if i>=5:
            print("="*80)
            break

# with open("task1.json", "w") as f:
#     json.dump(ranked_answers, f)

# Final output for synthesis
# List [{
#         "question": "xxxx",
#         "data": 
#         {
#             "answer": ["answer1", "answer2", ...],
#             "confidence": [confidence1, confidence2, ...],
#             "title": [title1, title2, ...],
#             "doi": [doi1, doi2, ...]
#             "sha": [sha1, sha2, ...]
#         }
# }]
