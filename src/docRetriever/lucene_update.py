import json
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

import numpy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

COVID_INDEX_PARA = './../covid19data/lucene-index-cord19-paragraph-2020-04-24/'

from pyserini.search import pysearch

searcher_para = pysearch.SimpleSearcher(COVID_INDEX_PARA)

def printResults():
    query = "risk factors in environment"
    hits = searcher_para.search(query)
    print(len(hits))
    # Prints the first 10 hits
    for i in range(0, 10):
        print(f'{i+1} {hits[i].docid} {hits[i].score} {hits[i].lucene_document.get("title")} {hits[i].score} {hits[i].lucene_document.get("doi")}')

# printResults()

def get_para_results(query):
    max_articles = 1000
    hits = searcher_para.search(query, max_articles)
    print(len(hits))
    temp = {}
    all_doi = {}
    i = 0
    j = 0
    k = 0
    output = []
    while i<len(hits) and i<max_articles:
        outJson = {}
        outJson['rank'] = j+1
        doi = hits[i].lucene_document.get('doi')
        if '.' in hits[i].docid:
            doc_id = hits[i].docid.split('.')[0]
            para_id = hits[i].docid.split('.')[1]
            paragraph = {}
            paragraph['text'] = hits[i].contents.split('\n')[-1]
            paragraph['id'] = para_id
            all_doi[doi] = True
            if doi not in temp:
                outJson['abstract'] = hits[i].lucene_document.get('abstract')
                article_data = json.loads(searcher_para.doc(doc_id).lucene_document().get('raw'))
                if 'body_text' in article_data:
                    outJson['body_text'] = article_data['body_text']
                temp[doi] = j
            outJson['paragraphs'] = []
            outJson['paragraphs'].append(paragraph)
        else:
            if doi in all_doi:
                i+=1
                continue
            else:
                all_doi[doi] = True
            outJson['abstract'] = hits[i].lucene_document.get('abstract')
        outJson['title'] = hits[i].lucene_document.get('title')
        outJson['sha'] = hits[i].lucene_document.get('sha')
        outJson['doi'] = doi
        outJson['score'] = hits[i].score
        output.append(outJson)
        i+=1
        j+=1
    print(i,j,k)
    return output
