from .abstractive_utils import abstractive_api


def abstractive_api_uni_para(query):
    return abstractive_api(query, 'unilm_para')

def abstractive_api_bart_para(query):
    return abstractive_api(query, 'bart_para')

def abstractive_api_uni_article(query):
    return abstractive_api(query, 'unilm_article')

def abstractive_api_bart_article(query):
    return abstractive_api(query, 'bart_article')
