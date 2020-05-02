import requests
import json
import os


def abstractive_api(query, model_name, topk = 3):
    url = "http://eez114.ece.ust.hk:4000/query_summary"

    payload = "{\n\t\"text\": \""+query+"\",\n\t\"model\": \""+model_name+"\"\n}"
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache",
        'Postman-Token': "696fa512-5fed-45ca-bbe7-b7a1b4d19fe4"
    }
    response = requests.request("POST", url, data=payload, headers=headers)

    response = response.json()
    
    return response

def get_ir_result(query, topk = 20):
    post = {}
    post['text'] = query
    r = requests.post('http://hlt027.ece.ust.hk:5000/query_paragraph', json=post)
    meta_info_list = []
    paragraphs_list = []
    meta_info_clean_list = []
    article_clean_list = []

    for i in range(len(r.json())):
        if i > topk - 1:
            break

        meta_info = {}

        # if 'paragraphs' in r.json()[i].keys():
        #     meta_info['paragraphs'] = r.json()[i]['paragraphs'][0]['text']
        if 'abstract' in r.json()[i].keys():
            meta_info['abstract'] = r.json()[i]['abstract']
        if 'doi' in r.json()[i].keys():
            meta_info['doi'] = r.json()[i]['doi']
        if 'title' in r.json()[i].keys():
            meta_info['title'] = r.json()[i]['title']

        paragraphs = []
        if 'body_text' in r.json()[i].keys():
            sections = {}

            for j in range(len(r.json()[i]['body_text'])):
                body_text = r.json()[i]['body_text'][j]
                section_name = body_text['section']
                if section_name in sections.keys():
                    sections[section_name] += ' ' + body_text['text']
                else:
                    sections[section_name] = body_text['text']
            
            for section in sections:
                one_line = {}
                if (section.lower() in ['conclusions', 'conclusion']):
                    one_line["src"] = sections[section]
                    one_line["tgt"] = ""
                    paragraphs.append(one_line)
                    break
                if section.lower().find('figure') != -1:
                    continue
                if section.lower().find('conflicts') != -1:
                    continue
                if section.lower().find('acknowledgments') != -1:
                    continue

                one_line["src"] = sections[section]
                one_line["tgt"] = ""
                paragraphs.append(one_line)
        meta_info_list.append(meta_info)
        paragraphs_list.append(paragraphs)
    
    meta_info_clean_list, article_clean_list = remove_duplicate_article(meta_info_list, paragraphs_list)

    return article_clean_list, meta_info_clean_list

def result_to_json(meta, summary):
    json = meta
    json['summary'] = summary
    return json

def remove_duplicate_article(meta_info_list, article_list):
    meta_info_clean_list = []
    article_clean_list = []
    for i in range(len(meta_info_list)):
        if meta_info_list[i] not in meta_info_list[i+1:]:
            meta_info_clean_list.append(meta_info_list[i])
            article_clean_list.append(article_list[i])
    return meta_info_clean_list, article_clean_list

def get_qa_result(query, topk = 10):
    post = {}
    post['text'] = query
    r = requests.post('http://eez114.ece.ust.hk:5000/query_qa', json=post)
    paragraphs_list = []

    for i in range(len(r.json())):
        if i > topk - 1:
            break

        if 'context' in r.json()[i].keys():
            one_line = {}
            one_line['src'] = r.json()[i]['context']
            one_line['tgt'] = ""
            paragraphs_list.append(one_line)
            
    return paragraphs_list


