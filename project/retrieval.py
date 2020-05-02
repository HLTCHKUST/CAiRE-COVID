
import json
import requests

def retrieve_paragraph(query):
    url = "http://hlt027.ece.ust.hk:5000/query_paragraph"

    payload = "{\n\t\"text\": \""+query+"\"\n}"
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache",
        'Postman-Token': "696fa512-5fed-45ca-bbe7-b7a1b4d19fe4"
    }
    response = requests.request("POST", url, data=payload, headers=headers)

    response = response.json()
    return response


def information_retrieval(file_name):
    """
    Inputs:
        file_name: file name
    Outputs:
        all_results:
            List [{
                "question": "xxxx",
                "data": retri_result
            }]
        data_for_qa:
            List [{
                "question": "xxxx",
                "data": 
                {
                    "answer": "",
                    "context": ['paragraph1', 'paragraph2', ],
                }
            }]
    """
    with open(file_name) as f:
        json_file = json.load(f)
    subtasks = json_file["sub_task"]
    
    all_results = []
    data_for_qa = []
    for item in subtasks:
        questions = item["questions"]
        for query in questions:
            result_item = {"question" : query}
            retri_result = retrieve_paragraph(query)
            result_item["data"] = retri_result

            qa_item = {"question": query}
            context = []
            titles = []
            doi = []
            count = 1
            for item in retri_result:
                #context.append(item["paragraph"] if "paragraph" in item and len(item["paragraph"]) > 0 else item["abstract"])
                if count>20:
                    break
                if 'abstract' in item and len(item['abstract']) > 0:
                    context.append(item['abstract'])
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1
                if 'paragraphs' in item:
                    # for para in item['paragraphs']:
                    #     context.append(para['text'])
                    #     count+=1
                    #     if count>20:
                    #         break
                    context.append(item['paragraphs'][0]['text'])   
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1

            qa_item["data"] = {"answer": "", "context": context, "doi": doi, "titles": titles}

            all_results.append(result_item)
            data_for_qa.append(qa_item)

    return all_results, data_for_qa