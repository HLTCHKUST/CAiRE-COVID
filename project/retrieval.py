import json
from ..src import get_para_results

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
            retri_result = get_para_results(query)
            result_item["data"] = retri_result

            qa_item = {"question": query}
            context = []
            titles = []
            doi = []
            count = 1
            for item in retri_result:
                if count>20:
                    break
                if 'abstract' in item and len(item['abstract']) > 0:
                    context.append(item['abstract'])
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1
                if 'paragraphs' in item:
                    context.append(item['paragraphs'][0]['text'])   
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1

            qa_item["data"] = {"answer": "", "context": context, "doi": doi, "titles": titles}

            all_results.append(result_item)
            data_for_qa.append(qa_item)

    return all_results, data_for_qa

if __name__ == "__main__":
    '''
    Here we provide a simple example to show the method to use the Search Engine.
    '''
    query = "How long is the incubation period of covid-19?"
    retri_result = get_para_results(query)
    print(retri_result)