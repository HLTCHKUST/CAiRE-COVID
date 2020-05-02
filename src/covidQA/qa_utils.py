import os
import sys
import math
from collections import namedtuple
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag

from .mrqa.predictor_kaggle import mrqa_predictor
from .biobert.predictor_biobert import biobert_predictor

stop_words = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]

class QaModule():
    def __init__(self, model_name, model_path, spiece_model, bert_config, bert_vocab):
        # init QA models
        self.model_name = model_name
        self.model_path = model_path
        self.spiece_model = spiece_model
        self.bert_config = bert_config
        self.bert_vocab = bert_vocab
        self.getPredictors()

    def readIR(self, data):
        synthetic = []

        idx = 0
        for data_item in data:
            question = data_item["question"]
            answer = data_item["data"]["answer"]
            contexts = data_item["data"]["context"]
            dois = data_item["data"]["doi"]
            titles = data_item["data"]["titles"]
            
            for (context, doi, title) in zip(contexts, dois, titles):
                data_sample = {
                    "context": context,
                    "qas": []
                }

                qas_item = {
                    "id": idx,
                    "question": question,
                    "answer": answer,
                    "doi": doi,
                    "title": title,
                }

                data_sample["qas"].append(qas_item)
                synthetic.append(data_sample)

                idx += 1
        return synthetic

    def mrqaPredictor(self, data):
        return mrqa_predictor(self.mrqaFLAGS, self.mrqa_predict_fn, data)
    
    def biobertPredictor(self, data):
        return biobert_predictor(self.bioFLAGS, self.bio_predict_fn, data)

    def getPredictors(self):
        if "mrqa" in self.model_name:
            self.mrqa_predict_fn = self.getPredictor("mrqa")
        if "biobert" in self.model_name:
            self.bio_predict_fn = self.getPredictor("biobert")

    def getPredictor(self, model_name):
        modelpath = self.getModelPath(model_name)
        if model_name == 'mrqa':
            d = {
                "uncased": False,
                "start_n_top": 5,
                "end_n_top": 5,
                "use_tpu": False,
                "train_batch_size": 1,
                "predict_batch_size": 1,
                "shuffle_buffer": 2048,
                "spiece_model_file": self.spiece_model,
                "max_seq_length": 512,
                "doc_stride": 128,
                "max_query_length": 64,
                "n_best_size": 5,
                "max_answer_length": 64,
            }
            self.mrqaFLAGS = namedtuple("FLAGS", d.keys())(*d.values())
            return tf.contrib.predictor.from_saved_model(modelpath)
        elif model_name == 'biobert':
            d = {
                "version_2_with_negative": False,
                "null_score_diff_threshold": 0.0,
                "verbose_logging": False,
                "init_checkpoint": None,
                "do_lower_case": False,
                "bert_config_file": self.bert_config,
                "vocab_file": self.bert_vocab,
                "train_batch_size": 1,
                "predict_batch_size": 1,
                "max_seq_length": 384,
                "doc_stride": 128,
                "max_query_length": 64,
                "n_best_size": 5,
                "max_answer_length": 30,
            }
            self.bioFLAGS = namedtuple("FLAGS", d.keys())(*d.values())
            return tf.contrib.predictor.from_saved_model(modelpath)
        else:
            raise ValueError("invalid model name")
    
    def getModelPath(self, model_name):
        index = self.model_name.index(model_name)
        return self.model_path[index]

    def getAnswers(self, data):
        """
        Output:
            List [{
                "question": "xxxx",
                "data": 
                    {
                        "answer": ["answer1", "answer2", ...],
                        "confidence": [1,2, ...],
                        "context": ["paragraph1", "paragraph2", ...],
                    }
            }]
        """
        answers = []
        qas = self.readIR(data)
        for qa in qas:
            question = qa["qas"][0]["question"]
            if len(answers)==0 or answers[-1]["question"]!=question:
                if len(answers) > 0:
                    scores = answers[-1]["data"]["confidence"]
                    answers[-1]["data"]["confidence"] = self._compute_softmax(scores)

                answer_sample = {}
                answer_sample["question"] = question
                answer_sample["data"] = {
                    "answer": [],
                    "context": [],
                    "title": [],
                    "doi": [],
                    "confidence": [],
                    "raw": [],
                }
                answers.append(answer_sample)

            context = qa["context"]
            doi = qa["qas"][0]["doi"]
            title = qa["qas"][0]["title"] 

            answers[-1]["data"]["context"].append(context)
            answers[-1]["data"]["doi"].append(doi)
            answers[-1]["data"]["title"].append(title)

            sents = sent_tokenize(context)
            spans = self.convert_idx(context, sents)
            
            raw_score_mrqa = 0
            raw_score_bio = 0

            if "mrqa" in self.model_name:
                raw_mrqa = self.mrqaPredictor([qa])
                # get sentence from MRQA
                raw = raw_mrqa[qa["qas"][0]["id"]]   
                raw_answer_mrqa = raw[0]
                raw_score_mrqa = raw[1]

                # question answering one by one
                answer_start = context.find(raw_answer_mrqa, 0)
                answer_end = answer_start + len(raw_answer_mrqa)
                answer_span = []
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)

                y1, y2 = answer_span[0], answer_span[-1]
                if not y1 == y2:
                    # context tokens in index y1 and y2 should be merged together
                    # print("Merge knowledge sentence")
                    answer_sent_mrqa = " ".join(sents[y1:y2+1])
                else:
                    answer_sent_mrqa = sents[y1]
                assert raw_answer_mrqa in answer_sent_mrqa
            else:
                answer_sent_mrqa = ""
            
            
            if "biobert" in self.model_name:
                raw_bio = self.biobertPredictor([qa])
                # get sentence from BioBERT
                raw = raw_bio[qa["qas"][0]["id"]]
                raw_answer_bio = raw[0]
                raw_score_bio = raw[1] 

                if raw_answer_bio == "empty" or "":
                    answer_sent_bio = ""
                    raw_score_bio = 0
                else:
                    # question answering one by one
                    answer_start = context.find(raw_answer_bio, 0)
                    answer_end = answer_start + len(raw_answer_bio)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(idx)

                    y1, y2 = answer_span[0], answer_span[-1]
                    if not y1 == y2:
                        # context tokens in index y1 and y2 should be merged together
                        # print("Merge knowledge sentence")
                        answer_sent_bio = " ".join(sents[y1:y2+1])
                    else:
                        answer_sent_bio = sents[y1]
                    
                    # if raw not in answer_sent_bio:
                    #     print("RAW", raw)
                    #     print("BIO", answer_sent_bio)
                    assert raw_answer_bio in answer_sent_bio
            else:
                answer_sent_bio = ""

            if answer_sent_mrqa == answer_sent_bio or answer_sent_mrqa in answer_sent_bio:
                # print("SAME OR QA < BIO")
                answer_sent = answer_sent_bio
                if raw_score_mrqa < 0 and raw_score_bio < 0:
                    if abs(raw_score_mrqa) < abs(raw_score_bio):
                        score = abs(raw_score_mrqa) * 0.5 + raw_score_bio
                    else:
                        score = raw_score_mrqa + abs(raw_score_bio) * 0.5
                else:
                    score = raw_score_mrqa + raw_score_bio
            elif answer_sent_bio in answer_sent_mrqa:
                # print("BIO < QA")
                answer_sent = answer_sent_mrqa
                if raw_score_mrqa < 0 and raw_score_bio < 0:
                    if abs(raw_score_mrqa) < abs(raw_score_bio):
                        score = abs(raw_score_mrqa) * 0.5 + raw_score_bio
                    else:
                        score = raw_score_mrqa + abs(raw_score_bio) * 0.5
                else:
                    score = raw_score_mrqa + raw_score_bio
            else:
                # print("DIFFERENT ANSWERS")
                answer_sent= " ".join([answer_sent_mrqa, answer_sent_bio])
                score = 0.5 * raw_score_mrqa + 0.5 * raw_score_bio
            
            if raw_answer_mrqa == raw_answer_bio or raw_answer_mrqa in raw_answer_bio:
                # print("SAME OR QA < BIO")
                answer = [raw_answer_bio]
            elif raw_answer_bio in raw_answer_mrqa:
                # print("BIO < QA")
                answer = [answer_sent_mrqa]
            else:
                # print("DIFFERENT ANSWERS")
                answer = [raw_answer_mrqa, raw_answer_bio]
            
            answers[-1]["data"]["answer"].append(answer_sent)
            answers[-1]["data"]["raw"].append(answer)
            answers[-1]["data"]["confidence"].append(score)
        
        # rerank the answers
        score_qa = get_rank_score(answers)
        return score_qa
    
    def _compute_softmax(self, scores):
        """Compute softmax probability over scores."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs
    
    def convert_idx(self, text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

def print_answers_in_file(answers, filepath="./answers.txt"):
    """
        Input:
            List [{
                "question": "xxxx",
                "data": 
                    {
                        "answer": ["answer1", "answer2", ...],
                        "confidence": [1,2, ...],
                        "context": ["paragraph1", "paragraph2", ...],
                    }
            }]
        """
    with open(filepath, "w") as f:
        print("WRITE ANSWERS IN FILES ...")
        for item in answers:
            question = item["question"]
            cas = item["data"]
            for (answer, context) in zip(cas["answer"], cas["context"]):
                f.write("-"*80+"\n")
                f.write("context: "+context+"\n")
                f.write("-"*80+"\n")
                f.write("question: "+question+"\n")
                f.write("-"*80+"\n")
                f.write("answer: "+answer+"\n")
            f.write("="*80+"\n")

def get_rank_score(qa_output):
    for item in qa_output:
        query = item["question"]
        context = item['data']['context']
        item['data']['matching_score'] = []
        item['data']['rerank_score'] = []
        # make new query with only n. and adj.
        tokens = word_tokenize(query.lower())
        tokens = [word for word in tokens if word not in stop_words]
        tagged = pos_tag(tokens)
        query_token = [tag[0] for tag in tagged if 'NN' in tag[1] or 'JJ' in tag[1] or 'VB' in tag[1]]

        for i in range(len(context)):
            text = context[i].lower()
            count = 0
            text_words = word_tokenize(text)
            for word in text_words:
                if word in query_token:
                    count += 1
            
            # matching_score = count/len(text_words)*10 if len(text_words)>50 else count/len(text_words)   # short sentence penalty
            matching_score = count / (1 + math.exp(-len(text_words)+50)) / 5
            item['data']['matching_score'].append(matching_score)
            item['data']['rerank_score'].append(matching_score + item['data']['confidence'][i])
        
        # sort QA results
        c = list(zip(item['data']['rerank_score'], item['data']['context'], item['data']['answer'], item['data']['confidence'], item['data']['doi'], item['data']['title'], item['data']['matching_score'], item['data']['raw']))
        c.sort(reverse = True)
        item['data']['rerank_score'], item['data']['context'], item['data']['answer'], item['data']['confidence'], item['data']['doi'], item['data']['title'], item['data']['matching_score'], item['data']['raw'] = zip(*c)
    return qa_output


    
