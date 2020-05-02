
import logging
import os
import random
import numpy as np

from .s2s_ft.tokenization_unilm import UnilmTokenizer
from .s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from .s2s_ft.utils import load_and_cache_retrieved_examples
import math
from tqdm import tqdm, trange
from .s2s_ft import s2s_loader as seq2seq_loader
import torch
from transformers import BertTokenizer
from .s2s_ft.tokenization_unilm import UnilmTokenizer



TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'unilm': UnilmTokenizer,
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


class abstractive_summary_model():
    def __init__(self, config):
        self.args = config

        self.model_path = self.args.model_path
        self.model_type = self.args.model_type

        self.tokenizer = TOKENIZER_CLASSES[self.model_type].from_pretrained(
            self.args.tokenizer_name, do_lower_case=self.args.do_lower_case, 
            cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        self.vocab = self.tokenizer.vocab

        self.tokenizer.max_len = self.args.max_seq_length
        self.config_file = self.args.config_path if self.args.config_path else os.path.join(self.args.model_path, "config.json")
        self.config = BertConfig.from_json_file(self.config_file)
        
        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.sep_token])

        forbid_ignore_set = self.get_forbid_ignore_set()

        if self.args.seed > 0:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            random_seed = random.randint(0, 10000)
            logger.info("Set random seed as: {}".format(random_seed))
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.seed)


        self.model = BertForSeq2SeqDecoder.from_pretrained(
            self.model_path, config=self.config, mask_word_id=mask_word_id, search_beam_size=self.args.beam_size,
            length_penalty=self.args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=self.args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=self.args.ngram_size, min_len=self.args.min_len, mode=self.args.mode,
            max_position_embeddings=self.args.max_seq_length, pos_shift=self.args.pos_shift, 
        )

        
    def get_forbid_ignore_set(self):

        forbid_ignore_set = None
        if self.args.forbid_ignore_word:
            w_list = []
        for w in self.args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(self.tokenizer.convert_tokens_to_ids(w_list))
        return forbid_ignore_set

    def bi_uni_pipeline(self):
        bi_uni_pipeline = []
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(self.vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.args.max_seq_length,
            max_tgt_length=self.args.max_tgt_length, pos_shift=self.args.pos_shift,
            source_type_id=self.config.source_type_id, target_type_id=self.config.target_type_id, 
            cls_token=self.tokenizer.cls_token, sep_token=self.tokenizer.sep_token, pad_token=self.tokenizer.pad_token))
        return bi_uni_pipeline

    def generate_summary(self, article):

        self.model.to(device)
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.args.fp16:
            self.model = self.model.half()
    
        torch.cuda.empty_cache()
        self.model.eval()

        bi_uni_pipeline = self.bi_uni_pipeline()
        max_src_length = self.args.max_seq_length - 2 - self.args.max_tgt_length

        to_pred = load_and_cache_retrieved_examples(
                article, self.tokenizer, local_rank=-1, cached_features_file=None, shuffle=False)
        
        input_lines = []
        for line in to_pred:
            input_lines.append(self.tokenizer.convert_ids_to_tokens(line["source_ids"])[:max_src_length])

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / self.args.batch_size)

        next_i = 0
        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + self.args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += self.args.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = self.model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    if self.args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                        if self.args.model_type == "roberta":
                            output_sequence = self.tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence
                        if first_batch or batch_count % 50 == 0:
                            logger.info("{} = {}".format(buf_id[i], output_sequence))
                        if self.args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                pbar.update(1)
                first_batch = False

        output_results = output_lines 

        return output_results       

