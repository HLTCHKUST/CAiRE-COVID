import argparse
import os

from transformers import BertTokenizer
from .s2s_ft.tokenization_unilm import UnilmTokenizer


TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'unilm': UnilmTokenizer,
}

import easydict

def set_config():

    args = easydict.EasyDict({
            "model_type": 'unilm',
            "tokenizer_name": 'unilm1.2-base-uncased',
            "config_path": None,
            "config_path": None,
            "max_seq_length": 512,
            "fp16": True,
            "split": "validation",
            "seed": 123,
            "do_lower_case": True,
            "batch_size": 1,
            "beam_size":5,
            "length_penalty": 0,
            "forbid_duplicate_ngrams": True,
            "forbid_ignore_word": '.',
            "min_len": 50,
            "ngram_size":3,
            "mode": 's2s',
            "max_tgt_length": 48,
            "cache_dir": None,
            "pos_shift": False,
            "need_score_traces": False,
            "model_path": ""
    })

    return args



def set_config1():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type", default='unilm', type=str,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default='./checkpoint/ckpt-32000', type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default='unilm1.2-base-uncased', type=str, 
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', default=True, type=bool,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    # parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="validation",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', type=bool, default=True)
    parser.add_argument('--forbid_ignore_word', type=str, default='.',
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=50, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=48,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")


    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()

    return args