import logging

from local.me.text.base_tokenizer import BaseTokenizer
from local.me.text.bpe_tokenizer import BpeTokenizer
from local.me.text.char_tokenizer import CharTokenizer
from wenet.text.paraformer_tokenizer import ParaformerTokenizer
from wenet.text.whisper_tokenizer import WhisperTokenizer

def init_tokenizer(configs) -> BaseTokenizer:
    # TODO(xcsong): Forcefully read the 'tokenizer' attribute.
    tokenizer_type = configs.get("tokenizer", "char")
    if tokenizer_type == "whisper":
        tokenizer = WhisperTokenizer(
            multilingual=configs['tokenizer_conf']['is_multilingual'],
            num_languages=configs['tokenizer_conf']['num_languages'])
    elif tokenizer_type == "char":
        tokenizer = CharTokenizer(
            configs['tokenizer_conf']['symbol_table_path'],
            configs['tokenizer_conf']['non_lang_syms_path'],
            split_with_space=configs['tokenizer_conf'].get(
                'split_with_space', False),
            connect_symbol=configs['tokenizer_conf'].get('connect_symbol', ''))
    elif tokenizer_type == "bpe":
        tokenizer = BpeTokenizer(
            configs['tokenizer_conf']['bpe_path'],
            configs['tokenizer_conf']['symbol_table_path'],
            configs['tokenizer_conf']['non_lang_syms_path'],
            split_with_space=configs['tokenizer_conf'].get(
                'split_with_space', False))
    elif tokenizer_type == 'paraformer':
        tokenizer = ParaformerTokenizer(
            symbol_table=configs['tokenizer_conf']['symbol_table_path'],
            seg_dict=configs['tokenizer_conf']['seg_dict_path'])
    else:
        raise NotImplementedError
    logging.info("use {} tokenizer".format(configs["tokenizer"]))

    return tokenizer
