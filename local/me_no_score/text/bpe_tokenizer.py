from os import PathLike
from typing import Dict, List, Optional, Union
from local.me.text.char_tokenizer import CharTokenizer
from local.me.text.tokenize_utils import tokenize_by_bpe_model


class BpeTokenizer(CharTokenizer):

    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
    ) -> None:
        super().__init__(symbol_table, non_lang_syms, split_with_space,
                         connect_symbol, unk)
        self._model = bpe_model
        # NOTE(Mddct): multiprocessing.Process() issues
        #              don't build sp here
        self.bpe_model = None

    def _build_sp(self):
        if self.bpe_model is None:
            import sentencepiece as spm
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.load(self._model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sp()
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
        return tokens

    def tokens2text(self, tokens: List[str]) -> str:
        self._build_sp()
        text = super().tokens2text(tokens)
        return text.replace("▁", ' ').strip()

    def get_dysf_label(self, line: str, original_dysf_label_str: str) -> List[int]:
        self._build_sp()
        tokens = line.strip().split()
        original_dysf_labels = [int(i) for i in original_dysf_label_str.strip().split()]
        start_index = 0
        dysf_labels = []
        #分词情况
        split_info = []
        for token in tokens:
            this_token_length = len(tokenize_by_bpe_model(self.bpe_model, token))
            if this_token_length == 1:
                dysf_labels.append(original_dysf_labels[start_index])
            else:
                dysf_labels.extend([original_dysf_labels[start_index]] * this_token_length)
            #1 for original, -1 for more split
            split_info.extend([1] + [-1] * (this_token_length - 1))
            start_index += 1
        assert start_index == len(tokens)
        assert len(dysf_labels) == len(split_info)
        return dysf_labels, split_info
