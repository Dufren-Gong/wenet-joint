import os
import shutil
import json
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.search import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring, DecodeResult)
from wenet.utils.mask import make_pad_mask
from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy, count_P_R_F1,
                                reverse_pad_list)
from wenet.utils.context_graph import ContextGraph
from local.me.model.jointBiTransformerDecoder import JointTransformerDecoder
from local.me.model.disf_classification_layer import DisfClassificationLayer

class Joint_Model(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        encoder: BaseEncoder,
        decoder: JointTransformerDecoder,
        ctc: CTC,
        dysf_head: DisfClassificationLayer,
        ctc_weight: float = 0.5,
        dysf_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: Optional[dict] = None,
        apply_non_blank_embedding: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<sos>", vocab_size - 1))
        self.eos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<eos>", vocab_size - 1))
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.dysf_weight = dysf_weight
        self.reverse_weight = reverse_weight
        self.apply_non_blank_embedding = apply_non_blank_embedding

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.dysf_head = dysf_head
        self.dysf_loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_id, reduce='none')
        # self.dysf_loss_function = torch.nn.BCELoss(ignore_index=ignore_id)
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
        temp_writer = None, 
        temp_compare = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        keys = batch['keys']
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        target_dysf = batch['target_dysf'].to(device)
        split_info = batch['split_info']

        if temp_compare != None:
            for key_index, key in enumerate(keys):
                length = text_lengths[key_index]
                temp_dict = dict(key = key,
                                 text = text[key_index][:length].tolist(),
                                 dysf = target_dysf[key_index][:length].tolist(),
                                 split_info = split_info[key_index].tolist())
                write_line = json.dumps(temp_dict, ensure_ascii=False)
                temp_compare.write(write_line + '\n')

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        #获取每个segment的有效chunk数量
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                           text_lengths)
        else:
            loss_ctc, ctc_probs = None, None

        # 2b. Attention-decoder branch
        # use non blank (token level) embedding for decoder
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            #删除ctc判断为blank的chunk，然后重组encoder_out和mask
            encoder_out, encoder_mask = self.filter_blank_embedding(
                ctc_probs, encoder_out)
        if self.ctc_weight != 1.0:
            loss_att, acc_att, concat_out, concat_feature, dysf_out_pad, result_tokens= self._calc_att_loss(
                encoder_out, encoder_mask, text, text_lengths, target_dysf, {
                    "langs": batch["langs"],
                    "tasks": batch["tasks"]
                })
        else:
            loss_att = None
            acc_att = None

        if self.dysf_weight != 0.0:
            dysf_loss, dysf_eval_info, dysf_results = self._calc_dysf_loss(concat_out, concat_feature, dysf_out_pad, split_info)
        else:
            dysf_loss = None
            dysf_eval_info = None

        if temp_writer != None:
            for key_index, key in enumerate(keys):
                length = text_lengths[key_index].item()
                temp_dict = dict(key = key,
                                 text = result_tokens[key_index][:length].tolist(),
                                 dysf = dysf_results[key_index][:length].tolist())
                write_line = json.dumps(temp_dict, ensure_ascii=False)
                temp_writer.write(write_line + '\n')

        if loss_ctc is None:
            if self.dysf_weight == 0:
                loss = loss_att
            else:
                loss = self.dysf_weight * dysf_loss + (1 -
                                                 self.dysf_weight) * loss_att
        elif loss_att is None:
            if self.dysf_weight == 0:
                loss = loss_ctc
            else:
                loss = self.dysf_weight * dysf_loss + (1 -
                                                 self.dysf_weight) * loss_ctc
        else:
            if self.dysf_weight == 0:
                loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
            else:
                loss = self.dysf_weight * dysf_loss + (1 -
                                                 self.dysf_weight) * (self.ctc_weight * loss_ctc + (1 -self.ctc_weight) * loss_att)
        info_dict = {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "th_accuracy": acc_att,
        }
        if dysf_eval_info != None:
            info_dict.update(dysf_eval_info)

        return info_dict

    def _calc_dysf_loss(self, concat_out, concat_feature, dysf_out_pad, split_info):
        dysf_predict = self.dysf_head(concat_out, concat_feature)

        batch_size = concat_out.size(0)
        #同步ASR求loss
        dysf_loss = self.dysf_loss_function(dysf_predict.view(-1, dysf_predict.size(-1)), dysf_out_pad.view(-1)).sum() / batch_size
        predict_target = torch.argmax(dysf_predict, dim=-1)
        count_info = self._accuracy_tagging(predict_target, dysf_out_pad, split_info)
        dysf_eval_info = count_P_R_F1(count_info['predict_dysf_number'], count_info['gold_dysf_number'], count_info['dysf_correct_number'])
        dysf_eval_info.update(count_info)
        return dysf_loss, dysf_eval_info, self.get_predict_result(dysf_predict)

    def _accuracy_tagging(self, predict_result_tagging, gold_result_tagging, split_info, split_only_count_first_flag = True) -> dict:
        assert len(predict_result_tagging) == len(gold_result_tagging)
        gold_number = 0
        predict_number = 0
        correct_number = 0
        #分batch size
        for j in range(0, len(gold_result_tagging)):
            split_info_tmp = split_info[j]
            length = len(split_info_tmp)
            gold_result_tmp = gold_result_tagging[j][0:length]
            predict_result_tmp = predict_result_tagging[j][0:length]
            #计算评估标注的时候对于一个单词分成多个token是否计算多有的token标签还是只计算第一个token的标签
            if split_only_count_first_flag:
                gold_result = []
                predict_result = []
                for k in range(0, len(split_info_tmp)):
                    if split_info_tmp[k] != -1:
                        gold_result.append(gold_result_tmp[k])
                        predict_result.append(predict_result_tmp[k])
            else:
                gold_result = gold_result_tmp.tolist()
                predict_result = predict_result_tmp.tolist()
            #gold中为disf的数量
            gold_number += gold_result.count(1)
            #predict中为disf的数量
            predict_number += predict_result.count(1)
            sum_result = list(map(lambda x: x[0] + x[1], zip(gold_result, predict_result)))
            #disf判断正确的数量
            correct_number += sum_result.count(2)
        count_info = dict(gold_dysf_number = torch.tensor(gold_number),
                  predict_dysf_number = torch.tensor(predict_number),
                  dysf_correct_number = torch.tensor(correct_number))
        return count_info

    @torch.jit.unused
    def _forward_ctc(
            self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                       text_lengths)
        return loss_ctc, ctc_probs

    def filter_blank_embedding(
            self, ctc_probs: torch.Tensor,
            encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        #只返回某个维度最大值的索引
        top1_index = torch.argmax(ctc_probs, dim=2)
        indices = []
        for j in range(batch_size):
            #如果ctc判断不为blank，加入这个chunk
            indices.append(
                torch.tensor(
                    [i for i in range(maxlen) if top1_index[j][i] != 0]))

        select_encoder_out = [
            #从encoder_out按照0维度根据索引提取张量
            torch.index_select(encoder_out[i, :, :], 0,
                               indices[i].to(encoder_out.device))
            for i in range(batch_size)
        ]
        select_encoder_out = pad_sequence(select_encoder_out,
                                          batch_first=True,
                                          padding_value=0).to(
                                              encoder_out.device)
        xs_lens = torch.tensor([len(indices[i]) for i in range(batch_size)
                                ]).to(encoder_out.device)
        T = select_encoder_out.size(1)
        encoder_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        encoder_out = select_encoder_out
        return encoder_out, encoder_mask

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        target_dysf,
        infos: Dict[str, List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #此处设计为让黄金标签也参与到attention结果当中，input pad第一个为eos，意味当判断第一个chunk的decoder结果时候看不到信息。
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        #获取dysf label in, sos/eos视为流畅
        dysf_in_pad, dysf_out_pad = add_sos_eos(target_dysf, 0, 0,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        #获取反向dysf label in
        r_dysf_pad = reverse_pad_list(target_dysf, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        r_dysf_in_pad, _ = add_sos_eos(r_dysf_pad, 0, 0,
                                            self.ignore_id)

        # 1. Forward decoder
        decoder_out, r_decoder_out, l_feature, r_feature, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     dysf_in_pad, r_dysf_in_pad,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        if self.reverse_weight > 0.0:
            #两个decoder结果加权
            concat_out = torch.add(torch.mul(decoder_out, 1 - self.reverse_weight), torch.mul(r_decoder_out, self.reverse_weight))
            concat_feature = torch.add(torch.mul(l_feature, 1 - self.reverse_weight), torch.mul(r_feature, self.reverse_weight))
        else:
            concat_out = decoder_out
            concat_feature = l_feature
        return loss_att, acc_att * 100, concat_out, concat_feature, dysf_out_pad, self.get_predict_result(decoder_out)

    def get_predict_result(self, decoder_out):
        return decoder_out.argmax(dim = -1).squeeze(dim = -1)

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    @torch.jit.unused
    def ctc_logprobs(self,
                     encoder_out: torch.Tensor,
                     blank_penalty: float = 0.0,
                     blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)

        return ctc_probs

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
    ) -> Dict[str, List[DecodeResult]]:
        """ Decode input speech

        Args:
            methods:(List[str]): list of decoding methods to use, which could
                could contain the following decoding methods, please refer paper:
                https://arxiv.org/pdf/2102.01547.pdf
                   * ctc_greedy_search
                   * ctc_prefix_beam_search
                   * atttention
                   * attention_rescoring
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns: dict results of all decoding methods
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
        results = {}
        #只用left decoder
        if 'attention' in methods:
            results['attention'] = attention_beam_search(
                self, encoder_out, encoder_mask, beam_size, length_penalty,
                infos)
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)
        if 'ctc_prefix_beam_search' in methods:
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens,
                                                       beam_size,
                                                       context_graph, blank_id)
            results['ctc_prefix_beam_search'] = ctc_prefix_result
        if 'attention_rescoring' in methods:
            # attention_rescoring depends on ctc_prefix_beam_search nbest
            if 'ctc_prefix_beam_search' in results:
                ctc_prefix_result = results['ctc_prefix_beam_search']
            else:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
            # print(f'tokens: {ctc_prefix_result[0].tokens}')
            # print(f'score: {ctc_prefix_result[0].score}')
            # print(f'times: {ctc_prefix_result[0].times}')
            # print(f'nbest: {ctc_prefix_result[0].nbest}')
            # print(f'nbest_scores: {ctc_prefix_result[0].nbest_scores}')
            # print(f'nbest_times: {ctc_prefix_result[0].nbest_times}')
            # exit(1)
            if self.apply_non_blank_embedding:
                encoder_out, _ = self.filter_blank_embedding(
                    ctc_probs, encoder_out)
            results['attention_rescoring'] = attention_rescoring(
                self, ctc_prefix_result, encoder_out, encoder_lens, ctc_weight,
                reverse_weight, infos)
        return results

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        print(encoder_out.size())
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        print(encoder_out.size())
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        print(encoder_mask.size())
        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        print(index_range)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        print(seq_len_expand)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        #获取指定位置的值，这里相当于倒置
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out
