import torch
from torch import nn
from local.me.model.focal_loss import FocalLoss
from local.me.model.attention import MultiHeadedCrossAttention

class Multi_Fusion(nn.Module):
    def __init__(
        self,
        output_dim,
        encoder_output_feature_dim,
        attention_heads,
        fusion_attention_heads,
        ignore_id = -1,
        loss_type: str = 'CE',
        src_attention_dropout_rate: float = 0.1
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.loss_type = loss_type
        self.ignore_id = ignore_id
        if loss_type == 'CE':
            self.dysf_loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_id, reduction='none')
        else:
            self.dysf_loss_function = FocalLoss(alpha=[1, 2], gamma=2, ignore_index=ignore_id, reduction='mean')
        #输出为2的N次方维度
        self.text_cross_attention_layer = MultiHeadedCrossAttention(
                    attention_heads, encoder_output_feature_dim, src_attention_dropout_rate)
        self.audio_cross_attention_layer = MultiHeadedCrossAttention(
                    attention_heads, encoder_output_feature_dim, src_attention_dropout_rate)
        self.fusion_cross_attention_layer = MultiHeadedCrossAttention(
                    fusion_attention_heads, encoder_output_feature_dim, src_attention_dropout_rate)
        self.mapping_linner_layer = torch.nn.Linear(encoder_output_feature_dim * 2, encoder_output_feature_dim)
        self.classification_layer = torch.nn.Linear(encoder_output_feature_dim * 2, output_dim)

    def predict(self, text_reps, wav_reps):
        text_attention_feature_out, _ = self.text_cross_attention_layer(wav_reps,
                                                           text_reps,
                                                           text_reps)
        audio_attention_feature_out, _ = self.audio_cross_attention_layer(text_reps,
                                                           wav_reps,
                                                           wav_reps)
        fusion_attention_feature_out, _ = self.fusion_cross_attention_layer(text_reps,
                                                                        text_attention_feature_out,
                                                                        text_attention_feature_out)
        concat_feature = torch.sigmoid(self.mapping_linner_layer(torch.cat((fusion_attention_feature_out, audio_attention_feature_out), dim=-1))) * fusion_attention_feature_out
        final_featue = torch.cat((fusion_attention_feature_out, concat_feature), dim=-1)
        classification_out = self.classification_layer(final_featue)    
        return classification_out

    def forward(self,
                batch,
                device):
        text_reps = batch['text_rep'].to(device)
        wav_reps = batch['wav_rep'].to(device)
        target_dysf = batch['target_dysf'].to(device)
        split_info = batch['split_info']
        # this_text_embedding_zoom = torch.log_softmax(this_text_embedding, dim=-1)
        # mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
        classification_out = self.predict(text_reps, wav_reps)
        dysf_loss = self.dysf_loss_function(classification_out.view(-1, classification_out.size(-1)), target_dysf.view(-1))
        if self.loss_type == 'CE':
            dysf_loss = dysf_loss.sum() / (torch.sum(target_dysf != self.ignore_id))
        count_info = self._accuracy_tagging(torch.argmax(classification_out, dim=-1), target_dysf, split_info)
        dysf_eval_info = self.count_P_R_F1(count_info['predict_dysf_number'], count_info['gold_dysf_number'], count_info['dysf_correct_number'])
        info_dict = {
            'loss': dysf_loss,
        }
        info_dict.update(dysf_eval_info)
        info_dict.update(count_info)
        return info_dict
    
    def cv(self,
            batch,
            device):
        text_reps = batch['text_rep'].to(device)
        wav_reps = batch['wav_rep'].to(device)
        target_dysf = batch['target_dysf'].to(device)
        split_info = batch['split_info']
        # this_text_embedding_zoom = torch.log_softmax(this_text_embedding, dim=-1)
        # mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
        classification_out = self.predict(text_reps, wav_reps)
        results, scores = self.get_predict_result_and_scores(classification_out, split_info)
        count_info = self._accuracy_tagging(torch.argmax(classification_out, dim=-1), target_dysf, split_info)
        return results, scores, count_info

    def get_predict_result_and_scores(self, decoder_out, split_info):
        scores = []
        results = []
        results_before =  decoder_out.argmax(dim = -1).squeeze(dim = -1)
        decoder_out = torch.nn.functional.softmax(decoder_out, dim=-1)
        for sample_index, sample_split_info in enumerate(split_info):
            sample_scores = []
            sample_results = []
            decoder_temp = decoder_out[sample_index][1:].tolist()
            result_temp = results_before[sample_index][1:].tolist()
            for info_index, split_flag in enumerate(sample_split_info):
                if split_flag != -1:
                    this_result = result_temp[info_index]
                    sample_results.append(this_result)
                    this_score = decoder_temp[info_index][this_result]
                    sample_scores.append(this_score)
            scores.append(sample_scores)
            results.append(sample_results)
        return results, scores
    
    def _accuracy_tagging(self, predict_result_tagging, gold_result_tagging, split_info, split_only_count_first_flag = True) -> dict:
        assert len(predict_result_tagging) == len(gold_result_tagging)
        gold_number = 0
        predict_number = 0
        correct_number = 0
        #分batch size
        for j in range(0, len(gold_result_tagging)):
            split_info_tmp = split_info[j]
            length = len(split_info_tmp)
            gold_result_tmp = gold_result_tagging[j][1:-1][0:length]
            predict_result_tmp = predict_result_tagging[j][1:-1][0:length]
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
    
    def count_P_R_F1(self, predict_dysf_number,
                      gold_dysf_number,
                      dysf_correct_number):
        if isinstance(predict_dysf_number, torch.Tensor):
            predict_dysf_number = predict_dysf_number.item()
        if isinstance(gold_dysf_number, torch.Tensor):
            gold_dysf_number = gold_dysf_number.item()
        if isinstance(dysf_correct_number, torch.Tensor):
            dysf_correct_number = dysf_correct_number.item()
        try:
            p_score = dysf_correct_number * 1.0 / predict_dysf_number
            r_score = dysf_correct_number * 1.0 / gold_dysf_number
            f_score = 2.0 * p_score * r_score / (p_score + r_score)
        except:
            p_score = 0
            r_score = 0
            f_score = 0
        info_dict = dict(p_score = p_score * 100,
                         r_score = r_score * 100,
                         f_score = f_score * 100)
        return info_dict