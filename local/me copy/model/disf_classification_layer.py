import torch

from wenet.transformer.attention import MultiHeadedCrossAttention

class DisfClassificationLayer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        decoder_output_feature_dim,
        output_dim,
        attention_heads,
        src_attention_dropout_rate: float = 0.0,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        #输出为2的N次方维度
        self.mapping_dim_layer = torch.nn.Linear(vocab_size, decoder_output_feature_dim)
        self.cross_attention_layer = MultiHeadedCrossAttention(
                    attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
        self.classification_layer = torch.nn.Linear(decoder_output_feature_dim, output_dim)

    def forward(self,
                this_text_embedding,
                decoder_feature_out):
        # this_text_embedding_zoom = torch.log_softmax(this_text_embedding, dim=-1)
        # mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
        mapping_text_feature = self.mapping_dim_layer(this_text_embedding)
        #此处未知是否需要设置全注意力mask
        # no_mask = torch.ones_like(this_text_embedding, dtype = torch.bool)
        attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
                                                           mapping_text_feature,
                                                           mapping_text_feature)
        classification_out = self.classification_layer(attention_feature_out)
        #如果使用crossentropy loss则不需要log softmax
        # predict = torch.log_softmax(classification_out, dim=-1)
        return classification_out
    
class DisfClassificationLayer_before(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        decoder_output_feature_dim,
        output_dim,
        attention_heads,
        src_attention_dropout_rate: float = 0.0,
        all_predict_info: bool = True
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        #输出为2的N次方维度
        self.all_predict_info = all_predict_info
        if self.all_predict_info:
            self.mapping_dim_layer = torch.nn.Linear(vocab_size, decoder_output_feature_dim)
        else:
            self.mapping_dim_layer = torch.nn.Embedding(vocab_size, decoder_output_feature_dim)
        self.cross_attention_layer = MultiHeadedCrossAttention(
                    attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
        self.classification_layer = torch.nn.Linear(decoder_output_feature_dim, output_dim)

    def forward(self,
                this_text_embedding,
                decoder_feature_out):
        if not self.all_predict_info:
            this_text_embedding = torch.argmax(this_text_embedding, dim=-1)
        mapping_text_feature = self.mapping_dim_layer(this_text_embedding)
        #此处未知是否需要设置全注意力mask
        # no_mask = torch.ones_like(this_text_embedding, dtype = torch.bool)
        attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
                                                           mapping_text_feature,
                                                           mapping_text_feature)
        classification_out = self.classification_layer(attention_feature_out)
        #如果使用crossentropy loss则不需要log softmax
        # predict = torch.log_softmax(classification_out, dim=-1)
        return classification_out
