import torch

from wenet.transformer.attention import MultiHeadedCrossAttention, MultiHeadedAttention

# #new2_sig_or_cross
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True,
#         embedding_shape: int = 128
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, embedding_shape)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + embedding_shape, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             score = torch.sigmoid(extra)
#             attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
#                                                             score,
#                                                             score)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

class DisfClassificationLayer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        decoder_output_feature_dim,
        output_dim,
        attention_heads,
        src_attention_dropout_rate: float = 0.0,
        use_extract_dysf_layer: bool = True,
        embedding_shape: int = 128
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        #输出为2的N次方维度
        self.use_extract_dysf_layer = use_extract_dysf_layer
        self.mapping_dim_layer = torch.nn.Embedding(vocab_size, embedding_shape)
        if use_extract_dysf_layer:
            self.audio_extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
            self.attention_layer = MultiHeadedAttention(
                        attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
            self.cross_attention_layer = MultiHeadedCrossAttention(
                        attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
        self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + embedding_shape, output_dim)

    def forward(self,
                encoder_out,
                encoder_mask,
                this_text_embedding,
                decoder_feature_out):
        this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
        mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
        if self.use_extract_dysf_layer:
            attention_feature_out, _ = self.attention_layer(decoder_feature_out,
                                                            decoder_feature_out,
                                                            decoder_feature_out)
            audio_extra = self.audio_extract_liner(encoder_out)
            audio_cross_out, _ = self.cross_attention_layer(decoder_feature_out,
                                                         audio_extra,
                                                         audio_extra,
                                                         encoder_mask)
            audio_cross_out_sig = torch.sigmoid(audio_cross_out)
            total_feature = torch.mul(attention_feature_out, audio_cross_out_sig)
            classification_out = self.classification_layer(torch.cat((total_feature, mapping_text_feature), dim=-1))
        else:
            classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
        return classification_out

# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True,
#         embedding_shape: int = 128
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, embedding_shape)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.audio_extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + embedding_shape, output_dim)

#     def forward(self,
#                 encoder_out,
#                 encoder_mask,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             score = torch.sigmoid(extra)
#             attention_feature_out, _ = self.attention_layer(decoder_feature_out,
#                                                             score,
#                                                             score)
#             audio_extra = self.audio_extract_liner(encoder_out)
#             audio_cross_out, _ = self.cross_attention_layer(decoder_feature_out,
#                                                          audio_extra,
#                                                          audio_extra,
#                                                          encoder_mask)
#             audio_cross_out_sig = torch.sigmoid(audio_cross_out)
#             total_feature = torch.mul(attention_feature_out, audio_cross_out_sig)
#             classification_out = self.classification_layer(torch.cat((total_feature, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

# # #new2_sig_or
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, 128)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + 128, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             score = torch.sigmoid(extra)
#             attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
#                                                             score,
#                                                             score)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

# # #new2_sig_cross
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, 128)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + 128, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             score = torch.sigmoid(extra)
#             attention_feature_out, _ = self.cross_attention_layer(extra,
#                                                             score,
#                                                             score)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out


# #new2_sig_be
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, 128)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + 128, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             score = torch.sigmoid(extra)
#             attention_feature_out, _ = self.cross_attention_layer(extra,
#                                                             score,
#                                                             score)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

# #new2_sig
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True,
#         embedding_shape: int = 128
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, embedding_shape)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + embedding_shape, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             score = torch.sigmoid(extra)
#             attention_feature_out, _ = self.cross_attention_layer(extra,
#                                                             score,
#                                                             score)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

#new2
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, 128)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + 128, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             attention_feature_out, _ = self.cross_attention_layer(extra,
#                                                             extra,
#                                                             extra)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

#new1
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, 128)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim + 128, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
#                                                             extra,
#                                                             extra)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, mapping_text_feature), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

#new_sig
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, decoder_output_feature_dim)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim * 2, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             extra = torch.sigmoid(extra)
#             attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
#                                                             mapping_text_feature,
#                                                             mapping_text_feature)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, extra), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

#new
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, decoder_output_feature_dim)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim * 2, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             attention_feature_out, _ = self.cross_attention_layer(decoder_feature_out,
#                                                             mapping_text_feature,
#                                                             mapping_text_feature)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, extra), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out

#or_sig
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, decoder_output_feature_dim)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim * 2, output_dim)

    # def forward(self,
    #             this_text_embedding,
    #             decoder_feature_out):
    #     this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
    #     mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
    #     if self.use_extract_dysf_layer:
    #         extra = self.extract_liner(decoder_feature_out)
    #         attention_feature_out, _ = self.cross_attention_layer(extra,
    #                                                         mapping_text_feature,
    #                                                         mapping_text_feature)
    #         classification_out = self.classification_layer(torch.cat((attention_feature_out, torch.sigmoid(extra)), dim=-1))
    #     else:
    #         classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
    #     return classification_out

#or
# class DisfClassificationLayer(torch.nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         decoder_output_feature_dim,
#         output_dim,
#         attention_heads,
#         src_attention_dropout_rate: float = 0.0,
#         use_extract_dysf_layer: bool = True
#     ):
#         """Construct an DecoderLayer object."""
#         super().__init__()
#         #输出为2的N次方维度
#         self.use_extract_dysf_layer = use_extract_dysf_layer
#         self.mapping_dim_layer = torch.nn.Embedding(vocab_size, decoder_output_feature_dim)
#         if use_extract_dysf_layer:
#             self.extract_liner = torch.nn.Linear(decoder_output_feature_dim, decoder_output_feature_dim)
#             self.cross_attention_layer = MultiHeadedCrossAttention(
#                         attention_heads, decoder_output_feature_dim, src_attention_dropout_rate)
#         self.classification_layer = torch.nn.Linear(decoder_output_feature_dim * 2, output_dim)

#     def forward(self,
#                 this_text_embedding,
#                 decoder_feature_out):
#         this_text_embedding_zoom = torch.argmax(this_text_embedding, dim=-1)
#         mapping_text_feature = self.mapping_dim_layer(this_text_embedding_zoom)
#         if self.use_extract_dysf_layer:
#             extra = self.extract_liner(decoder_feature_out)
#             attention_feature_out, _ = self.cross_attention_layer(extra,
#                                                             mapping_text_feature,
#                                                             mapping_text_feature)
#             classification_out = self.classification_layer(torch.cat((attention_feature_out, extra), dim=-1))
#         else:
#             classification_out = self.classification_layer(torch.cat((decoder_feature_out, mapping_text_feature), dim=-1))
#         return classification_out
