from copy import deepcopy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler, BertLayerNorm, gelu
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

def clones(module, N):
    return nn.ModuleList([cp(module) for _ in range(N)])

class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.token_type_embeddings_extended = nn.Embedding(10, config.hidden_size)
        #self.token_type_embeddings_extended.weight[:1, :] = self.token_type_embeddings.weight

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings_extended(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        #return super(RobertaEmbeddings, self).forward(input_ids,
        #                                              token_type_ids=token_type_ids,
        #                                              position_ids=position_ids,
        #                                              inputs_embeds=inputs_embeds)

class BertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        position_ids,
        token_type_ids,
        attention_mask
    ):

        device = input_ids.device
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            device=device, dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
        )

        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, ) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, (hidden_states), (attentions)



class RobertaModel(BertModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return F.log_softmax(x, dim = -1)

class myRobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(myRobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
            self,
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask,
    ):
        outputs = self.roberta(
            input_ids,
            position_ids,
            token_type_ids,
            attention_mask
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        return prediction_scores

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self,
                src_inputs,
                tgt_inputs,
                all_inputs,
    ):

        # Source side Hidden States
        output_src = self.roberta(
            *src_inputs
        )
        hidden_src = output_src[0][:, 0, :]

        # Target side Hidden States
        output_tgt = self.roberta(
            *tgt_inputs
        )
        hidden_tgt = output_tgt[0][:, 0, :]

        # Both
        output_all = self.roberta(
            *all_inputs
        )
        hidden_all = output_all[0][:, 0, :]

        new_hidden = torch.cat([hidden_src, hidden_tgt, hidden_all, hidden_src-hidden_tgt, hidden_src * hidden_tgt], dim = -1)
        result = self.classifier(new_hidden)

        return F.sigmoid(result)