import torch.nn.functional as F

from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler, BertForMaskedLM, BertOnlyMLMHead


class myBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super(myBertEmbeddings, self).__init__(config)

    def forward(self, input_ids, position_ids, token_type_ids):
        inputs_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class myBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(myBertModel, self).__init__(config)
        self.config = config

        self.embeddings = myBertEmbeddings(config)
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

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask):
        device = input_ids.device
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            device=device, dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
        )

        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, (hidden_states), (attentions)

class myBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super(myBertForMaskedLM, self).__init__(config)
        self.bert = myBertModel(config)
        self.init_weights()

    def forward(self, input_ids, position_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids, position_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return F.log_softmax(prediction_scores, dim=-1)
