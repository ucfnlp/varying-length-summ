from transformers import BertTokenizer, RobertaTokenizer
from utility import detokenize

class myTokenizer:
    def __init__(self, config):
        self.SEP = config.SEP
        self.model = config.pre_training_model
        self.tokenizer = None
        if self.model == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.model == "roberta-base":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def encode(self, sequence):
        if self.model == "bert-base-uncased":
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        elif self.model == "roberta-base":
            return self.tokenizer.encode(sequence, add_special_tokens=False)

    def decode(self, sequence, mapping=None):
        cutOff = []
        for token in sequence:
            if token == self.SEP:
                break
            else:
                cutOff.append(token)

        if len(cutOff) == 0:
            return ""

        if self.model == "bert-base-uncased":
            tokens = self.tokenizer.convert_ids_to_tokens(cutOff)
            tokens = [token.replace('##', '') if token.startswith('##') else ' ' + token for token in tokens]
            return detokenize("".join(tokens)[1:], mapping)
        elif self.model == "roberta-base":
            return self.tokenizer.decode(cutOff)

    def tokenize(self, sequence):
        return self.tokenizer.tokenize(sequence)