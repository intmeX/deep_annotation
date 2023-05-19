import torch
from torch import nn
from transformers import BertModel, BertConfig


class BERT2CNN(nn.Module):
    def __init__(self, bert_name='bert-base-uncased', bert_config=None, cnn=None):
        super(BERT2CNN, self).__init__()
        self.bert_name = bert_name
        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=self.bert_name, config=self.bert_config)
        self.cnn = cnn

    def forward(self, **kwargs):
        x = self.bert(**kwargs)[0]
        x = self.cnn(x)
        return x

