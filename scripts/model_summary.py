from torchsummary import summary
from model import text_cnn, bert2cnn
from transformers import BertForSequenceClassification, BertConfig
from configs.config import *


if __name__ == '__main__':
    # BERT参数设置
    bert_config = BertConfig.from_pretrained(bert_name)
    bert_config.output_hidden_states = False
    bert_config.num_labels = num_classes
    bert_config.hidden_dropout_prob = hidden_dropout_prob
    # 模型加载
    bert_cls_model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=bert_name, config=bert_config)
    print("\n\n")
    print(bert_cls_model)
    # summary(bert_cls_model, input_size=[(16, 100, 768), (16, 100, 768), (16, 100, 768)])
    # summary(bert_cls_model)
    cnn_model = text_cnn.TextCNN(None, vocab_len, vec_dim=vec_dim, num_kernel=num_kernel, max_length=max_length, num_classes=num_classes, conv_sizes=conv_sizes, dropout=cnn_dropout)
    print("\n\n")
    print(cnn_model)
    # summary(bert_cls_model, input_size=(16, 100, 300))
    # summary(bert_cls_model)
    bert_cls_model = bert2cnn.BERT2CNN(bert_name, bert_config, cnn_model)
    # summary(bert_cls_model, input_size=[(16, 100, 768), (16, 100, 768), (16, 100, 768)])
    # summary(bert_cls_model)
    print("\n\n")
    print(bert_cls_model)

