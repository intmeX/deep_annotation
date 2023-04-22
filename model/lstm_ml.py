import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, embedding, vocab_len, vec_dim, hidden_dim, num_layers, num_classes=10, dropout=0.5, init_weights=True):
        super(LSTM, self).__init__()
        '''
        '''
        self.vec_dim = vec_dim
        self.vocab_len = vocab_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.embedding = embedding
        self.lstm = nn.LSTM(vec_dim, hidden_dim, num_layers, bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, num_classes),
            torch.nn.Sigmoid(),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = self.embedding(x)
        x, _ = self.lstm(x)
        # 取最后一维
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        '''
        如果一开始参数都是0，并不见得是一个好的选择，使用一种得到实践认可的初始化方法是非常有用的
        当然可以选择迁移学习是一种更好的方法
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

