import torch
from torch import nn


class cpcCNN(nn.Module):
    def __init__(self, embedding, vocab_len, vec_dim, num_kernel, max_length=200, num_classes=10, conv_sizes=None, dropout=0.5, init_weights=True):
        super(cpcCNN, self).__init__()
        '''
        '''
        if conv_sizes is None:
            self.conv_sizes = [2, 3, 4]
        else:
            self.conv_sizes = conv_sizes
        self.vec_dim = vec_dim
        self.vocab_len = vocab_len
        self.num_kernel = num_kernel
        self.num_classes = num_classes
        self.dropout = dropout
        self.embedding = embedding
        self.max_length = max_length
        self.conv1Ds = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.vec_dim, self.num_kernel, kernel_size=s),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_length - s + 1),
            )
            for s in conv_sizes
        ])
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.num_kernel * len(conv_sizes), num_classes),
            # 如果train内loss已经附带了Sigmoid，则不需要重复添加
            # torch.nn.Sigmoid(),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        c = []
        for m in self.conv1Ds:
            c.append(torch.squeeze(m(x), 2))
        x = torch.cat(c, dim=1)
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

