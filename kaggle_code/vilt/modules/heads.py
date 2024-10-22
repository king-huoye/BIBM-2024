import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


class ModmisBinHead(nn.Module):
    def __init__(self, hidden_size: int, label_count: int):
        super().__init__()
        self.binarizer = nn.Linear(hidden_size, label_count)

    def forward(self, x):
        return self.binarizer(x)


class ModmisClsHead(nn.Module):
    def __init__(self, hidden_size: int, class_num_list: list):
        super().__init__()
        self.classifier_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, class_num),
            )
            for class_num in class_num_list
        ])

    def forward(self, x):
        return [classifier(x) for classifier in self.classifier_list]


class ModmisRegHead(nn.Module):
    def __init__(self, hidden_size: int, regression_num: int):
        super().__init__()
        # 先尝试简单的线性头 之后再考虑加其它操作（除了pooling已经加上去了）
        self.regresser = nn.Linear(hidden_size, regression_num)

    def forward(self, x):
        return self.regresser(x)
