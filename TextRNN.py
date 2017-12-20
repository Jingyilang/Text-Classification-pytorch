import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, **kwargs):
        super(RNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.NUM_LAYERS = kwargs["NUM_LAYERS"]
        self.HIDDEN_SIZE = kwargs["HIDDEN_SIZE"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        if self.MODEL == "static":
            self.embedding.weight.requires.grad = False

        self.BiGRU = nn.GRU(self.WORD_DIM, self.HIDDEN_SIZE, dropout=0.5, num_layers=self.NUM_LAYERS, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(300 * 2, self.CLASS_SIZE)

    def init_hidden(self, num_layers, batch_size):
        return torch.zeros(num_layers * 2, batch_size, 300)

    def forward(self, x):
        x = self.embedding(x)
        hidden = nn.Parameter(self.init_hidden(1, self.BATCH_SIZE)).cuda()
        if x.size(0) != self.BATCH_SIZE:
            hidden = nn.Parameter(self.init_hidden(1, x.size(0))).cuda()

        self.BiGRU.flatten_parameters()
        gru_out, hidden = self.BiGRU(x, hidden)

        gru_out = torch.transpose(gru_out, 1, 2).contiguous()
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # print(gru_out.size())
        x = F.relu(gru_out)
        x = F.dropout(x, p=0.5, training=self.train())

        x = self.fc(x)

        return x