import torch.nn as nn
import torch

class LstmModel(nn.Module):
    def __init__(self, embedding_num, hidden_num, corpus_num, bi, class_num, pad_index):
        super().__init__()
        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.corpus_num = corpus_num
        self.bi = bi

        self.embedding = nn.Embedding(corpus_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num, batch_first=True, bidirectional=bi)

        if bi:
            self.classifier = nn.Linear(hidden_num*2, class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.cross_loss = nn.CrossEntropyLoss(ignore_index=pad_index)

    def forward(self, data_index, data_len, tag_index = None):
        em = self.embedding(data_index)
        pack = nn.utils.rnn.pack_padded_sequence(em, data_len, batch_first=True, enforce_sorted=False)#将填充后的变长序列压紧
        output,_ = self.lstm(pack)
        output, lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)#将压缩完的数据再填充回来

        pre = self.classifier(output)

        self.result = torch.argmax(pre, dim=-1)
        self.pre = torch.argmax(pre, dim=-1).reshape(-1)


        if tag_index is not None:
            loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), tag_index.reshape(-1))
            return loss
