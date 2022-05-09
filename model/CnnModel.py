import torch.nn as nn
import torch

class CnnModel(nn.Module):
    def __init__(self, vocab_size, embed_size, class_num, pad_index):
        super().__init__()
        self.kernel_size = 3
        self.embedding = nn.Embedding(vocab_size, embed_size)

        #创建一维卷积层
        self.conv1 = nn.Conv1d(embed_size, embed_size, self.kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(embed_size, embed_size, self.kernel_size, stride=1, padding=1)
        self.classifier = nn.Linear(embed_size, class_num)

        self.cross_loss = nn.CrossEntropyLoss(ignore_index=pad_index)



    def forward(self, inputs, tag_index=None):
        embedding = self.embedding(inputs)
        embedding = embedding.permute(0,2,1)

        x = self.relu(self.conv1(embedding))
        h = self.conv2(x)
        h = h.permute(0,2,1)

        output = self.classifier(h)
        self.result = torch.argmax(output, dim=-1)
        self.pre = torch.argmax(output, dim=-1).reshape(-1)

        if tag_index is not None:
            loss = self.cross_loss(output.reshape(-1, output.shape[-1]), tag_index.reshape(-1))
            return loss


