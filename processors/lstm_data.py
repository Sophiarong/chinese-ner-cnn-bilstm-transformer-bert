from torch.utils.data import Dataset
import torch


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

def build_corpus(mode, data_dir, make_vocab=True):
    """读取数据"""
    assert mode in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '<PAD>': 4}

    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words = line.split("  ")
            word_list = []
            tag_list = []
            for word in words:
                for i, ch in enumerate(word):
                    word_list.append(ch)
                    if i == 0 and len(word) > 1:
                        tag_list.append('B')
                    elif i == 0 and len(word) == 1:
                        tag_list.append('S')
                    elif i != 0 and i+1 == len(word):
                        tag_list.append('E')
                    else:
                        tag_list.append('M')
            word_lists.append(word_list)
            tag_lists.append(tag_list)


        # if mode in ['train']:
        #     word_lists = sorted(word_lists, key=lambda x:len(x), reverse=True)
        #     tag_lists = sorted(tag_lists, key=lambda x:len(x), reverse=True)

        if make_vocab:
            word2id = build_map(word_lists)
            # tag2id = build_map(tag_lists)
            word2id['<UNK>'] = len(word2id)
            word2id['<PAD>'] = len(word2id)

            # tag2id['<PAD>'] = len(tag2id)
            return word_lists, tag_lists, word2id, tag2id
        else:
            return word_lists, tag_lists

class MyDataset(Dataset):
    def __init__(self, args, datas, tags, word2id, tag2id):
        self.args = args
        self.datas = datas
        self.tags = tags
        self.word2id = word2id
        self.tag2id = tag2id

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        #将一句话中所有的字转换成index，如果该字符在map中不存在，返回unk的index
        data_index = [self.word2id.get(i, self.word2id['<UNK>']) for i in data]
        tag_index = [self.tag2id[i] for i in tag]

        return data_index, tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.datas)

    def batch_data_pro(self, batch_datas):
        global device
        data, tag = [], []
        da_len = []
        for da, ta in batch_datas:
            data.append(da)
            tag.append(ta)
            da_len.append(len(da))
        max_len = max(da_len)

        #对每一句话（index序列）进行填充
        data = [i + [self.word2id['<PAD>']]*(max_len - len(i)) for i in data]
        tag = [i + [self.tag2id['<PAD>']]*(max_len - len(i)) for i in tag]

        data = torch.tensor(data, dtype=torch.long, device=self.args.device)
        tag = torch.tensor(tag, dtype=torch.long, device=self.args.device)

        return data, tag, da_len






