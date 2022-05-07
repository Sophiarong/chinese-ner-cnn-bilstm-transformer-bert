import os.path

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from model.LstmModel import LstmModel
from processors.lstm_output import write_output
from utils.argparse import get_argparse
from processors.lstm_data import build_corpus,MyDataset
import torch
from utils.common import init_logger, logger
from utils.common import seed_everything
import prettytable as pt
import time
from loss.calculate_loss import pre2region, ans2region, calculate


def main():
    #config参数读取
    args = get_argparse().parse_args()
    init_logger(log_file="./log/lstm_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
    logger.info(args)

    #设置随机数
    seed_everything()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    #读取数据集
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", args.train_file)
    dev_word_lists, dev_tag_lists = build_corpus("dev", args.dev_file, make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", args.test_file, make_vocab=False)

    corpus_num = len(word2id)
    class_num = len(tag2id)

    table = pt.PrettyTable(["train", "dev", "test", "tags"])
    table.add_row([len(train_word_lists), len(dev_word_lists), len(test_word_lists), class_num])
    logger.info("\n{}".format(table))


    #将数据集变成dataset
    train_dataset = MyDataset(args, train_word_lists, train_tag_lists, word2id, tag2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dataset.batch_data_pro)

    dev_dataset = MyDataset(args, dev_word_lists, dev_tag_lists, word2id, tag2id)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dev_dataset.batch_data_pro)

    test_dataset = MyDataset(args, test_word_lists, test_tag_lists, word2id, tag2id)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.batch_data_pro)

    #初始化model
    embedding_num = 128
    hidden_num = 129
    bi = True

    model = LstmModel(embedding_num, hidden_num, corpus_num, bi, class_num, word2id['<PAD>'])
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info("\n{}".format(model))

    best_dev = -1
    #训练
    for e in range(args.epoch):
        epoch_loss = 0
        epoch_step = 0

        model.train()
        for data, tag, da_len in train_dataloader:
            loss = model.forward(data, da_len, tag)
            epoch_loss = epoch_loss + loss
            epoch_step = epoch_step + 1
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        dev_res = []

        for dev_data, dev_tag, dev_da_len in dev_dataloader:
            model.forward(dev_data, dev_da_len, dev_tag)

            r = model.result.cpu().numpy()
            for i in range(len(r)):
                dev_res.append(r[i].tolist())

        #f1计算
        precision, recall, f1 = calculate(dev_res, dev_tag_lists)
        table = pt.PrettyTable(["Train {} Loss".format(e), "Dev Precision", "Dev Recall", "f1"])
        table.add_row(["{:.4f}".format(epoch_loss/epoch_step), "{:.4f}".format(precision), "{:.4f}".format(recall), "{:.4f}".format(f1)])
        logger.info("\n{}".format(table))

        if f1 > best_dev:
            best_dev = f1
            torch.save(model.state_dict(), 'lstm.params')


    clone = LstmModel(embedding_num, hidden_num, corpus_num, bi, class_num, word2id['<PAD>'])
    clone.load_state_dict(torch.load('lstm.params'))
    clone.to(args.device)
    clone.eval()
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    result = []
    for test_data, test_tag, test_da_len in test_dataloader:
        clone.forward(test_data, test_da_len, test_tag)
        r = clone.result.cpu().numpy()
        for i in range(len(r)):
            result.append(r[i].tolist())
    write_output(args, result)

if __name__ == '__main__':
    main()
