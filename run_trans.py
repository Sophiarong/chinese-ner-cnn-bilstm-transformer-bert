import os.path
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from callback.lr_scheduler import get_linear_schedule_with_warmup
from model.TransformerModel import TransformerModel, PositionalEncoding, generate_square_subsequent_mask
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
    init_logger(log_file="./log/trans_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
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
    d_model = 128
    d_hid = 128
    nlayers = 2
    nhead = 2

    model = TransformerModel(corpus_num, class_num, d_model, d_hid, nlayers, nhead, word2id['<PAD>'], args.dropout)
    model = model.to(args.device)

    #设置optimizer和schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    logger.info("\n{}".format(model))

    if args.continue_train:
        checkpoint = torch.load(args.path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    best_dev = -1
    #训练
    for e in range(args.epoch):
        epoch_loss = 0
        epoch_step = 0

        model.train()
        for i, data_tag_len in tqdm(enumerate(train_dataloader)):
            data, tag, _ = data_tag_len
            src_mask = generate_square_subsequent_mask(data.size(1)).to(args.device)
            src_padding_mask = (data == word2id['<PAD>'])
            loss = model.forward(data, src_mask, src_padding_mask, tag)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_loss = epoch_loss + loss

            if (i+1)%args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                model.zero_grad()
                epoch_step = epoch_step + 1

        scheduler.step()

        model.eval()
        dev_res = []
        for dev_data, dev_tag, _ in dev_dataloader:
            dev_mask = generate_square_subsequent_mask(dev_data.size(1)).to(args.device)
            dev_padding_mask = (dev_data == word2id['<PAD>'])
            model.forward(dev_data, dev_mask, dev_padding_mask, dev_tag)

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
            torch.save(model.state_dict(), 'trans.params')

    if args.continue_save:
        checkpoint = {
            "net":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict()
        }
        torch.save(checkpoint, args.save_checkpoint)

    clone = TransformerModel(corpus_num, class_num, d_model, d_hid, nlayers, nhead, word2id['<PAD>'], args.dropout)
    clone.load_state_dict(torch.load('trans.params'))
    clone.to(args.device)
    clone.eval()
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    result = []
    for test_data, test_tag, _ in test_dataloader:
        test_mask = generate_square_subsequent_mask(test_data.size(1)).to(args.device)
        test_padding_mask = (test_data == word2id['<PAD>'])
        clone.forward(test_data, test_mask, test_padding_mask, test_tag)
        r = clone.result.cpu().numpy()
        for i in range(len(r)):
            result.append(r[i].tolist())
    write_output(args, result)

if __name__ == '__main__':
    main()