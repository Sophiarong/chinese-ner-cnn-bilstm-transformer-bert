import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    #Required parameters
    # parser.add_argument("--train", action="store_true",
    #                     help="Whether to run training.")
    # parser.add_argument("--predict", action="store_true",
    #                     help="Whether to predict")
    parser.add_argument("--device", default=None, type=int, required=True,
                        help="GPU number")
    parser.add_argument("--batch_size", default=None, type=int, required=True,
                        help="train and dev has the same batch size")
    parser.add_argument("--epoch", default=None, type=int, required=True)
    parser.add_argument("--lr", default=None, type=float, required=True)
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--dev_file", default=None, type=str, required=True)
    parser.add_argument("--test_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    return parser