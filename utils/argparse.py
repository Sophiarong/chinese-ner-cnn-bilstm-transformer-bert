import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    #Required parameters
    # parser.add_argument("--predict", action="store_true",
    #                     help="Whether to predict")
    parser.add_argument("--device", default=None, type=int, required=True,
                        help="GPU number")
    parser.add_argument("--batch_size", default=None, type=int, required=True,
                        help="train and dev has the same batch size")
    parser.add_argument("--epoch", default=None, type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", default=None, type=int, required=True)
    parser.add_argument("--dropout", default=None, type=float, required=True)
    parser.add_argument("--lr", default=None, type=float, required=True)
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--dev_file", default=None, type=str, required=True)
    parser.add_argument("--test_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    parser.add_argument("--scheduler_step_size", default=50, type=int, required=True)
    parser.add_argument("--scheduler_gamma", default=0.5, type=float, required=True)

    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--path_checkpoint", type=str)
    parser.add_argument("--continue_save", action="store_true")
    parser.add_argument("--save_checkpoint", type=str)


    return parser