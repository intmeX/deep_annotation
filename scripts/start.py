import os
import shutil
import argparse
from learn import learn_cnn, learn_bert, learn_lstm
from predict import pre_cnn, pre_bert, pre_lstm


config_path = '../configs/config.py'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False)
    parser.add_argument('-d', '--data', required=False)
    parser.add_argument('-f', '--feature', default='text', required=False)
    parser.add_argument('-l', '--label', default='tag', required=False)
    parser.add_argument('-len', '--max_length', type=int, required=False)

    parser.add_argument('-m', '--model_name', required=False)
    parser.add_argument('-mp', '--model_path', required=False)

    parser.add_argument('-lr', '--learning_rate', type=float, required=False)

    parser.add_argument('-p', '--predict', default=False, action='store_true', required=False)

    args, _ = parser.parse_known_args()

    if args.config and os.path.exists(args.config) and args.config[-3:] == '.py':
        if os.path.exists(args.config):
            os.remove(config_path)
        shutil.copy(args.config, config_path)
    from configs import config as cfg
    if args.data:
        cfg.data_path = args.data
    if args.feature:
        cfg.feature = args.feature
    if args.label:
        cfg.label = args.label
    if args.max_length:
        cfg.max_length = args.max_length
    if args.model_name:
        cfg.model_name = args.model_name
    if args.model_path:
        cfg.model_path = args.model_path
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate

    if args.predict:
        if args.model_path is None:
            print('no model path in this predict task\n')
            return
        if cfg.model_name == 'lstm':
            pre_lstm.main()
        elif cfg.model_name == 'cnn':
            pass
        elif cfg.model_name == 'bert':
            pass
    else:
        if cfg.model_name == 'lstm':
            learn_lstm.main()
        elif cfg.model_name == 'cnn':
            pass
        elif cfg.model_name == 'bert':
            pass


if __name__ == '__main__':
    main()
