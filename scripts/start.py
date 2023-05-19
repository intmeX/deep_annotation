import os
import shutil
import argparse


config_path = '../configs/config.py'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False)
    parser.add_argument('-d', '--data_path', required=False)
    parser.add_argument('-f', '--feature', default='text', required=False)
    parser.add_argument('-l', '--label', default='tag', required=False)
    parser.add_argument('-len', '--max_length', type=int, required=False)
    parser.add_argument('-n', '--num_classes', type=int, required=False)

    parser.add_argument('-m', '--model_name', required=False)
    parser.add_argument('-mp', '--model_path', required=False)

    parser.add_argument('-t', '--threshold', type=float, required=False)
    parser.add_argument('-b', '--batch_size', type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', type=float, required=False)
    parser.add_argument('-e', '--epoch', type=int, required=False)
    parser.add_argument('-sc', '--schedule', required=False)
    parser.add_argument('-eg', '--exp_gamma', type=float, required=False)
    parser.add_argument('-w', '--warmup', type=int, required=False)
    parser.add_argument('-mi', '--max_iters', type=int, required=False)
    parser.add_argument('-ds', '--decay_start', type=int, required=False)

    parser.add_argument('-bn', '--bert_name', required=False)

    parser.add_argument('--num_kernel', type=int, required=False)
    parser.add_argument('--cnn_dropout', type=float, required=False)

    parser.add_argument('-sl', '--single_label', default=False, action='store_true', required=False)

    parser.add_argument('-p', '--predict', default=False, action='store_true', required=False)

    args, _ = parser.parse_known_args()

    if args.config and os.path.exists(args.config) and args.config[-3:] == '.py':
        if os.path.exists(args.config):
            os.remove(config_path)
        shutil.copy(args.config, config_path)
    from configs import config as cfg
    if args.data_path:
        cfg.data_path = args.data_path
    if args.feature:
        cfg.feature = args.feature
    if args.label:
        cfg.label = args.label
    if args.max_length:
        cfg.max_length = args.max_length
    if args.num_classes:
        cfg.num_classes = args.num_classes
    if args.model_name:
        cfg.model_name = args.model_name
    if args.model_path:
        cfg.model_path = args.model_path

    if args.batch_size:
        cfg.batch_size = args.batch_size
    if not (args.threshold is None):
        cfg.threshold = args.threshold
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.epoch:
        cfg.epoch = args.epoch
    if args.schedule:
        cfg.schedule = args.schedule
    if args.exp_gamma:
        cfg.exp_gamma = args.exp_gamma
    if args.warmup:
        cfg.warmup = args.warmup
    if args.max_iters:
        cfg.max_iters = args.max_iters
    if args.decay_start:
        cfg.decay_start = args.decay_start

    if args.bert_name:
        cfg.bert_name = args.bert_name

    if args.num_kernel:
        cfg.num_kernel = args.num_kernel
    if args.cnn_dropout:
        cfg.cnn_dropout = args.cnn_dropout

    if args.single_label:
        cfg.single_label = True

    if args.predict:
        if args.model_path is None:
            print('no model path in this predict task\n')
            return
        if cfg.model_name == 'lstm':
            from predict import pre_lstm as pre
        elif cfg.model_name == 'cnn':
            from predict import pre_cnn as pre
        elif cfg.model_name == 'bert':
            from predict import pre_bert as pre
        elif cfg.model_name == 'bert2cnn':
            from predict import pre_bert2cnn as pre
        else:
            raise Exception('No such model: {}'.format(cfg.model_name))
        pre.main()
    else:
        if args.model_path is None:
            cfg.model_path = '_'
        if cfg.model_name == 'lstm':
            from learn import learn_lstm as learning
        elif cfg.model_name == 'cnn':
            from learn import learn_cnn as learning
        elif cfg.model_name == 'bert':
            from learn import learn_bert as learning
        elif cfg.model_name == 'bert2cnn':
            from learn import learn_bert2cnn as learning
        else:
            raise Exception('No such model: {}'.format(cfg.model_name))
        learning.main()


if __name__ == '__main__':
    main()

