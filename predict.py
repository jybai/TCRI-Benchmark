import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import argparse
import GPUtil
from attrdict import AttrDict
import yaml
import pandas as pd
from tqdm import tqdm
from functools import partial

from dataloader import Dataloader
from utils.utility import mask_gpu, seed_everything, prepare_session

def predict(model_dir, epoch, split='test', gpu=None, mem=None, verbose=False):

    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as yf:
        opts = AttrDict(yaml.load(yf))

    mask_gpu(gpu)

    with tf.variable_scope("input") as scope:
        training = tf.constant(False, name="training")

    dl = Dataloader(opts=opts.dataloader, data_split=split,
                    training=False)
    batch_x, batch_y, batch_seqlen = dl.get_next()

    module = getattr(__import__('models', fromlist=[opts.flavour]), opts.flavour)
    model = module.Model(training, batch_x, batch_seqlen, batch_y, 
                         config_filepath=config_path, 
                         bias=opts.class_weight)

    sess = prepare_session(mem)
    loader = tf.train.Saver()

    with sess.as_default():
        with tf.device('/gpu:0'):
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            try:
                epoch_int = int(epoch)
                ckpt_epoch = f"epoch{epoch_int}"
            except:
                ckpt_epoch = epoch

            ckpt_file = os.path.join(model_dir, ckpt_epoch)
            loader.restore(sess, ckpt_file)

            dl.initialize()

            while True:
                try:
                    sess.run([model.metric_op])
                except tf.errors.OutOfRangeError:
                    break
            metrics =  sess.run(model.metric)

            if verbose:
                print(model_dir)
                print(metrics)

    return metrics

def main():
    seed_everything()
    # get argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="directory of models")
    parser.add_argument("--models", help="path of models, seperated by commas")
    parser.add_argument("--epoch", help="pretrain model load epoch, can be the number of epoch (ex.  300)\
                                         or the best of some metric (ex. best-prauc)", default='best-prauc')
    parser.add_argument("--split", help="train | valid | test", default="test")
    parser.add_argument("--save_path", help="file path to save predicted results", default=None)
    parser.add_argument("--gpu", help="use specific index gpu", type=int, default=None)
    parser.add_argument("--mem", help="vram usage ratio", type=float, default=None)
    args = parser.parse_args()

    # parse models to model names 
    model_names = args.models.split(',')

    partial_predict = partial(predict, epoch=args.epoch, split=args.split, 
                              gpu=args.gpu, mem=args.mem, verbose=True)

    dfs = []

    for model_name in tqdm(model_names):
        # must reset or else different model going to reuse
        tf.reset_default_graph()

        model_dir = os.path.join(args.model_dir, model_name)
        metrics = {'model': model_name}
        metrics.update(partial_predict(model_dir=model_dir))
        # hack for pd.from_dict
        metrics = {k: [v] for k, v in metrics.items()}

        df = pd.DataFrame.from_dict(metrics)
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)
    if args.save_path is not None:
        dfs.to_csv(args.save_path, index=False)

if __name__ == '__main__':
    main()
