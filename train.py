import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import time
from datetime import datetime
import sys
import h5py
import argparse
import importlib
import GPUtil
import shutil
from attrdict import AttrDict
import yaml

from dataloader import Dataloader
from utils.metrics import cm_summary
from utils.utility import mask_gpu, seed_everything, prepare_session

def main():
    seed_everything()
    # get argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="path of config file", default="./config.yaml")
    parser.add_argument("--gpu", help="use specific index gpu", type=int, default=None)
    parser.add_argument("--mem", help="vram usage ratio", type=float, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    args = parser.parse_args()
    with open(args.exp, 'r') as yf:
        opts = AttrDict(yaml.load(yf))

    mask_gpu(args.gpu)

    with tf.variable_scope("input") as scope:
        training = tf.placeholder(tf.bool, name="training")
        data_split = tf.placeholder(tf.string, name="data_split")

    dl_train = Dataloader(opts=opts.dataloader, data_split='train', 
                          training=True)
    dl_valid = Dataloader(opts=opts.dataloader, data_split='valid',
                          training=False)
    batch_x, batch_y, batch_seqlen = tf.cond(tf.equal(data_split, 'train'), 
                                             dl_train.get_next, 
                                             dl_valid.get_next)

    module = getattr(__import__('models', fromlist=[opts.flavour]), opts.flavour)
    model = module.Model(training, batch_x, batch_seqlen, batch_y, 
                         config_filepath=args.exp, bias=opts.class_weight)

    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    model_name = f"ConvLSTM_{timestamp}_"
    print(model_name)

    if not args.test:
        directory = os.path.join(opts.saved_model_path, model_name)
        assert(not tf.gfile.Exists(directory))
        tf.gfile.MkDir(directory)
        # copy config file over
        shutil.copy(args.exp, os.path.join(directory, 'config.yaml'))
        saver = tf.train.Saver(max_to_keep=100)

    summary = tf.summary.merge(model.summaries)

    stream_vars = [v for v in tf.local_variables() if 'metric/' in v.name]

    sess = prepare_session(args.mem)

    if not args.test:
        train_writer = tf.summary.FileWriter(
            os.path.join(opts.log_dir, model_name, 'train'), 
            sess.graph)
        valid_writer = tf.summary.FileWriter(
            os.path.join(opts.log_dir, model_name, 'valid'), 
            sess.graph)

    best_prauc = 0
    now_prauc = 0
    best_hss = -np.inf
    now_hss = -np.inf

    with sess.as_default():
        # Select device.
        with tf.device('/gpu:0'):
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            for i in range(1, opts.epochs + 1):
                # train
                dl_train.initialize()

                while True:
                    try:
                        sess.run([model.optimize],
                                 feed_dict={training: True, data_split: 'train'})
                    except tf.errors.OutOfRangeError:
                        break

                # monitoring and validation
                if i % 10 == 0 or args.test:

                    dl_train.initialize()
                    dl_valid.initialize()

                    for data_split_ in ['train', 'valid']:

                        tf.variables_initializer(stream_vars).run()
                        while True:
                            try:
                                sess.run([model.metric_op],
                                         feed_dict={training: False, data_split: data_split_})
                            except tf.errors.OutOfRangeError:
                                break

                        log = sess.run(summary)
                        if not args.test:
                            writer = train_writer if data_split_ == 'train' else valid_writer
                            writer.add_summary(log, i)

                    now_prauc = model.metric['prauc'].eval()
                    now_hss = model.metric['heidke'].eval()

                    if not args.test:
                        if now_prauc > best_prauc:
                            best_prauc = now_prauc
                            saver.save(sess, os.path.join(directory, "best-prauc"))
                        if now_hss > best_hss:
                            best_hss = now_hss
                            saver.save(sess, os.path.join(directory, "best-hss"))

                if i % 100 == 0:
                    if not args.test:
                        saver.save(sess, os.path.join(directory, f"epoch{i}"))

                t = time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))
                print(f"{t} epoch: {i:4d}", end='\r', flush=True)

if __name__ == '__main__':
    main()

