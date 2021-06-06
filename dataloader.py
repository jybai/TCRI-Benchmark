import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import os
from functools import partial

from utils.utility import deserialize_example, rotate_and_crop, mask_gpu

class Dataloader():
    def __init__(self, opts, data_split, training):
        '''
        opts: dict
        data_split: 'train' | 'valid' | 'test'
        training: tf.placeholder(dtype=tf.bool)
        '''
        assert(data_split == 'train' or data_split == 'valid' or data_split == 'test')
        self.threshold = [-np.inf, 35, np.inf]

        self.training = training
        self.data_path = opts.data_path + '.' + data_split
        self.batch_size = opts.batch_size
        self.num_workers = opts.num_workers
        self.estimate_distance = opts.estimate_distance
        self.encode_length = opts.encode_length
        self.rotate_type = opts.rotate_type
        self.return_angle = False if 'return_angle' not in opts else opts.return_angle

        self.prepare_dataset()

    def prepare_dataset(self):
        rc_with_type = partial(rotate_and_crop, 
                               rotate_type=self.rotate_type, 
                               return_angle=self.return_angle)
        crop_only = partial(rotate_and_crop, 
                            rotate_type=None, 
                            return_angle=self.return_angle)

        self.dataset = tf.data.TFRecordDataset(
            [self.data_path], num_parallel_reads=self.num_workers
        ).map(
            deserialize_example, num_parallel_calls=self.num_workers
        ).filter(
            lambda x, y, l: l >= self.encode_length + 
                                 self.estimate_distance
        ).apply(tf.contrib.data.parallel_interleave(
            self.split_by_n,
            cycle_length=8,
            sloppy=True
        ))
        
        if self.training:
            self.dataset = self.dataset.map(rc_with_type, num_parallel_calls=self.num_workers)\
                                       .shuffle(buffer_size=1000)
        else:
            self.dataset = self.dataset.map(crop_only, num_parallel_calls=self.num_workers)

        self.dataset = self.dataset.batch(
            self.batch_size,
            drop_remainder=False
        ).prefetch(
            buffer_size=4
        ).make_initializable_iterator()

    def split_by_n(self, img, intensity, seqlen, n=None, threshold=None):

        el = self.encode_length
        ed = self.estimate_distance
        assert(el > 0)

        if threshold is None:
            threshold = self.threshold

        ncopies = seqlen - (el + ed) + 1
        # shape=(seqlen,  201, 201, 4)
        rolling = tf.range(ncopies)
        # shape=(seqlen - el)
        crop_img = tf.image.crop_to_bounding_box(img, 54, 54, 93, 93)

        truncated_img = tf.map_fn(
                lambda r: crop_img[r: r + el],
                rolling,
                parallel_iterations=self.num_workers,
                dtype=tf.float32
                )
        # shape=(seqlen - el, el, 201, 201, 4)

        shifted_intensity_diff = intensity[el - 1 + ed:] - intensity[el - 1:-ed]
        # shape=(seqlen - el)
        truncated_ri = tf.stack([
            tf.logical_and(
                shifted_intensity_diff >= threshold[i],
                shifted_intensity_diff < threshold[i+1])
            for i in range(len(threshold) - 1)], -1)
        # shape=(seqlen - el, len(threshold) - 1)
        truncated_ri = tf.cast(truncated_ri, tf.float32)
        truncated_ri = tf.expand_dims(truncated_ri, 1)
        # shape=(seqlen - el, len(threshold) - 1)

        extended_seqlen = tf.reshape(tf.tile([el], [ncopies]), [-1])
        # shape=(seqlen - el)

        return tf.data.Dataset.from_tensor_slices(
                (truncated_img, truncated_ri, extended_seqlen))

    def initialize(self):
        self.dataset.initializer.run()

    def get_next(self):
        return self.dataset.get_next()

def main():
    import time
    import argparse
    import GPUtil
    from attrdict import AttrDict
    import yaml
    import os
    import numpy as np
    from tqdm import tqdm
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--exp", help="path config file, default is './config.yaml'", default="./config.yaml")
    _ = parser.add_argument("-v", help="verbose output", action='store_true')
    args = parser.parse_args()


    mask_gpu()

    training = tf.placeholder(tf.bool, name='training')
    with open(args.exp, 'r') as yf:
        opts = AttrDict(yaml.load(yf))

    dl = Dataloader(opts=opts.dataloader, data_split='train', training=training)
    x, y, s = dl.get_next()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    repeat = 10
    t = []
    if args.v:
        repeat = 1
    with sess.as_default():
        for _ in tqdm(range(repeat)):
            start = time.time()
            dl.initialize()
            distribution = 0
            while True:
                try:
                    batch_x, batch_y, batch_seqlen = sess.run([x, y, s])
                    distribution += np.sum(batch_y, axis=(0, 1))
                    if args.v:
                        print(batch_x.shape)
                        print(batch_y.shape)
                        print(batch_seqlen)
                except tf.errors.OutOfRangeError:
                    break
            print(distribution)
            end = time.time()
            t.append(end - start)
    print('avg_t=%g over %d times'%(np.average(t), repeat))

if __name__ == '__main__':
    main()
