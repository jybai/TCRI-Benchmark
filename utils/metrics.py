import tensorflow as tf
import numpy as np
import tqdm
import os
import GPUtil
import matplotlib
import matplotlib.pyplot as plt
import io
import re
import itertools
from functools import partial

def brier(labels, predictions, weights=None):
    pred_prob = tf.nn.softmax(predictions)
    return metrics_dict['mse'](
	    labels[:, :, 0], pred_prob[:, :, 0], weights=weights)

'''
def bss(labels, predictions, weights=None, basis=0.13):
    bs, bs_op = brier(labels, predictions, weights=weights)
    return 1 - bs/basis, bs_op
'''
def bss(labels, predictions, weights=None, pos_ratio=0.9577):
    bs, bs_op = brier(labels, predictions, weights=weights)
    predictions_ref = tf.ones_like(labels[:, :, 0]) * pos_ratio
    bs_ref, bs_ref_op = metrics_dict['mse'](labels[:, :, 0], 
                                            predictions_ref,
                                            weights=weights)
    return 1 - bs/bs_ref, tf.group([bs_op, bs_ref_op])

def heidke(labels, predictions, weights=None):
    tp, tp_op = tf.metrics.true_positives(labels, predictions, weights=weights)
    tn, tn_op = tf.metrics.true_negatives(labels, predictions, weights=weights)
    fp, fp_op = tf.metrics.false_positives(labels, predictions, weights=weights)
    fn, fn_op = tf.metrics.false_negatives(labels, predictions, weights=weights)
    heidke_op = tf.group([tp_op, tn_op, fp_op, fn_op])
    heidke = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return heidke, heidke_op

def cm_summary(cm, labels, tag='confusion_matrix'):
    fig = _plot_confusion_matrix(cm, labels)   
    summary = _figure_to_summary(fig, tag)
    return summary

def _figure_to_summary(fig, tag):
    if fig.canvas is None:
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # get PNG data from the figure
    png_buffer = io.BytesIO()
    fig.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()

    summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
    encoded_image_string=png_encoded)
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=summary_image)])
    return summary

def _plot_confusion_matrix(cm, labels):
    '''
    :param cm: A confusion matrix: A square ```numpy array``` of the same size as labels
    `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
    '''
    numClasses = len(labels)

    fig = plt.figure(figsize=(numClasses, numClasses), dpi=150, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    # classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    # classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]
    classes = [str(l) for l in labels]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted')
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(numClasses), range(numClasses)):
        ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.', horizontalalignment="center", verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig

def streaming_consufion_matrix(label, prediction, num_classes, weights=None, normalize=False):
    # Compute a per-batch confusion
    batch_confusion = tf.confusion_matrix(label, prediction,
	num_classes=num_classes,
	weights=weights,
	name='batch_confusion')
    confusion = tf.Variable(tf.zeros([num_classes,num_classes], 
	dtype=tf.int32),
	name='confusion',
	collections=[GraphKeys.LOCAL_VARIABLES])
    confusion_update_op = confusion.assign(confusion + batch_confusion)
    normalized_confusion = tf.cast(confusion, tf.float32) / tf.cast(tf.reduce_sum(confusion), tf.float32)

    if normalize:
        return normalized_confusion, confusion_update_op
    else:
        return confusion, confusion_update_op

class Metrics():
    def __init__(self, metrics_type, gt, prediction, weights=None):
        if metrics_type == 'accuracy':
            metric = tf.metrics.accuracy
        elif metrics_type == 'f1':
            metric = tf.contrib.metrics.f1_score
        elif metrics_type == 'precision':
            metric = tf.metrics.precision
        elif metrics_type == 'mean_squared_error':
            metric = tf.metrics.mean_squared_error
        else:
            assert(False)

        self.metrics_type = 'metrics/%s'%metrics_type
        with tf.name_scope(self.metrics_type):
            self.metric, self.update_op = metric(gt, prediction, weights=weights)
        self.stream_vars = [
            v for v in tf.local_variables() if self.metrics_type in v.name]

    def initialize(self):
        tf.variables_initializer(self.stream_vars).run()

def loop(ops, feed_dict={}):

    gpu_index = 0
    max_free_memory = -np.inf
    for i in range(len(GPUtil.getGPUs())):
        gpu = GPUtil.getGPUs()[i]
        if max_free_memory < gpu.memoryFree:
            max_free_memory = gpu.memoryFree
            gpu_index = i
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE_LIST = GPUtil.getGPUs()
    DEVICE_ID = DEVICE_LIST[gpu_index].id # grab first element from list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    while 1:
        try:
            sess = tf.Session(config=config)
        except:
            time.sleep(30)
            continue
        break

    with sess.as_default():
        with tf.device('/gpu:0'):
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            while True:
                try:
                    sess.run(ops, feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break

metrics_dict = {
    'acc': tf.metrics.accuracy,
    'f1': tf.contrib.metrics.f1_score,
    'pre': tf.metrics.precision,
    'rec': tf.metrics.recall,
    'mse': tf.metrics.mean_squared_error,
    'brier': brier,
    'bss': bss,
    'heidke': heidke,
    'prauc': partial(tf.metrics.auc, curve='PR', summation_method='careful_interpolation')
}

