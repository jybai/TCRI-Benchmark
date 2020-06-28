import numpy as np
import tensorflow as tf
import yaml
from attrdict import AttrDict

from utils.utility import *
from utils.metrics import metrics_dict as md
from utils.metrics import streaming_consufion_matrix, cm_summary

model_name = 'ConvLSTM'

class Model():
    def __init__(self, training, x, seqlen, y, config_filepath='./config.yaml', 
                 bias=None):
        with open(config_filepath, 'r') as yf:
            config = yaml.load(yf)

        self.config = AttrDict(config)
        self.training = training
        self.x = x 
        self.seqlen = seqlen 
        self.y = y

        if bias == None:
            self.bias = tf.ones([2], tf.float32)
        else:
            assert(len(bias) == 2)
            self.bias = np.array(bias)

        self.prediction
        self.optimize
        self.regularizer_sum
        self.error

        self.define_metrics()

    def define_metrics(self):
# {{{
        with tf.name_scope('metric'):
            self.metric = {}
            self.metric_op = {}
            self.summaries = []
            weights = tf.sequence_mask(self.seqlen, maxlen=tf.shape(self.y)[1])

            for m in ['acc', 'f1', 'pre', 'rec', 'heidke', 'prauc']:
                self.metric[m], self.metric_op[m] = md[m](
                    self.discrete_y, 
                    self.discrete_prediction, 
                    weights=weights)
                self.summaries.append(tf.summary.scalar(m, self.metric[m]))
            self.metric['bss'], self.metric_op['bss'] = md['bss'](
                    self.y, self.prediction, weights=weights)
            self.summaries.append(tf.summary.scalar('bss', self.metric['bss']))
# }}}

    def convlstm_layer(self, config, hidden_length, input_channels):
# {{{
        lstm_cells = []
        for lstm in config.cells:

            input_keep_prob = tf.cast(config.input_keep_prob, tf.float32)
            output_keep_prob = tf.cast(config.output_keep_prob, tf.float32)
            state_keep_prob = tf.cast(config.state_keep_prob, tf.float32)

            conv_lstm = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[hidden_length, hidden_length, input_channels],
                    output_channels=lstm.hidden_size,
                    kernel_shape=[lstm.kernel_size, lstm.kernel_size],
                    use_bias=True,
                    skip_connection=False,
                    forget_bias=1.0,
                    name=lstm.name
                    ) # output shape = [?, ?, 16, 16, 16]
            conv_lstm = tf.contrib.rnn.DropoutWrapper(
                    conv_lstm, 
                    input_keep_prob=tf.cond(
                        self.training, (lambda: input_keep_prob), (lambda: 1.0)),
                    output_keep_prob=tf.cond(
                        self.training, (lambda: output_keep_prob), (lambda: 1.0)),
                    state_keep_prob=tf.cond(
                        self.training, (lambda: state_keep_prob), (lambda: 1.0)),
                    dtype=tf.float32
            )
            input_channels = lstm.hidden_size
            lstm_cells.append(conv_lstm)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        return stacked_lstm, input_channels
# }}}

    def _cnn_encoder_block(self, conv_input, hidden_length, input_channels):
# {{{
        for conv in self.config.cnn:
            conv_input = tf.layers.conv2d(
                    conv_input, 
                    filters=conv.filter_size,
                    kernel_size=conv.kernel_size,
                    strides=conv.strides,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    bias_initializer=tf.constant_initializer(0),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.regular_factor),
                    name=conv.name
                )
            # bn.
            conv_input = tf.layers.batch_normalization(conv_input, training=self.training)

            hidden_length = int((hidden_length - conv.kernel_size)/conv.strides + 1)
            input_channels = conv.filter_size
        return conv_input, hidden_length, input_channels
# }}}

    def _rnn_block(self, rnn_input, seqlen, batch_size, hidden_length, input_channels):
# {{{
        cell, input_channels = self.convlstm_layer(self.config.lstm, hidden_length, input_channels)

        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_output, _ = tf.nn.dynamic_rnn(
                cell,
                rnn_input,
                sequence_length=seqlen,
                initial_state=initial_state,
                time_major=False,
                dtype=tf.float32)
        return rnn_output, hidden_length, input_channels
# }}}

    def _fc_block(self, fc_input):
# {{{
        fc_keep_prob = self.config.fc_keep_prob
        for fc in self.config.fc:
            dropout_input = tf.layers.dense(
                    fc_input,
                    fc.hidden_size,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.regular_factor),
                    name=fc.name,
                )
            fc_input = tf.nn.dropout(
                   dropout_input,
                   keep_prob=tf.cond(
                       self.training, (lambda: tf.cast(fc_keep_prob, tf.float32)), (lambda: 1.0))
                )
        return fc_input
# }}}

    @lazy_property_with_scope
    def discrete_prediction(self):
        prediction_max = tf.argmax(self.prediction, axis=-1)
        return prediction_max

    @lazy_property_with_scope
    def discrete_y(self):
        y_max = tf.argmax(self.y, axis=-1)
        return y_max

    @lazy_property_with_scope
    def error(self):
        weights = tf.reduce_sum(self.y * self.bias, -1)
        err = tf.losses.softmax_cross_entropy(
            self.y, 
            self.prediction, 
            weights=weights)
        return err + self.regularizer_sum

    @lazy_property_with_scope
    def optimize(self):
        # update bn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            return optimizer.minimize(self.error)
 
    @lazy_property_with_scope
    def prediction(self, x=None, seqlen=None):
# {{{
        if x is None:
            x = self.x
        if seqlen is None:
            seqlen = self.seqlen

        batch_size = tf.shape(x)[0]
        max_time = self.config.dataloader.encode_length

        squeezed_x = tf.reshape(x, [batch_size * max_time, 64, 64, 2])

        hidden_length = 64
        input_channels = 2

        conv_output, hidden_length, input_channels = self._cnn_encoder_block(
                squeezed_x, hidden_length, input_channels)
        
        conv_output_expand = tf.reshape(
            conv_output, 
            [batch_size, max_time, hidden_length, hidden_length, input_channels])

        rnn_output, hidden_length, input_channels = self._rnn_block(
                conv_output_expand, seqlen, batch_size, hidden_length, input_channels)
        rnn_output_transposed = tf.transpose(rnn_output, [0, 2, 3, 1, 4])
        rnn_output_flat = tf.reshape(rnn_output, [batch_size, hidden_length, hidden_length, input_channels * max_time])

        rnn_output_compressed = tf.layers.conv2d(
                rnn_output_flat, 
                filters=input_channels,
                kernel_size=1,
                strides=1,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.regular_factor),
                name='state_compression_conv1'
            )
        rnn_output_flat = tf.reshape(rnn_output_compressed, [batch_size, hidden_length * hidden_length * input_channels])

        fc_output = self._fc_block(rnn_output_flat)

        careful_bias = np.log(self.bias[0] / self.bias[1])
        careful_bias_initializer = tf.constant_initializer(careful_bias)

        logits = tf.layers.dense(
                fc_output,
                2,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                bias_initializer=careful_bias_initializer,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.regular_factor),
                name="prob_positive",
                )
        logits = tf.reshape(logits, [batch_size, 1, 2])

        return logits
# }}}

    @lazy_property_with_scope
    def prediction_prob(self):
        return tf.nn.sigmoid(self.prediction)

    @lazy_property_with_scope
    def regularizer_sum(self):
        all_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        return sum(all_regularizers)
