import numpy as np
import tensorflow as tf
import yaml
from attrdict import AttrDict

from utils.utility import *
from utils.metrics import metrics_dict as md
from utils.metrics import streaming_consufion_matrix, cm_summary

model_name = 'ConvLSTM'

from models.model_plain import Model as PlainModel

class Model(PlainModel):
    def __init__(self, training, x, seqlen, y, config_filepath, 
                 bias=None):
        self._attn = None
        super().__init__(training, x, seqlen, y, config_filepath, bias)

    def self_attention_layer(self, sa_input, hidden_length, config):
# {{{
        # expected sa_input shape = (?, H, W, C)
        attn = tf.layers.conv2d(
            sa_input,
            filters=config.hidden_size,
            kernel_size=config.kernel_size,
            padding='same',
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.regular_factor),
            name='attn'
            )
        attn = tf.layers.conv2d(
            attn,
            filters=1,
            kernel_size=1,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config.regular_factor),
            name='attn_flat'
            )
        attn = tf.reshape(attn, [-1, hidden_length * hidden_length])
        attn = tf.nn.softmax(attn)  
        attn = tf.reshape(attn, [-1, hidden_length, hidden_length])
        self._attn = attn
        # [batch_size, sequence_length, sequence_length]

        attn = tf.nn.dropout(attn, keep_prob=tf.cond(
            self.training,
            lambda: tf.cast(config.keep_prob, tf.float32),
            lambda: 1.0))
        attn = tf.expand_dims(attn, -1)
        attn = attn * hidden_length * hidden_length

        sa_output = attn * sa_input

        return sa_output
# }}}

    def _rnn_block_wrapped(self, rnn_input, seqlen, batch_size, hidden_length, input_channels):
# {{{
        max_time = tf.shape(rnn_input)[1]
        rnn_input_flat = tf.reshape(
                rnn_input, [-1, hidden_length, hidden_length, input_channels])

        sa_output = self.self_attention_layer(
                rnn_input_flat, hidden_length, self.config.lstm.selfattn)

        sa_output_expand = tf.reshape(
                sa_output, [batch_size, max_time, hidden_length, hidden_length, input_channels])

        rnn_output, hidden_length, input_channels = self._rnn_block(
                sa_output_expand, seqlen, batch_size, hidden_length, input_channels)

        return rnn_output, hidden_length, input_channels
# }}}

    @lazy_property_with_scope
    def prediction(self, x=None, seqlen=None):
# {{{
        if x is None:
            x = self.x
        if seqlen is None:
            seqlen = self.seqlen

        batch_size = tf.shape(x)[0]
        # max_time = tf.shape(x)[1]
        max_time = self.config.dataloader.encode_length # DIFFERENT!

        squeezed_x = tf.reshape(x, [batch_size * max_time, 64, 64, 2])

        hidden_length = 64
        input_channels = 2

        conv_output, hidden_length, input_channels = self._cnn_encoder_block(
                squeezed_x, hidden_length, input_channels)
        
        conv_output_expand = tf.reshape(
            conv_output, 
            [batch_size, max_time, hidden_length, hidden_length, input_channels])

        rnn_output, hidden_length, input_channels = self._rnn_block_wrapped(
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
