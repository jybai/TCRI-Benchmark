import numpy as np
import tensorflow as tf
import yaml
from attrdict import AttrDict

from Utils.attention_2d_wrapper import Luong2DAttentionMechanism

from model_sa import Model as SaModel

model_name = 'ConvLSTM'

class Model(SaModel):
    # content straight copy from seq
    def convlstm_layer(self, config, hidden_length, input_channels,
                       memory, memory_sequence_length):
# {{{
        def cell_input_fn(inputs, attention):
            attention_2d = tf.reshape(attention, tf.shape(inputs))
            return tf.concat([inputs, attention_2d], axis=-1)

        attention_mechanism = Luong2DAttentionMechanism(
            num_units=config.seqattn.hidden_size,
            filter_size=[config.seqattn.filter_size] * 2,
            query_shape=[hidden_length, hidden_length, input_channels],
            memory=memory,
            memory_sequence_length=memory_sequence_length)

        stacked_lstm, input_channels = super().convlstm_layer(config, 
                                                              hidden_length, 
                                                              input_channels)

        attn_lstm = tf.contrib.seq2seq.AttentionWrapper(
            stacked_lstm,
            attention_mechanism,
            cell_input_fn=cell_input_fn,
            output_attention=False)

        return attn_lstm, input_channels
# }}}

    def _rnn_block(self, rnn_input, seqlen, batch_size, hidden_length, input_channels):
# {{{
        cell, input_channels = self.convlstm_layer(self.config.lstm, hidden_length, 
                                                   input_channels, rnn_input, seqlen)

        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_output, _ = tf.nn.dynamic_rnn(
                cell,
                rnn_input,
                sequence_length=seqlen,
                initial_state=initial_state,
                time_major=False,
                dtype=tf.float32)
        t = tf.shape(rnn_input)[1]
        rnn_output = tf.reshape(rnn_output, [-1, t, hidden_length, hidden_length, input_channels])
        return rnn_output, hidden_length, input_channels
# }}}


