import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.layers import core as layers_core
from tensorflow.python.layers import convolutional as layers_conv

from tensorflow.contrib.image.python.ops import image_ops
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

from utils.stn import spatial_transformer_network as stn

class Luong2DAttentionMechanism(attention_wrapper._BaseAttentionMechanism):
# {{{
    def __init__(self, num_units, filter_size, query_shape, memory, 
            memory_sequence_length=None, scale=False, probability_fn=None, 
            score_mask_value=None, mask=True, shift=None, dtype=None, name="Luong2DAttention"):
        if mask is False and shift is not None:
            raise ValueError(
                    "Shift will only work when mask is True.")
        if probability_fn is None:
            probability_fn = _masked_probability_fn if mask else \
                    lambda score, _: nn_ops.softmax(score)
        if dtype is None:
            dtype = dtypes.float32
        if shift is None:
            shift = 0
        # wrapped_probability_fn = lambda score, _: probability_fn(score)

        query_layer = layers_conv.Conv2D(num_units, kernel_size=filter_size, 
                use_bias=False, padding='SAME', dtype=dtype, name="query_layer")

        memory_layer = layers_conv.Conv3D(num_units, kernel_size=[1] + filter_size, 
                use_bias=False, padding='SAME', dtype=dtype, name="memory_layer")

        super(Luong2DAttentionMechanism, self).__init__(
                query_layer=query_layer,
                memory_layer=memory_layer,
                memory=memory,
                # probability_fn=wrapped_probability_fn,
                probability_fn=probability_fn,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value,
                name=name)

        self._num_units = num_units
        self._filter_size = filter_size
        if len(query_shape) != 3:
            raise ValueError(
                    "Query shape is required to be [H, W, C]")
        self._query_shape = query_shape
        self._scale = scale
        self._mask = mask
        self._shift = shift
        self._name = name

    @property
    def values(self):
        shape = array_ops.shape(self._values)
        hsize = 1
        for s in self._query_shape:
            hsize *= s
        t = shape[1]
        return gen_array_ops.reshape(self._values, [-1, t, hsize])

    def initial_alignments(self, bsize, dtype):
        t = self._alignments_size
        s = self._shift
        if s == 0 or not self._mask:
            return super().initial_alignments(bsize, dtype)
        # s - 1 because before the first attention.
        init_expose_len = array_ops.ones([bsize], dtype=dtypes.int32) * (s - 1)
        clipped_expose_len = gen_math_ops.minimum(init_expose_len, t)
        uniform_init_alignments = array_ops.sequence_mask(clipped_expose_len, 
                t, dtype=dtype) / (s - 1)
        return uniform_init_alignments

    def __call__(self, query, state):
        """
        query: [batch_size, H, W, query_depth]
        state: unused
        return: [batch_size, max_time]
        """
        with vs.variable_scope(None, "luong_2d_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _luong2d_score(processed_query, self._keys, self._scale)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state
# }}}

def _luong2d_score(query, keys, scale):
# {{{
    """
    query: [batch_size, H, W, num_units]
    keys: [batch_size, max_time, H, W, num_units]
    return: [batch_size, max_time]
    """
    qh, qw, qd = query.get_shape()[-3:]
    kh, kw, kd = keys.get_shape()[-3:]
    # max_time = keys.get_shape()[1]
    max_time = array_ops.shape(keys)[1]

    if qh != kh or qw != kw or qd != kd:
        raise ValueError(
                "Query(%d, %d, %d) and keys(%d, %d, %d) must match hidden_dims for Luong Score"
                %(qh, qw, qd, kh, kw, kd))
    dtype = query.dtype
    
    query = gen_array_ops.reshape(query, [-1, 1, qh * qw * qd])
    keys = gen_array_ops.reshape(keys, [-1, max_time, kh * kw * kd])

    score = math_ops.matmul(query, keys, transpose_b=True)
    score = array_ops.squeeze(score, [1])

    if scale:
        g = vs.get_variable("attention_g", dtype=dtype,
                initializer=init_ops.ones_initializer, shape=())
        score = g * score

    return score
# }}}

def _masked_probability_fn(score, prev_alignments):
# {{{
    t = array_ops.shape(score)[1]
    prev_mask = math_ops.cast(prev_alignments > 0, dtypes.int32)
    prev_expose_len = math_ops.reduce_sum(prev_mask, 1)
    
    new_expose_len = gen_math_ops.minimum(prev_expose_len + 1, t)
    new_mask = array_ops.sequence_mask(
            new_expose_len, maxlen=t, dtype=dtypes.float32)
    inf_mask = (1 - new_mask) * np.finfo(np.float32).min
    new_alignments = nn_ops.softmax(score * new_mask + inf_mask)
    return new_alignments
# }}}

def _circle_mask(d):
    r = int(d/2)
    center = [r, r]
    Y, X = np.ogrid[:d, :d]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= r
    return math_ops.cast(mask, dtypes.float32)

def _rotate(inputs, angles):
    # inputs = [B, H, W, C]
    # angles = [B]
    # params = [a, b, c, d, e, f]
    # | a, b, c |
    # | d, e, f |

    cos = gen_math_ops.cos(angles)
    sin = gen_math_ops.sin(angles)
    zeros = array_ops.zeros_like(angles)
    params = array_ops.stack([cos, -sin, zeros, sin, cos, zeros], axis=-1)
    return stn(inputs, params)

def _reverse_rotate_memory(memory, angle):
    # angle = (bsize, t)
    shape = array_ops.shape(memory)
    memory_flat = gen_array_ops.reshape(memory, array_ops.concat([[-1], shape[-3:]], 0))
    angle_flat = gen_array_ops.reshape(angle, [-1])
    rr_memory = _rotate(memory_flat, -1 * angle_flat)
    # rr_memory = image_ops.rotate(memory_flat, -1 * angle_flat)
    rr_memory_expand = gen_array_ops.reshape(rr_memory, shape)
    return rr_memory_expand

class AngleAdjustedLuong2DAttentionMechanism(Luong2DAttentionMechanism):
    def __init__(self, num_units, filter_size, query_shape, angle, memory, 
            memory_sequence_length=None, scale=False, probability_fn=None, 
            score_mask_value=None, mask=True, shift=None, dtype=None, 
            name="AngleAdjustedLuong2DAttention"):

        super(AngleAdjustedLuong2DAttentionMechanism, self).__init__(
                num_units=num_units,
                filter_size=filter_size,
                query_shape=query_shape,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                scale=scale,
                probability_fn=probability_fn,
                score_mask_value=score_mask_value,
                mask=mask,
                shift=shift,
                dtype=dtype,
                name=name)

        # generate angle adjusted memory.
        cmask = _circle_mask(query_shape[0])
        cmask = array_ops.expand_dims(cmask, -1)
        self._angle_adjusted_keys = _reverse_rotate_memory(
                self._keys, angle) * cmask

    def __call__(self, query, state):
        """
        query: [batch_size, H, W, query_depth]
        state: unused
        return: [batch_size, max_time]
        """
        # query must be angle adjusted already.
        with vs.variable_scope(None, "luong_2d_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _luong2d_score(processed_query, self._angle_adjusted_keys, self._scale)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

def _compute_attention(attention_mechanism, cell_output, attention_state,
        angle, attention_layer):
    cell_output_shape = array_ops.shape(cell_output)
    angle_adjusted_cell_output = _rotate(cell_output, -1 * angle)
    # angle_adjusted_cell_output = image_ops.rotate(cell_output, -1 * angle)
    angle_adjusted_cell_output = gen_array_ops.reshape(angle_adjusted_cell_output, cell_output_shape)
    # hack because cell_output lose shape after image transform.
    alignments, next_attention_state = attention_mechanism(
        angle_adjusted_cell_output, state=attention_state)

    expanded_alignments = array_ops.expand_dims(alignments, 1)
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state


class AngleAdjustedAttentionWrapper(attention_wrapper.AttentionWrapper):
    def __init__(self, cell, angle, attention_mechanism, 
            attention_layer_size=None, alignment_history=False, 
            cell_input_fn=None, output_attention=True,
            initial_cell_state=None, name=None, attention_layer=None):

        super(AngleAdjustedAttentionWrapper, self).__init__(
                cell=cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=attention_layer_size,
                alignment_history=alignment_history,
                cell_input_fn=cell_input_fn,
                output_attention=output_attention,
                initial_cell_state=initial_cell_state,
                name=name,
                attention_layer=attention_layer)

        self._angle = angle

    def call(self, inputs, state):
        # almost the same as original call except _compute_attention
        if not isinstance(state, attention_wrapper.AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead."  % type(state))
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        
        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []

        # get angle.
        angle = self._angle[:, state.time]

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                angle, # plug in angle for reverse rotation.
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        self._alignments = alignment_history.stack()

        attention = array_ops.concat(all_attentions, 1)
        next_state = attention_wrapper.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))


        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

