# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)


def _gru_encoder(cell, inputs, sequence_length, initial_state, dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


def _encoder(cell_fw, cell_bw, inputs, sequence_length, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length]):
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)

        with tf.variable_scope("forward"):
            output_fw, state_fw = _gru_encoder(cell_fw, inputs_fw,
                                               sequence_length, None,
                                               dtype=dtype)

        with tf.variable_scope("backward"):
            output_bw, state_bw = _gru_encoder(cell_bw, inputs_bw,
                                               sequence_length, None,
                                               dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)

        results = {
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            }
        }

        return results


def _decoder(cell_cond, cell, inputs, memory, sequence_length, initial_state, incre_state=None, dtype=None,
             scope=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    #zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)
    zero_value = tf.zeros([batch, output_size], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=tf.float32)
        bias = layers.attention.attention_bias(mem_mask, "masking")
        bias = tf.squeeze(bias, axis=[1])
        #bias = tf.squeeze(bias, axis=[1, 2])
        cache = layers.attention.attention_mhead(None, memory, None, output_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
	outputstate_ta = tf.TensorArray(tf.float32, time_steps,
                                        tensor_array_name="outputstate_array")
        value_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="value_array")
        alpha_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="alpha_array")
        input_ta = input_ta.unstack(inputs)

        if incre_state is not None:
	    initial_state = incre_state

        def loop_func(t, out_ta, att_ta, val_ta, state, outstate_ta, cache_key):
            inp_t = input_ta.read(t)
            output1, state1 = cell_cond(inp_t, state)
            state1 = _copy_through(t, sequence_length["target"], state,
                                   state1)
            results = layers.attention.attention_mhead(state1, memory, bias,
                                                       output_size,
                                                       cache={"key": cache_key})

            alpha = results["weight"]
            context = results["value"]
            cell_input = [context]
            cell_output, new_state = cell(cell_input, state1)

            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state1,
                                      new_state)
            new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)

            out_ta = out_ta.write(t, cell_output)
            outstate_ta = outstate_ta.write(t, new_state)
            att_ta = att_ta.write(t, alpha)
            val_ta = val_ta.write(t, new_value)
            cache_key = tf.identity(cache_key)
            return t + 1, out_ta, att_ta, val_ta, new_state, outstate_ta, cache_key

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, alpha_ta, value_ta, initial_state, outputstate_ta,
                     cache["key"])

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        state_final_ta = outputs[5]
        value_final_ta = outputs[3]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

        final_state = state_final_ta.stack()
        final_state.set_shape([None, None, output_size])
        final_state = tf.transpose(final_state, [1, 0, 2])

        final_value = value_final_ta.stack()
        #final_value.set_shape([None, None, memory.shape[-1].value])
        final_value.set_shape([None, None, output_size])
        final_value = tf.transpose(final_value, [1, 0, 2])

        result = {
            "outputs": final_output,
            "values": final_value,
            "state": final_state[:,-1,:]
        }

    return result


def encoding_graph(features, mode, params):

    if mode != "train":
	params.dropout = 0.0
	params.rnn_dropout = 0.0
        params.use_variational_dropout = False
	params.label_smoothing = 0.0

    src_vocab_size = len(params.vocabulary["source"])

    with tf.variable_scope("source_embedding"):
        src_emb = tf.get_variable("embedding",
                                  [src_vocab_size, params.embedding_size])
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    src_inputs = layers.attention.add_timing_signal(src_inputs)

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)

    # encoder
    cell_fw = layers.rnn_cell.DL4MTGRULAUTransiLNCell(params.hidden_size, 1.0 - params.rnn_dropout)
    cell_bw = layers.rnn_cell.DL4MTGRULAUTransiLNCell(params.hidden_size, 1.0 - params.rnn_dropout)

    if params.use_variational_dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell_fw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell_bw,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            input_size=params.embedding_size,
            dtype=tf.float32
        )

    encoder_output = _encoder(cell_fw, cell_bw, src_inputs,
                              features["source_length"])

    src_mask = tf.sequence_mask(features["source_length"],
                                maxlen=tf.shape(encoder_output["annotation"])[1],
                                dtype=tf.float32)
    src_mask = tf.expand_dims(src_mask, axis=2)
    initial_state = tf.reduce_sum(encoder_output["annotation"] * src_mask, axis=1) / tf.reduce_sum(src_mask, axis=1)
    initial_state = layers.nn.linear(initial_state, cell_fw.output_size, True,
                                     False, scope="s_transform")
    initial_state = tf.tanh(initial_state)

    return encoder_output, initial_state


def decoding_graph(features, state, mode, params):

    if mode != "train":
	params.dropout = 0.0
	params.rnn_dropout = 0.0
        params.use_variational_dropout = False
	params.label_smoothing = 0.0

    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.variable_scope("target_embedding"):
        tgt_emb = tf.get_variable("embedding",
                                  [tgt_vocab_size, params.embedding_size])
        tgt_bias = tf.get_variable("bias", [params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(tgt_emb, features["target"])

    tgt_inputs = tf.nn.bias_add(tgt_inputs, tgt_bias)
    tgt_inputs = layers.attention.add_timing_signal(tgt_inputs)

    if params.dropout and not params.use_variational_dropout:
        tgt_inputs = tf.nn.dropout(tgt_inputs, 1.0 - params.dropout)

    encoder_output = state["encoder"]
    initial_state = state["initstate"]

    # decoder
    cell = layers.rnn_cell.DL4MTGRULAUTransiLNCell(params.hidden_size, 1.0 - params.rnn_dropout)
    cell_cond = layers.rnn_cell.DL4MTGRULAUTransiLNCell(params.hidden_size, 1.0 - params.rnn_dropout)

    if params.use_variational_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1.0 - params.dropout,
            output_keep_prob=1.0 - params.dropout,
            state_keep_prob=1.0 - params.dropout,
            variational_recurrent=True,
            # input + context
            input_size=params.embedding_size + 2 * params.hidden_size,
            dtype=tf.float32
        )

    length = {
        "source": features["source_length"],
        "target": features["target_length"]
    }
   
    shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
    shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]
    maxout_size = params.hidden_size // params.maxnum

    if mode != "infer":
    	decoder_output = _decoder(cell_cond, cell, shifted_tgt_inputs, encoder_output["annotation"],
                              	  length, initial_state)
    else:
    	# Shift left
    	shifted_tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
        shifted_tgt_inputs = shifted_tgt_inputs[:, :-1, :]
	decoder_output = _decoder(cell_cond, cell, shifted_tgt_inputs[:, -1:, :], encoder_output["annotation"],
                                  length, initial_state, incre_state=state["decoder"])
	shifted_outputs = decoder_output["outputs"]

	maxout_features = [
            shifted_tgt_inputs[:, -1, :],
            shifted_outputs[:, -1, :],
            decoder_output["values"][:, -1, :]
        ]

	readout = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                                   concat=False)
        readout = tf.tanh(readout)

        # Prediction
        logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                                  scope="softmax")

	log_prob = tf.nn.log_softmax(logits)

	return log_prob, {"encoder": encoder_output, "initstate": initial_state, "decoder": decoder_output["state"]}


    shifted_outputs = decoder_output["outputs"]

    maxout_features = [
        shifted_tgt_inputs,
        shifted_outputs,
        decoder_output["values"]
    ]

    readout = layers.nn.maxout(maxout_features, maxout_size, params.maxnum,
                               concat=False)
    readout = tf.tanh(readout)

    if params.dropout and not params.use_variational_dropout:
        readout = tf.nn.dropout(readout, 1.0 - params.dropout)

    # Prediction
    logits = layers.nn.linear(readout, tgt_vocab_size, True, False,
                              scope="softmax")
    logits = tf.reshape(logits, [-1, tgt_vocab_size])
    labels = features["target"]

    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(labels))
    tgt_mask = tf.to_float(
        tf.sequence_mask(
            features["target_length"],
            maxlen=tf.shape(features["target"])[1]
        )
    )

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def model_graph(features, mode, params):

    encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output[0], 
	"initstate": encoder_output[1]
    }
    output = decoding_graph(features, state, mode, params)

    return output


class RNNsearch(interface.NMTModel):

    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearch, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):

        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
        	batch = tf.shape(encoder_output[0]["annotation"])[0]
                state = {
                    "encoder": encoder_output[0],
                    "initstate": encoder_output[1],
		    "decoder": encoder_output[1]
                }
            return state
	
	def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "rnnsearch"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<eos>",
            append_eos=False,
            # model
            rnn_cell="DL4MTGRULAUTransiLNCell",
            embedding_size=1024,
            hidden_size=1024,
            maxnum=1,
            # regularization
            dropout=0.5,
	    rnn_dropout=0.3,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=100,
            clip_grad_norm=5.0
        )

        return params

