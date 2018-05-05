import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention, AttentionWrapper, TrainingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode, BeamSearchDecoder, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import tile_batch
from dataset.data_prepro import GO, EOS
from tqdm import tqdm


class Chatbot:
    def __init__(self, config, model_name="Seq2SeqChatbot"):
        self.cfg = config
        self.logger = config.logger
        self.model_name = model_name
        self.sess, self.saver = None, None
        self.merged_summaries, self.summary_writer = None, None
        self._add_placeholders()
        self._build_model()
        self._build_decode_op()
        self._build_loss_op()
        self._build_train_op()
        self.logger.info('number of trainable parameters: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in
                                                                             tf.trainable_variables()])))
        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        else:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, step):
        self.saver.save(self.sess, self.cfg.ckpt_path + self.model_name, global_step=step)

    def close_session(self):
        self.sess.close()

    def reinitialize_weights(self, scope_name=None):
        if scope_name is not None:  # reinitialize weights in the given scope name
            variables = tf.contrib.framework.get_variables(scope_name)
        else:  # reinitialize all weights
            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        self.sess.run(tf.variables_initializer(variables))

    def _add_summary(self):
        self.merged_summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.cfg.summary_dir, self.sess.graph)

    def _add_placeholders(self):
        # shape = (batch_size, max_words_len)
        self.enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="encoder_input")
        self.dec_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_input")
        # shape = (batch_size)
        self.enc_seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="encoder_seq_length")
        self.dec_seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="decoder_seq_length")
        # hyper-parameters
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(dtype=tf.float32, name='learning_rate')

    def _get_feed_dict(self, batch_data, keep_prob=None, lr=None):
        feed_dict = {self.enc_input: batch_data["enc_input"], self.dec_input: batch_data["dec_input"],
                     self.enc_seq_len: batch_data["enc_seq_len"], self.dec_seq_len: batch_data["dec_seq_len"],
                     self.batch_size: len(batch_data["enc_input"])}
        if keep_prob is not None:
            feed_dict[self.keep_prob] = keep_prob
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _build_model(self):
        with tf.variable_scope("embeddings"):
            self.embeddings = tf.get_variable(name='embeddings', shape=[self.cfg.vocab_size, self.cfg.emb_dim],
                                              dtype=tf.float32, trainable=True)
            self.encoder_emb = tf.nn.embedding_lookup(self.embeddings, self.enc_input)

        with tf.variable_scope("encoder"):
            enc_cells = MultiRNNCell([DropoutWrapper(LSTMCell(self.cfg.num_units), output_keep_prob=self.keep_prob)] *
                                     self.cfg.num_layers)
            enc_outputs, enc_states = dynamic_rnn(enc_cells, self.encoder_emb, sequence_length=self.enc_seq_len,
                                                  dtype=tf.float32)
            enc_seq_len = self.enc_seq_len
            if self.cfg.use_beam_search:
                enc_outputs = tile_batch(enc_outputs, multiplier=self.cfg.beam_size)
                enc_states = nest.map_structure(lambda s: tile_batch(s, self.cfg.beam_size), enc_states)
                enc_seq_len = tile_batch(enc_seq_len, multiplier=self.cfg.beam_size)

        with tf.variable_scope("attention"):
            if self.cfg.attention == "Bahdanau":  # Bahdanau attention mechanism
                attention_mechanism = BahdanauAttention(num_units=self.cfg.num_units, memory=enc_outputs,
                                                        memory_sequence_length=enc_seq_len)
            elif self.cfg.attention == "Luong":  # Luong attention mechanism
                attention_mechanism = LuongAttention(num_units=self.cfg.num_units, memory=enc_outputs,
                                                     memory_sequence_length=enc_seq_len)
            else:  # default using Bahdanau attention mechanism
                attention_mechanism = BahdanauAttention(num_units=self.cfg.num_units, memory=enc_outputs,
                                                        memory_sequence_length=enc_seq_len)

        with tf.variable_scope("decoder"):
            self.max_dec_seq_len = tf.reduce_max(self.dec_seq_len, name="max_dec_seq_len")
            dec_cells = MultiRNNCell([DropoutWrapper(LSTMCell(self.cfg.num_units), output_keep_prob=self.keep_prob)] *
                                     self.cfg.num_units)
            self.dec_cells = AttentionWrapper(cell=dec_cells, attention_mechanism=attention_mechanism,
                                              attention_layer_size=self.cfg.num_units, name='attention_wrapper')
            # re-allocate batch_size if using beam search strategy
            batch_size = self.batch_size * self.cfg.beam_size if self.cfg.use_beam_search else self.batch_size
            # define output layer
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            self.dense_layer = tf.layers.Dense(units=self.cfg.vocab_size, kernel_initializer=initializer)
            # use the last state of encoder as the initial state of decoder
            self.dec_init_states = self.dec_cells.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
                cell_state=enc_states)
            # for training, remove the "<EOS>" token from the end and add the "<GO>" token at the beginning
            dec_input = tf.strided_slice(self.dec_input, [0, 0], [self.batch_size, -1], [1, 1])
            dec_input = tf.concat([tf.fill([self.batch_size, 1], self.cfg.word_dict[GO]), dec_input], axis=1)
            decoder_emb = tf.nn.embedding_lookup(self.embeddings, dec_input)
            train_helper = TrainingHelper(decoder_emb, sequence_length=self.dec_seq_len, name="train_helper")
            train_decoder = BasicDecoder(self.dec_cells, helper=train_helper, output_layer=self.dense_layer,
                                         initial_state=self.dec_init_states)
            self.dec_output, _, _ = dynamic_decode(train_decoder, impute_finished=True,
                                                   maximum_iterations=self.max_dec_seq_len)

    def _build_decode_op(self):
        start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * self.cfg.word_dict[GO]
        end_token = self.cfg.word_dict[EOS]
        if self.cfg.use_beam_search:
            infer_decoder = BeamSearchDecoder(self.dec_cells, self.embeddings, start_tokens=start_token,
                                              end_token=end_token, initial_state=self.dec_init_states,
                                              beam_width=self.cfg.beam_size, output_layer=self.dense_layer)
        else:
            dec_helper = GreedyEmbeddingHelper(self.embeddings, start_token, end_token)
            infer_decoder = BasicDecoder(self.dec_cells, helper=dec_helper, initial_state=self.dec_init_states,
                                         output_layer=self.dense_layer)
        infer_dec_output, _, _ = dynamic_decode(infer_decoder, maximum_iterations=10)
        if self.cfg.use_beam_search:
            self.dec_predicts = infer_dec_output.predicted_ids
        else:
            self.dec_predicts = tf.expand_dims(infer_dec_output.sample_id, axis=-1)

    def _build_loss_op(self):
        dec_logits = tf.identity(self.dec_output.rnn_output)
        dec_mask = tf.sequence_mask(self.dec_seq_len, self.max_dec_seq_len, dtype=tf.float32, name="dec_mask")
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=dec_logits, targets=self.dec_input,
                                                     weights=dec_mask)
        tf.summary.scalar('loss', self.loss)

    def _build_train_op(self):
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, dataset, epochs, shuffle=True):
        self.logger.info("Start training...")
        self._add_summary()
        cur_step = 0
        for epoch in range(1, epochs + 1):
            if shuffle:
                random.shuffle(dataset)
            for batch_data in tqdm(dataset, desc="Epoch {}:".format(epoch)):
                cur_step += 1
                feed_dict = self._get_feed_dict(batch_data, keep_prob=self.cfg.keep_prob, lr=self.lr)
                _, loss, summary = self.sess.run([self.train_op, self.loss, self.merged_summaries], feed_dict=feed_dict)
                if cur_step % 10 == 0:
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (cur_step, loss, perplexity))
                    self.summary_writer.add_summary(summary, cur_step)

    def evaluate(self):
        pass

    def inference(self, data):
        feed_dict = self._get_feed_dict(data, keep_prob=1.0)
        predicts = self.sess.run([self.dec_predicts], feed_dict=feed_dict)
        pass


