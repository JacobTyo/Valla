import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
from math import ceil
from sklearn.utils import shuffle
from eval_metrics import evaluate_all


class AdHominem_flat():
    """
        AdHominem describes a Siamese network topology for (binary) authorship verification, also known as pairwise
        (1:1) forensic text comparison.

        References:
        [1] Benedikt Boenninghoff, Robert M. Nickel, Steffen Zeiler, Dorothea Kolossa, 'Similarity Learning for
            Authorship Verification in Social Media', IEEE ICASSP 2019.
        [2] Benedikt Boenninghoff, Steffen Hessler, Dorothea Kolossa, Robert M. Nickel 'Explainable Authorship
            Verification in Social Media via Attention-based Similarity Learning', IEEE BigData 2019.
    """
    def __init__(self, hyper_parameters, E_w_init):

        # reset graph
        tf.reset_default_graph()

        # hyper-parameters
        self.hyper_parameters = hyper_parameters

        # placeholders for input variables
        self.placeholders, self.thetas_E = self.initialize_placeholders(E_w_init)

        # batch size
        self.B = tf.shape(self.placeholders['e_w_L'])[0]

        # trainable parameters
        self.theta = self.initialize_parameters()
        # initialize dropout
        self.dropout = self.initialize_dropout()

        with tf.variable_scope('feature_extraction_doc2vec'):
            e_c = tf.concat([self.placeholders['e_c_L'], self.placeholders['e_c_R']], axis=0)
            e_w = tf.concat([self.placeholders['e_w_L'], self.placeholders['e_w_R']], axis=0)
            N_w = tf.concat([self.placeholders['N_w_L'], self.placeholders['N_w_R']], axis=0)
            N_s = tf.concat([self.placeholders['N_s_L'], self.placeholders['N_s_R']], axis=0)
            # document embeddings
            e_d = self.feature_extraction(e_c, e_w, N_w, N_s)

        with tf.variable_scope('metric_learning'):
            # author2vec
            self.features_L, self.features_R = self.metric_layer(e_d)

        with tf.variable_scope('Euclidean_distance_and_kernel_function'):
            self.distance = self.compute_distance()
            self.pred = self.kernel_function()

        # loss function
        if self.hyper_parameters['loss'] == 'contrastive':
            self.loss = self.contrastive_loss_function()
        elif self.hyper_parameters['loss'] == 'modified_contrastive':
            self.loss = self.loss_function()
        else:
            tmp = self.hyper_parameters['loss']
            assert False, f'{tmp} is not defined as a loss function'

        # optimizer
        self.optimizer, self.step = self.prepare_optimizer()

        # launch session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    ###################
    # prepare optimizer
    ###################
    def prepare_optimizer(self):

        # global step counter
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.placeholders['lr'])

        # local gradient normalization
        grads_and_vars = opt.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.0), var) if grad is not None and var is not None else (grad, var) for grad, var in grads_and_vars]
        optimizer = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)

        return optimizer, global_step

    ################################################
    # initialize all placeholders and look-up tables
    ################################################
    def initialize_placeholders(self, E_w_init):

        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        D_c = self.hyper_parameters['D_c']
        D_w = self.hyper_parameters['D_w']
        V_c = self.hyper_parameters['V_c']
        train_word_embeddings = self.hyper_parameters["train_word_embeddings"]

        # input character placeholder
        x_c_L = tf.placeholder(dtype=tf.int32,
                               shape=[None, T_s, T_w, T_c],
                               name='x_c_L',
                               )
        x_c_R = tf.placeholder(dtype=tf.int32,
                               shape=[None, T_s, T_w, T_c],
                               name='x_c_R',
                               )

        # initialize embedding matrix for characters
        with tf.variable_scope('character_embedding_matrix'):
            # zero-padding embedding
            E_c_0 = tf.zeros(shape=[1, D_c], dtype=tf.float32)
            # trainable embeddings
            r = 0.1
            E_c_1 = tf.get_variable(name='E_c_1',
                                    shape=[len(V_c) - 1, D_c],
                                    initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                    trainable=True,
                                    dtype=tf.float32,
                                    )
            # concatenate special-token embeddings + regular-token embeddings
            E_c = tf.concat([E_c_0, E_c_1], axis=0)

        # character embeddings, shape=[B, T_s, T_w, T_c, D_c]
        e_c_L = tf.nn.embedding_lookup(E_c, x_c_L)
        e_c_R = tf.nn.embedding_lookup(E_c, x_c_R)

        # word-based placeholder for two documents
        x_w_L = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_L')
        x_w_R = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_R')

        # true sentence / document lengths
        N_w_L = tf.placeholder(dtype=tf.int32, shape=[None, T_s], name='N_w_L')
        N_w_R = tf.placeholder(dtype=tf.int32, shape=[None, T_s], name='N_w_R')
        N_s_L = tf.placeholder(dtype=tf.int32, shape=[None], name='N_s_L')
        N_s_R = tf.placeholder(dtype=tf.int32, shape=[None], name='N_s_R')

        # matrix for word embeddings, shape=[len(V_w), D_w]
        with tf.variable_scope("word_embedding_matrix"):
            # zero-padding embedding
            E_w_0 = tf.zeros(shape=[1, D_w], dtype=tf.float32)
            # trainable special tokens
            E_w_1 = tf.Variable(E_w_init[1:6, :],
                                name='E_trainable_special_tokens',
                                trainable=True,
                                dtype=tf.float32,
                                )
            # pre-trained word embeddings
            E_w_2 = tf.Variable(E_w_init[6:, :],
                                name='E_pretrained_tokens',
                                trainable=train_word_embeddings,
                                dtype=tf.float32,
                                )
            # concatenate special-token embeddings + regular-token embeddings
            E_w = tf.concat([E_w_0, E_w_1, E_w_2], axis=0)

        # word embeddings, shape=[B, T_s, T_w, D_w]
        e_w_L = tf.nn.embedding_lookup(E_w, x_w_L)
        e_w_R = tf.nn.embedding_lookup(E_w, x_w_R)

        ####################
        # training variables
        ####################
        # labels
        labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
        # training mode (for dropout regularization)
        is_training = tf.placeholder(dtype=tf.bool, name='training_mode')
        # learning rate
        lr = tf.placeholder(tf.float32, [], name='lr')

        #############
        # make tuples
        #############
        placeholders = {'x_c_L': x_c_L,
                        'x_c_R': x_c_R,
                        #
                        'e_c_L': e_c_L,
                        'e_c_R': e_c_R,
                        #
                        'x_w_L': x_w_L,
                        'x_w_R': x_w_R,
                        #
                        'e_w_L': e_w_L,
                        'e_w_R': e_w_R,
                        #
                        'N_w_L': N_w_L,
                        'N_w_R': N_w_R,
                        'N_s_L': N_s_L,
                        'N_s_R': N_s_R,
                        #
                        'labels': labels,
                        'is_training': is_training,
                        'lr': lr,
                        }

        thetas_E = {'E_c_1': E_c_1,
                    'E_w_1': E_w_1,
                    }
        if train_word_embeddings:
            thetas_E['E_w_2'] = E_w_2

        return placeholders, thetas_E

    ################################################
    # feature extraction: words-to-document encoding
    ################################################
    def feature_extraction(self, e_c, e_w, N_w, N_s):

        with tf.variable_scope('characters_to_word_encoding'):
            r_c = self.cnn_layer_cw(e_c)
        with tf.variable_scope('words_to_sentence_encoding'):
            e_cw = tf.concat([e_w, r_c], axis=3)
            h_w = self.bilstm_layer_ws(e_cw, N_w)
            e_s = self.att_layer_ws(h_w, N_w)
        # with tf.variable_scope('sentences_to_document_encoding'):
        #     h_s = self.bilstm_layer_sd(e_s, N_s)
        #     e_d = self.att_layer_sd(h_s, N_s)

        return e_s

    ##########################################
    # compute distance between feature vectors
    ##########################################
    def compute_distance(self):
        # define euclidean distance, shape = (B, D_h)
        distance = tf.subtract(self.features_L, self.features_R)
        distance = tf.square(distance) + 1e-10
        # shape = (B, 1)
        distance = tf.reduce_sum(distance, 1, keepdims=True)
        distance = tf.sqrt(distance)
        return distance

    def kernel_function(self):
        pred = tf.math.pow(x=self.distance, y=3)
        pred = tf.math.exp(-0.09 * pred)
        pred = pred + 1e-10
        return pred

    #####################
    # estimate new labels
    #####################
    def compute_labels(self, labels, pred):
        # threshold
        th = 0.5  # = (t_s + t_d)/2
        # numpy array for estimated labels
        labels_hat = np.ones(labels.shape, dtype=np.float32)
        # dissimilar pairs --> 0, similar pairs --> 1
        labels_hat[pred < th] = 0
        return labels_hat

    ###############
    # loss function
    ###############
    def loss_function(self):

        t_s = self.hyper_parameters['t_s']
        t_d = self.hyper_parameters['t_d']
        labels = self.placeholders['labels']
        pred = self.pred

        # define contrastive loss:
        l1 = tf.multiply(tf.subtract(1.0, labels), tf.square(tf.maximum(tf.subtract(pred, t_d), 0.0)))
        l2 = tf.multiply(labels, tf.square(tf.maximum(tf.subtract(t_s, pred), 0.0)))
        loss = tf.add(l1, l2)
        loss = tf.reduce_mean(loss)

        return loss

    ##################################
    # normal contrastive loss function
    ##################################
    def contrastive_loss_function(self):
        t_s = self.hyper_parameters['t_s']
        labels = self.placeholders['labels']
        pred = self.pred

        # define contrastive loss:
        l1 = tf.multiply(tf.subtract(1.0, labels), tf.square(tf.maximum(tf.subtract(pred, t_s), 0.0)))
        l2 = tf.multiply(labels, tf.square(tf.maximum(tf.subtract(t_s, pred), 0.0)))
        loss = tf.add(l1, l2)
        loss = tf.reduce_mean(loss)

        return loss

    ###############################
    # MLP layer for metric learning
    ###############################
    def metric_layer(self, e_d):

        is_training = self.placeholders['is_training']
        dropout_mask = tf.concat([self.dropout['metric']['x'],
                                  self.dropout['metric']['x']],
                                 axis=0)

        # apply dropout
        y = tf.cond(tf.equal(is_training, tf.constant(True)),
                    lambda: tf.multiply(dropout_mask, e_d),
                    lambda: e_d,
                    )
        # fully-connected layer
        y = tf.nn.xw_plus_b(y,
                            self.theta['metric']['W'],
                            self.theta['metric']['b'],
                            )
        # nonlinear output
        y = tf.nn.tanh(y)

        return y[:self.B, :], y[self.B:, :]

    ########################################
    # 1D-CNN for characters-to-word encoding
    ########################################
    def cnn_layer_cw(self, e_c):

        T_s = self.hyper_parameters['T_s']
        T_w = self.hyper_parameters['T_w']
        T_c = self.hyper_parameters['T_c']
        h = self.hyper_parameters['w']
        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']

        is_training = self.placeholders['is_training']
        dropout_mask = tf.concat([self.dropout['cnn'],
                                  self.dropout['cnn']],
                                 axis=0)

        # dropout and zero-padding
        # reshape: [2 * B, T_s, T_w, T_c, D_c] --> [2 * B * T_s * T_w, T_c, D_c]
        e_c = tf.reshape(e_c, shape=[2 * self.B * T_s * T_w, T_c, D_c])
        # dropout
        e_c = tf.cond(tf.equal(is_training, tf.constant(True)),
                      lambda: tf.multiply(dropout_mask, e_c),
                      lambda: e_c,
                      )
        # zero-padding, shape = [2 * B * T_s * T_w, T_c + 2 * (h-1), D_c]
        e_c = tf.pad(e_c,
                     tf.constant([[0, 0], [h - 1, h - 1], [0, 0]]),
                     mode='CONSTANT',
                     )

        # 1D convolution
        # shape = [2 * B * T_s * T_w, T_c + 2 * (h-1) - h + 1, D_r] = [2 * B * T_s * T_w, T_c + h - 1, D_r]
        r_c = tf.nn.conv1d(e_c,
                           self.theta['cnn']['W'],
                           stride=1,
                           padding='VALID',
                           name='chraracter_1D_cnn',
                           )
        # apply bias term
        r_c = tf.nn.bias_add(r_c, self.theta['cnn']['b'])
        # apply nonlinear function
        r_c = tf.nn.tanh(r_c)

        # max-over-time pooling
        # shape = [2 * B * T_s * T_w, T_c + h - 1, D_r, 1]
        r_c = tf.expand_dims(r_c, 3)
        # max-over-time-pooling, shape = [2 * B * T_s * T_w, 1, D_r, 1]
        r_c = tf.nn.max_pool(r_c,
                             ksize=[1, T_c + h - 1, 1, 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID',
                             )
        # shape = [2 + B * T_s * T_w, D_r]
        r_c = tf.squeeze(r_c)
        #  shape = [2 * B, T_s, T_w, D_r]
        r_c = tf.reshape(r_c, [2 * self.B, T_s, T_w, D_r])

        return r_c

    #############################################
    # BiLSTM layer for words-to-sentence encoding
    #############################################
    def bilstm_layer_ws(self, e_w_f, N_w):

        D_w = self.hyper_parameters['D_w']
        D_r = self.hyper_parameters['D_r']
        D_s = self.hyper_parameters['D_s']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        # reshape N_w, shape = [2 * B * T_s]
        N_w = tf.reshape(N_w, shape=[2 * self.B * T_s])
        # reshape input word embeddings, shape = [2 * B * T_s, T_w, D_w + D_r]
        e_w_f = tf.reshape(e_w_f, shape=[2 * self.B * T_s, T_w, D_w + D_r])
        # reverse input sentences
        e_w_b = tf.reverse_sequence(e_w_f, seq_lengths=N_w, seq_axis=1)

        h_0_f = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)
        h_0_b = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)
        c_0_f = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)
        c_0_b = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)

        t_0 = tf.constant(0, dtype=tf.int32)

        N_w_t = tf.tile(tf.expand_dims(N_w, axis=1), tf.constant([1, T_w], tf.int32))

        states = tf.scan(self.bilstm_cell_ws,
                         [tf.transpose(e_w_f, perm=[1, 0, 2]),
                         tf.transpose(e_w_b, perm=[1, 0, 2]),
                         tf.transpose(N_w_t, perm=[1, 0])],
                         initializer=[h_0_f, h_0_b, c_0_f, c_0_b, t_0],
                         name='lstm_ws_layer',
                         )

        h_f = states[0]
        h_b = states[1]
        h_f = tf.transpose(h_f, perm=[1, 0, 2])
        h_b = tf.transpose(h_b, perm=[1, 0, 2])
        # reverse again backward state
        h_b = tf.reverse_sequence(h_b, seq_lengths=N_w, seq_axis=1)

        # concatenate hidden states, shape=[2 * B * T_s, T_w, 2 * D_s]
        h = tf.concat([h_f, h_b], axis=2)
        # reshape input word embeddings, shape = [2 * B, T_s, T_w, 2 * D_s]
        h = tf.reshape(h, shape=[2 * self.B, T_s, T_w, 2 * D_s])

        return h

    ############################################
    # BiLSTM cell for words-to-sentence encoding
    ############################################
    def bilstm_cell_ws(self, prev, input):

        # input parameters
        h_prev_f = prev[0]
        h_prev_b = prev[1]
        c_prev_f = prev[2]
        c_prev_b = prev[3]
        t = prev[4]

        e_w_f = input[0]
        e_w_b = input[1]
        N_w = input[2]

        # compute next forward states
        h_next_f, c_next_f = self.lstm_cell(e_w_f, h_prev_f, c_prev_f,
                                            self.theta['lstm_ws_forward'],
                                            self.dropout['lstm_ws_forward'],
                                            )
        # compute next backward states
        h_next_b, c_next_b = self.lstm_cell(e_w_b, h_prev_b, c_prev_b,
                                            self.theta['lstm_ws_backward'],
                                            self.dropout['lstm_ws_backward'],
                                            )

        # t < T
        condition = tf.less(t, N_w)

        # copy-through states if t > T
        h_next_f = tf.where(condition, h_next_f, h_prev_f)
        c_next_f = tf.where(condition, c_next_f, c_prev_f)
        h_next_b = tf.where(condition, h_next_b, h_prev_b)
        c_next_b = tf.where(condition, c_next_b, c_prev_b)

        return [h_next_f, h_next_b, c_next_f, c_next_b, tf.add(t, 1)]

    #################################################
    # BiLSTM layer for sentences-to-document encoding
    #################################################
    def bilstm_layer_sd(self, e_s_f, N_s):

        D_d = self.hyper_parameters['D_d']
        T_s = self.hyper_parameters['T_s']

        # reverse input sentences
        e_s_b = tf.reverse_sequence(e_s_f, seq_lengths=N_s, seq_axis=1)

        h_0_f = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)
        h_0_b = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)
        c_0_f = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)
        c_0_b = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)

        t_0 = tf.constant(0, dtype=tf.int32)

        N_s_t = tf.tile(tf.expand_dims(N_s, axis=1), tf.constant([1, T_s], tf.int32))

        states = tf.scan(self.bilstm_cell_sd,
                         [tf.transpose(e_s_f, perm=[1, 0, 2]),
                          tf.transpose(e_s_b, perm=[1, 0, 2]),
                          tf.transpose(N_s_t, perm=[1, 0])],
                         initializer=[h_0_f, h_0_b, c_0_f, c_0_b, t_0],
                         name='lstm_sd_layer',
                         )

        h_f = states[0]
        h_b = states[1]
        h_f = tf.transpose(h_f, perm=[1, 0, 2])
        h_b = tf.transpose(h_b, perm=[1, 0, 2])
        # reverse again backward state
        h_b = tf.reverse_sequence(h_b, seq_lengths=N_s, seq_axis=1)

        # concatenate hidden states, shape=[2 * B, T_s, 2 * D_d]
        h = tf.concat([h_f, h_b], axis=2)

        return h

    ################################################
    # BiLSTM cell for sentences-to-document encoding
    ################################################
    def bilstm_cell_sd(self, prev, input):

        # input parameters
        h_prev_f = prev[0]
        h_prev_b = prev[1]
        c_prev_f = prev[2]
        c_prev_b = prev[3]
        t = prev[4]

        e_s_f = input[0]
        e_s_b = input[1]
        N_s = input[2]

        # compute next forward states
        h_next_f, c_next_f = self.lstm_cell(e_s_f, h_prev_f, c_prev_f,
                                            self.theta['lstm_sd_forward'],
                                            self.dropout['lstm_sd_forward'],
                                            )
        # compute next backward states
        h_next_b, c_next_b = self.lstm_cell(e_s_b, h_prev_b, c_prev_b,
                                            self.theta['lstm_sd_backward'],
                                            self.dropout['lstm_sd_backward'],
                                            )

        # t < T
        condition = tf.less(t, N_s)

        # copy-through states if t > T
        h_next_f = tf.where(condition, h_next_f, h_prev_f)
        c_next_f = tf.where(condition, c_next_f, c_prev_f)
        h_next_b = tf.where(condition, h_next_b, h_prev_b)
        c_next_b = tf.where(condition, c_next_b, c_prev_b)

        return [h_next_f, h_next_b, c_next_f, c_next_b, tf.add(t, 1)]

    ##################
    # single LSTM cell
    ##################
    def lstm_cell(self, e_w, h_prev, c_prev, params, dropout):

        dropout_x = tf.concat([dropout['x'],
                               dropout['x']],
                              axis=0)
        dropout_h = tf.concat([dropout['h'],
                               dropout['h']],
                              axis=0)

        W_i = params['W_i']
        U_i = params['U_i']
        b_i = params['b_i']
        W_f = params['W_f']
        U_f = params['U_f']
        b_f = params['b_f']
        W_o = params['W_o']
        U_o = params['U_o']
        b_o = params['b_o']
        W_c = params['W_c']
        U_c = params['U_c']
        b_c = params['b_c']

        is_training = self.placeholders['is_training']

        e_w = tf.cond(tf.equal(is_training, tf.constant(True)),
                      lambda: tf.multiply(dropout_x, e_w),
                      lambda: e_w,
                      )
        h_prev = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_h, h_prev),
                         lambda: h_prev,
                         )
        # forget
        i_t = tf.sigmoid(tf.matmul(e_w, W_i) + tf.matmul(h_prev, U_i) + b_i)
        # input
        f_t = tf.sigmoid(tf.matmul(e_w, W_f) + tf.matmul(h_prev, U_f) + b_f)
        # new memory
        c_tilde = tf.tanh(tf.matmul(e_w, W_c) + tf.matmul(h_prev, U_c) + b_c)
        # final memory
        c_next = tf.multiply(i_t, c_tilde) + tf.multiply(f_t, c_prev)
        # output
        o_t = tf.sigmoid(tf.matmul(e_w, W_o) + tf.matmul(h_prev, U_o) + b_o)
        # next hidden state
        h_next = tf.multiply(o_t, tf.tanh(c_next))

        return h_next, c_next

    ################################################
    # attention layer for words-to-sentence encoding
    ################################################
    def att_layer_ws(self, h_w, N_w):

        D_s = self.hyper_parameters['D_s']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        # prepare "siamese" dropout
        is_training = self.placeholders['is_training']
        dropout_Wb = tf.concat([self.dropout['att_ws']['Wb'],
                                self.dropout['att_ws']['Wb']],
                                axis=0)
        dropout_v = tf.concat([self.dropout['att_ws']['v'],
                               self.dropout['att_ws']['v']],
                               axis=0)

        # apply dropout, shape=[2 * B, T_s, T_w, 2 * D_s]
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                        lambda: tf.multiply(dropout_Wb, h_w),
                        lambda: h_w,
                        )
        # shape=[2 * B * T_s * T_w, 2 * D_s]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s * T_w, 2 * D_s])
        scores = tf.nn.tanh(tf.nn.xw_plus_b(scores,
                                           self.theta['att_ws']['W'],
                                           self.theta['att_ws']['b']))
        # shape=[2 * B * T_s, T_w, D_s]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s, T_w, D_s])
        # apply dropout
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                        lambda: tf.multiply(dropout_v, scores),
                        lambda: scores,
                        )
        # shape=[2 * B * T_s * T_w, 2 * D_s]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s * T_w, D_s])

        # shape=[2 * B * T_s * T_w, 1]
        scores = tf.matmul(scores, self.theta['att_ws']['v'])
        # shape=[2 * B, T_s, T_w]
        scores = tf.reshape(scores, shape=[2 * self.B, T_s, T_w])

        # binary mask, shape = [2 * B, T_s, T_w]
        mask = tf.sequence_mask(N_w, maxlen=T_w, dtype=tf.float32)
        mask = (1.0 - mask) * -5000.0

        # shape = [2 * B, T_s, T_w]
        scores = scores + mask
        scores = tf.nn.softmax(scores, axis=2)

        # expand to shape=[2 * B, T_s, T_w, 1]
        alpha = tf.expand_dims(scores, axis=3)
        # fill up to shape=[2 * B, T_s, T_w, D_s]
        alpha = tf.tile(alpha, tf.stack([1, 1, 1, 2 * D_s]))
        # combine to get sentence representations, shape=[2 * B, T_s, 2 * D_s]
        e_s = tf.reduce_sum(tf.multiply(alpha, h_w), axis=2, keepdims=False)

        e_s = tf.squeeze(e_s)

        return e_s

    ####################################################
    # attention layer for sentences-to-docuemnt encoding
    ####################################################
    def att_layer_sd(self, h_s, N_s):

        D_d = self.hyper_parameters['D_d']
        T_s = self.hyper_parameters['T_s']

        # prepare "siamese" dropout
        is_training = self.placeholders['is_training']
        dropout_Wb = tf.concat([self.dropout['att_sd']['Wb'],
                                self.dropout['att_sd']['Wb']],
                               axis=0)
        dropout_v = tf.concat([self.dropout['att_sd']['v'],
                               self.dropout['att_sd']['v']],
                              axis=0)

        # apply dropout, shape=[2 * B, T_s, 2 * D_d]
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_Wb, h_s),
                         lambda: h_s,
                         )
        # shape=[2 * B * T_s, 2 * D_d]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s, 2 * D_d])
        scores = tf.nn.tanh(tf.nn.xw_plus_b(scores,
                                            self.theta['att_sd']['W'],
                                            self.theta['att_sd']['b']))
        # shape=[2 * B, T_s, 2 * D_d]
        scores = tf.reshape(scores, shape=[2 * self.B, T_s, D_d])

        # apply dropout
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_v, scores),
                         lambda: scores,
                         )
        # shape=[2 * B * T_s, 2 * D_d]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s, D_d])

        # shape=[2 * B * T_s, 1]
        scores = tf.matmul(scores, self.theta['att_sd']['v'])
        # shape=[2 * B, T_s]
        scores = tf.reshape(scores, shape=[2 * self.B, T_s])

        # binary mask, shape = [2 * B, T_s]
        mask = tf.sequence_mask(N_s, maxlen=T_s, dtype=tf.float32)
        mask = (1.0 - mask) * -5000.0

        # shape = [2 * B, T_s]
        scores = scores + mask
        scores = tf.nn.softmax(scores, axis=1)

        # expand to shape=[2 * B, T_s, 1]
        alpha = tf.expand_dims(scores, axis=2)
        # fill up to shape=[2 * B, T_s, 2 * D_d]
        alpha = tf.tile(alpha, tf.stack([1, 1, 2 * D_d]))
        # combine to get doc representations, shape=[2 * B, 2 * D_d]
        e_d = tf.reduce_sum(tf.multiply(alpha, h_s), axis=1, keepdims=False)

        return e_d

    #########
    # dropout
    #########
    def make_dropout_mask(self, shape, keep_prob):
        keep_prob = tf.convert_to_tensor(keep_prob, dtype=tf.float32)
        random_tensor = keep_prob + tf.random_uniform(shape, dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        dropout_mask = tf.divide(binary_tensor, keep_prob)
        return dropout_mask

    def initialize_dropout(self):

        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']
        D_w = self.hyper_parameters['D_w']
        D_s = self.hyper_parameters['D_s']
        D_d = self.hyper_parameters['D_d']
        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        dropout = {}

        with tf.variable_scope('dropout_cnn'):
            dropout['cnn'] = self.make_dropout_mask(shape=[self.B * T_s * T_w, T_c, D_c],
                                                    keep_prob=self.hyper_parameters['keep_prob_cnn'],
                                                    )
        with tf.variable_scope('dropout_lstm_ws_forward'):
            dropout['lstm_ws_forward'] = {}
            dropout['lstm_ws_forward']['x'] = self.make_dropout_mask(shape=[self.B * T_s, D_w + D_r],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
            dropout['lstm_ws_forward']['h'] = self.make_dropout_mask(shape=[self.B * T_s, D_s],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
        with tf.variable_scope('dropout_lstm_ws_backward'):
            dropout['lstm_ws_backward'] = {}
            dropout['lstm_ws_backward']['x'] = self.make_dropout_mask(shape=[self.B * T_s, D_w + D_r],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
            dropout['lstm_ws_backward']['h'] = self.make_dropout_mask(shape=[self.B * T_s, D_s],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
        with tf.variable_scope('dropout_lstm_sd_forward'):
            dropout['lstm_sd_forward'] = {}
            dropout['lstm_sd_forward']['x'] = self.make_dropout_mask(shape=[self.B, 2 * D_s],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
            dropout['lstm_sd_forward']['h'] = self.make_dropout_mask(shape=[self.B, D_d],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
        with tf.variable_scope('dropout_lstm_sd_backward'):
            dropout['lstm_sd_backward'] = {}
            dropout['lstm_sd_backward']['x'] = self.make_dropout_mask(shape=[self.B, 2 * D_s],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
            dropout['lstm_sd_backward']['h'] = self.make_dropout_mask(shape=[self.B, D_d],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
        with tf.variable_scope('dropout_att_ws'):
            dropout['att_ws'] = {}
            dropout['att_ws']['Wb'] = self.make_dropout_mask(shape=[self.B, T_s, 1, 2 * D_s],
                                                             keep_prob=self.hyper_parameters['keep_prob_att'],
                                                             )
            dropout['att_ws']['v'] = self.make_dropout_mask(shape=[self.B * T_s, 1, D_s],
                                                            keep_prob=self.hyper_parameters['keep_prob_att'],
                                                            )
        with tf.variable_scope('dropout_att_sd'):
            dropout['att_sd'] = {}
            dropout['att_sd']['Wb'] = self.make_dropout_mask(shape=[self.B, 1, 2 * D_d],
                                                             keep_prob=self.hyper_parameters['keep_prob_att'],
                                                             )
            dropout['att_sd']['v'] = self.make_dropout_mask(shape=[self.B, 1, D_d],
                                                            keep_prob=self.hyper_parameters['keep_prob_att'],
                                                            )
        with tf.variable_scope('dropout_metric'):
            dropout['metric'] = {}
            dropout['metric']['x'] = self.make_dropout_mask(shape=[self.B, 2 * D_d],
                                                            keep_prob=self.hyper_parameters['keep_prob_metric'],
                                                            )
        return dropout

    def initialize_parameters(self):

        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']
        h = self.hyper_parameters['w']
        D_w = self.hyper_parameters['D_w']
        D_s = self.hyper_parameters['D_s']
        D_d = self.hyper_parameters['D_d']
        D_mlp = self.hyper_parameters['D_mlp']

        theta = {}

        with tf.variable_scope('theta_cnn'):
            theta['cnn'] = self.initialize_cnn(D_c, D_r, h)

        with tf.variable_scope('theta_lstm_ws_forward'):
            theta['lstm_ws_forward'] = self.initialize_lstm(D_w + D_r, D_s)
        with tf.variable_scope('theta_lstm_ws_backward'):
            theta['lstm_ws_backward'] = self.initialize_lstm(D_w + D_r, D_s)

        with tf.variable_scope('theta_lstm_sd_forward'):
            theta['lstm_sd_forward'] = self.initialize_lstm(2 * D_s, D_d)
        with tf.variable_scope('theta_lstm_sd_backward'):
            theta['lstm_sd_backward'] = self.initialize_lstm(2 * D_s, D_d)

        with tf.variable_scope('theta_att_ws'):
            theta['att_ws'] = self.initialize_att(2 * D_s, D_s)
        with tf.variable_scope('theta_att_sd'):
            theta['att_sd'] = self.initialize_att(2 * D_d, D_d)

        with tf.variable_scope('theta_metric'):
            theta['metric'] = self.initialize_mlp(2 * D_d, D_mlp)

        return theta

    def initialize_mlp(self, D_in, D_out):
        r = 0.4
        theta = {'W': tf.get_variable('W',
                                      shape=[D_in, D_out],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'b': tf.get_variable('b',
                                      shape=[D_out],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 }
        return theta

    def initialize_cnn(self, D_in, D_out, h):
        r = 0.1
        theta = {'W': tf.get_variable(name='W',
                                      shape=[h, D_in, D_out],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'b': tf.get_variable(name='b',
                                      shape=[D_out],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 }

        return theta

    def initialize_att(self, D_in, D_out):
        r = 0.03
        theta = {'W': tf.get_variable('W_a',
                                      shape=[D_in, D_out],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'v': tf.get_variable('v_a', shape=[D_out, 1],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'b': tf.get_variable('b_a', shape=[D_out],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 }

        return theta

    def initialize_lstm(self, D_in, D_out):
        r = 0.05
        theta = {'W_i': tf.get_variable('W_i', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_i': tf.get_variable('U_i', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_i': tf.get_variable('b_i', shape=[1, D_out],
                                        initializer=tf.constant_initializer(0.0),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'W_f': tf.get_variable('W_f', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_f': tf.get_variable('U_f', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_f': tf.get_variable('b_f', shape=[1, D_out],
                                        initializer=tf.constant_initializer(2.5),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'W_c': tf.get_variable('W_c', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_c': tf.get_variable('U_c', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_c': tf.get_variable('b_c', shape=[1, D_out],
                                        initializer=tf.constant_initializer(0.0),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'W_o': tf.get_variable('W_o', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_o': tf.get_variable('U_o', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_o': tf.get_variable('b_o', shape=[1, D_out],
                                        initializer=tf.constant_initializer(1.0),
                                        trainable=True,
                                        dtype=tf.float32,
                                        )
                 }
        return theta

    #############################
    # update weights of the model
    #############################
    def update_model(self, x_w_L, x_w_R, x_c_L, x_c_R, labels, N_w_L, N_w_R, N_s_L, N_s_R, lr):

        feed_dict = {self.placeholders['x_w_L']: x_w_L,
                     self.placeholders['x_w_R']: x_w_R,
                     self.placeholders['x_c_L']: x_c_L,
                     self.placeholders['x_c_R']: x_c_R,
                     self.placeholders['labels']: labels,
                     self.placeholders['N_w_L']: N_w_L,
                     self.placeholders['N_w_R']: N_w_R,
                     self.placeholders['N_s_L']: N_s_L,
                     self.placeholders['N_s_R']: N_s_R,
                     self.placeholders['is_training']: True,
                     self.placeholders['lr']: lr,
                     }
        _, step, loss, pred = self.sess.run([self.optimizer, self.step, self.loss, self.pred], feed_dict=feed_dict)

        # execute label computation function
        labels_hat = self.compute_labels(labels, pred)
        # compute values for accuracy, F1-score and c@1
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN(labels, labels_hat)

        return step, loss, TP, FP, TN, FN

    ################
    # evaluate model
    ################
    @staticmethod
    def grid_search(pred, labels):
        L = list(np.linspace(0, 1, 100))
        scores = []
        scores_o = []
        pred.reshape(-1)
        labels.reshape(-1)
        for l in L:
            l1 = 0.5 - l
            l2 = 0.5 + l
            m1 = pred > l1
            m2 = pred < l2
            pred_l = pred.copy()
            pred_l[m1 * m2] = 0.5
            s = evaluate_all(pred_y=pred_l, true_y=labels)
            scores.append(s)
            scores_o.append(s['overall'])
        j = int(np.argmax(scores_o))
        return scores[j], L[j]

    def evaluate_model(self, docs_L, docs_R, labels, batch_size):

        num_batches = ceil(len(labels) / batch_size)

        TP, FP, TN, FN = 0, 0, 0, 0
        pred = []

        for i in range(num_batches):

            # get next batch
            docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                           (i + 1) * batch_size,
                                                           docs_L,
                                                           docs_R,
                                                           labels,
                                                           )
            B = len(labels_i)

            if B > 0:
                # word/character embeddings
                x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                # accuracy for training set
                curr_TP, curr_FP, curr_TN, curr_FN, curr_pred \
                    = self.compute_eval_measures(x_w_L=x_w_L,
                                                 x_w_R=x_w_R,
                                                 x_c_L=x_c_L,
                                                 x_c_R=x_c_R,
                                                 labels=np.array(labels_i).reshape((B, 1)),
                                                 N_w_L=N_w_L,
                                                 N_w_R=N_w_R,
                                                 N_s_L=N_s_L,
                                                 N_s_R=N_s_R,
                                                 )
                TP += curr_TP
                FP += curr_FP
                TN += curr_TN
                FN += curr_FN
                pred.extend(curr_pred)

        acc = self.compute_accuracy(TP, FP, TN, FN)
        scores, th = self.grid_search(np.array(pred), np.array(labels))


        return acc, scores, th

    ###############################################
    # evaluate model 'AdHominem' for a single batch
    ###############################################
    def compute_eval_measures(self, x_w_L, x_w_R, x_c_L, x_c_R, labels, N_w_L, N_w_R, N_s_L, N_s_R):

        # compute distances
        pred = self.sess.run(self.pred,
                             feed_dict={self.placeholders['x_w_L']: x_w_L,
                                        self.placeholders['x_w_R']: x_w_R,
                                        self.placeholders['x_c_L']: x_c_L,
                                        self.placeholders['x_c_R']: x_c_R,
                                        self.placeholders['labels']: labels,
                                        self.placeholders['N_w_L']: N_w_L,
                                        self.placeholders['N_w_R']: N_w_R,
                                        self.placeholders['N_s_L']: N_s_L,
                                        self.placeholders['N_s_R']: N_s_R,
                                        self.placeholders['is_training']: False,
                                        })

        # execute label computation function
        labels_hat_kernel = self.compute_labels(labels, pred)
        # compute values for accuracy, F1-score and c@1
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN(labels, labels_hat_kernel)

        return TP, FP, TN, FN, pred

    ##########################
    # calculate TP, FP, TN, FN
    ##########################
    @staticmethod
    def compute_TP_FP_TN_FN(labels, labels_hat):

        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(labels_hat)):
            if labels[i] == 1 and labels_hat[i] == 1:
                TP += 1
            if labels[i] == 0 and labels_hat[i] == 1:
                FP += 1
            if labels[i] == 0 and labels_hat[i] == 0:
                TN += 1
            if labels[i] == 1 and labels_hat[i] == 0:
                FN += 1

        return TP, FP, TN, FN

    ##################
    # compute accuracy
    ##################
    @staticmethod
    def compute_accuracy(TP, FP, TN, FN):

        acc = (TP + TN) / (TP + FP + TN + FN)

        return acc

    ################
    # get next batch
    ################
    @staticmethod
    def next_batch(t_s, t_e, docs_L, docs_R, labels):

        docs_L = docs_L[t_s:t_e]
        docs_R = docs_R[t_s:t_e]
        labels = labels[t_s:t_e]

        return docs_L, docs_R, labels

    ################################################
    # transform document to a tensor with embeddings
    ################################################
    def doc2mat(self, docs):
        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        V_c = self.hyper_parameters['V_c']
        V_w = self.hyper_parameters['V_w']

        # batch size
        B = len(docs)
        N_w = np.zeros((B, T_s), dtype=np.int32)
        N_s = np.zeros((B,), dtype=np.int32)

        # word-based tensor, shape = [B, T_s, T_w]
        x_w = np.zeros((B, T_s, T_w), dtype=np.int32)
        # character-based tensor
        x_c = np.zeros((B, T_s, T_w, T_c), dtype=np.int32)

        # current document
        for i, doc in enumerate(docs):
            N_s[i] = len(doc[:T_s])
            # current sentence
            for j, sentence in enumerate(doc[:T_s]):
                tokens = sentence.split()
                N_w[i, j] = len(tokens)
                # current tokens
                for k, token in enumerate(tokens):
                    if token in V_w:
                        x_w[i, j, k] = V_w[token]
                    else:
                        x_w[i, j, k] = V_w['<UNK>']
                    # current character
                    for l, chr in enumerate(token[:T_c]):
                        if chr in V_c:
                            x_c[i, j, k, l] = V_c[chr]
                        else:
                            x_c[i, j, k, l] = V_c['<UNK>']
        return x_w, N_w, N_s, x_c

    #################
    # train AdHominem
    #################
    def train_model(self, train_set, test_set, file_results):

        # total number of epochs
        epochs = self.hyper_parameters['epochs']

        # number of batches for dev/test set
        batch_size_tr = self.hyper_parameters['batch_size_tr']
        batch_size_te = self.hyper_parameters['batch_size_te']

        # extract train, test sets
        docs_L_tr, docs_R_tr, labels_tr = train_set
        docs_L_te, docs_R_te, labels_te = test_set

        # define learning rate weighting for adversarial domain adaption
        p = np.array(range(epochs)) / epochs
        lr = [min(self.hyper_parameters['initial_learning_rate'] / ((1. + 5.0 * i) ** 0.4), 0.0015) for i in p]

        # number of batches for training
        num_batches_tr = ceil(len(labels_tr) / batch_size_tr)

        ################
        # start training
        ################
        for epoch in range(epochs):

            # shuffle
            docs_L_tr, docs_R_tr, labels_tr = shuffle(docs_L_tr, docs_R_tr, labels_tr)

            # average loss and accuracy
            loss = []
            TP, FP, TN, FN = 0, 0, 0, 0

            # loop over all batches
            for i in range(num_batches_tr):

                # get next batch
                docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size_tr,
                                                               (i + 1) * batch_size_tr,
                                                               docs_L_tr,
                                                               docs_R_tr,
                                                               labels_tr,
                                                               )

                # current batch size
                B = len(labels_i)

                if B > 0:

                    # word / character embeddings
                    x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                    x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                    # update model parameters
                    step, curr_loss, curr_TP, curr_FP, curr_TN, curr_FN, \
                        = self.update_model(x_w_L=x_w_L,
                                            x_w_R=x_w_R,
                                            x_c_L=x_c_L,
                                            x_c_R=x_c_R,
                                            labels=np.array(labels_i).reshape((B, 1)),
                                            N_w_L=N_w_L,
                                            N_w_R=N_w_R,
                                            N_s_L=N_s_L,
                                            N_s_R=N_s_R,
                                            lr=lr[epoch],
                                            )
                    loss.append(curr_loss)
                    TP += curr_TP
                    FP += curr_FP
                    TN += curr_TN
                    FN += curr_FN
                    curr_acc = self.compute_accuracy(curr_TP, curr_FP, curr_TN, curr_FN)

                    # print current results
                    s = "epoch:" + str(epoch) \
                        + ", batch: " + str(round(100 * (i + 1) / num_batches_tr, 2)) \
                        + ", loss: " + str(np.mean(loss)) \
                        + ", acc: " + str(round(100 * (TP + TN) / (TP + FP + TN + FN), 2)) \
                        + ", curr Loss: " + str(round(curr_loss, 2)) \
                        + ", curr Acc: " + str(round(100 * curr_acc, 2)) \
                        + ", lr: " + str(round(float(lr[epoch]), 6))
                    print(s)

            # compute accuracy on train set
            acc_tr = self.compute_accuracy(TP, FP, TN, FN)
            # compute accuracy on test set (including PAN 2020 metrics and grid search for threshold)
            acc_te, scores, th = self.evaluate_model(docs_L_te, docs_R_te, labels_te, batch_size_te)

            # update "results.txt"-file
            s = 'epoch: ' + str(epoch) \
                + ', loss: ' + str(round(float(np.mean(loss)), 4)) \
                + ', acc (tr): ' + str(round(100 * acc_tr, 4)) \
                + ', acc (te): ' + str(round(100 * acc_te, 4)) \
                + ', th : ' + str(th)  \
                + ', ' + str(scores)
            open(file_results, 'a').write(s + '\n')

