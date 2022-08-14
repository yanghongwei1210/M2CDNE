import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from flip_gradient import flip_gradient
import globalvar as gl

class M2CDNE(object):
    def __init__(self, n_input, n_hidden, n_emb, num_class, clf_type, l2_w, net_pro_w, batch_size):

        self.X = tf.sparse_placeholder(dtype=tf.float32,name='X')  #each node's own attributes
        self.X_nei = tf.sparse_placeholder(dtype=tf.float32,name='X_nei')   #each node's weighted neighbors' attributes
        self.y_true = tf.placeholder(dtype=tf.float32,name='y_true')
        self.d_label = tf.placeholder(dtype=tf.float32,name='d_label')  #domain label, source network [1 0] or target network [0 1]
        self.Ada_lambda = tf.placeholder(dtype=tf.float32,name='Ada_lambda')  # grl_lambda Gradient reversal scaler
        self.dropout = tf.placeholder(tf.float32,name='dropout') # dropout
        self.A_s = tf.sparse_placeholder(dtype=tf.float32,name='A_s')  # network proximity matrix of source network
        self.A_t = tf.sparse_placeholder(dtype=tf.float32,name='A_t')  # network proximity matrix of target network
        self.mask = tf.placeholder(dtype=tf.float32)  # check a node is with observable label (1) or not (0)
        self.learning_rate = tf.placeholder(dtype=tf.float32,name='learning_rate')


        with tf.name_scope('Network_Embedding'):
            ##feature exactor 1
            h1_self = utils.fc_layer(self.X, n_input, n_hidden[0], layer_name='hidden1_self', input_type='sparse',drop=self.dropout)
            h2_self = utils.fc_layer(h1_self, n_hidden[0], n_hidden[1], layer_name='hidden2_self')

            ##feature exactor 2
            h1_nei = utils.fc_layer(self.X_nei, n_input, n_hidden[0], layer_name='hidden1_nei', input_type='sparse',drop=self.dropout)
            h2_nei = utils.fc_layer(h1_nei, n_hidden[0], n_hidden[1], layer_name='hidden2_nei')

            ##concatenation layer, final embedding vector representation
            self.emb = utils.fc_layer(tf.concat([h2_self, h2_nei], 1), n_hidden[-1] * 2, n_emb, layer_name='concat')
            _emb=tf.slice(self.emb,[0,0],[-1,-1],name='_emb')

            ##pairwise constraint
            emb_s = tf.slice(self.emb, [0, 0], [int(batch_size / 2), -1],name='emb_s')
            emb_t = tf.slice(self.emb, [int(batch_size / 2), 0], [int(batch_size / 2), -1],name='emb_t')

            # L2 distance between source nodes
            r_s = tf.reduce_sum(emb_s * emb_s, 1)
            r_s = tf.reshape(r_s, [-1, 1])
            Dis_s = r_s - 2 * tf.matmul(emb_s, tf.transpose(emb_s)) + tf.transpose(r_s)  #||ei-ej||**2 (Vs)
            net_pro_loss_s = tf.reduce_mean(tf.sparse.reduce_sum(self.A_s.__mul__(Dis_s), axis=1))

            # L2 distance between target nodes
            r_t = tf.reduce_sum(emb_t * emb_t, 1)
            r_t = tf.reshape(r_t, [-1, 1])
            Dis_t = r_t - 2 * tf.matmul(emb_t, tf.transpose(emb_t)) + tf.transpose(r_t) #||ei-ej||**2 (Vt)
            net_pro_loss_t = tf.reduce_mean(tf.sparse.reduce_sum(self.A_t.__mul__(Dis_t), axis=1))

            self.net_pro_loss = net_pro_w * (net_pro_loss_s + net_pro_loss_t)


        ## node classification
        with tf.name_scope('Node_Classifier'):
            W_clf = tf.Variable(tf.truncated_normal([n_emb, num_class], stddev=1. / tf.sqrt(n_emb / 2.)),name='clf_weight')
            b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
            pred_logit = tf.matmul(self.emb, W_clf) + b_clf

            ## multi-class, softmax output
            if clf_type == 'multi-class':
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_logit, labels=self.y_true)
                loss = loss * self.mask
                self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
                self.pred_prob = tf.nn.softmax(pred_logit,name='clf_pred_prob')

            ## multi-label, sigmod output
            elif clf_type == 'multi-label':
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logit, labels=self.y_true)
                loss = loss * self.mask[:, None]  # count loss only based on labeled nodes, each column mutiply by mask
                self.clf_loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)
                self.pred_prob = tf.sigmoid(pred_logit,name='clf_pred_prob')

        ##Global Domain_Discriminator
        with tf.name_scope('Domain_Discriminator'):
            h_grl = flip_gradient(self.emb, self.Ada_lambda)
            ##MLP for domain classification
            h_dann_1 = utils.fc_layer(h_grl, n_emb, 128, layer_name='dann_fc_1')
            h_dann_2 = utils.fc_layer(h_dann_1, 128, 128, layer_name='dann_fc_2')
            W_domain = tf.Variable(tf.truncated_normal([128, 2], stddev=1. / tf.sqrt(128 / 2.)), name='dann_weight')
            b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
            d_logit = tf.matmul(h_dann_2, W_domain) + b_domain
            self.d_softmax = tf.nn.softmax(d_logit,name='d_softmax')
            self.domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logit, labels=self.d_label))

        ##Local_Discriminator
        with tf.name_scope('Local_Discriminator'):
            self.dci={}
            self.local_loss=0
            tmpd_c=0

            for i in range(num_class):
                #h_grl = flip_gradient(self.emb, self.Ada_lambda)
                p_source=self.pred_prob[:,i]
                ps=tf.reshape(p_source,(batch_size,1))
                fs=ps * h_grl[i]

                h_daan_1 = utils.fc_layer(fs, n_emb, 128, layer_name='daan_fc_1')  #n_emb
                h_daan_2 = utils.fc_layer(h_daan_1, 128, 128, layer_name='daan_fc_2')
                W_domain1 = tf.Variable(tf.truncated_normal([128, 2], stddev=1. / tf.sqrt(128 / 2.)), name='daan_weight')
                b_domain1 = tf.Variable(tf.constant(0.1, shape=[2]), name='daan_bias')
                d_logit1 = tf.matmul(h_daan_2, W_domain1) + b_domain1
                # local_out.append(d_logit1)

                self.d_softmax1 = tf.nn.softmax(d_logit1, name='d_softmax1')
                local_loss_i = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logit1, labels=self.d_label))
                self.local_loss = self.local_loss + local_loss_i
                tmpd_c=tmpd_c+2*(1-2*(local_loss_i))

            tmpd_c=tmpd_c/num_class


        D_M = gl.get_value('D_M')
        D_C = gl.get_value('D_C')
        MU = gl.get_value('MU')

        D_C = D_C + tmpd_c
        D_M = D_M+2*(1-2*self.domain_loss)
        self.D_C=D_C
        self.D_M=D_M


        all_variables = tf.trainable_variables()
        self.l2_loss = l2_w * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])

        self.join_loss=(1 - MU) * self.domain_loss + MU * self.local_loss

        a1=self.net_pro_loss + self.clf_loss

        a2=self.join_loss + self.l2_loss

        self.total_loss=tf.add(a1,a2,name='total_loss')
        tf.summary.scalar('cross-entropy', self.total_loss)
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.total_loss,name='train_op')













