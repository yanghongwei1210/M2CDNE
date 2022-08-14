import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from scipy.sparse import vstack
import scipy.io
from scipy.sparse import lil_matrix
from M2CDNE_model import M2CDNE
import matplotlib.pyplot as plt
import globalvar as gl


global D_M, D_C, MU
gl._init()
gl.set_value('D_M', 0)
gl.set_value('D_C', 0)
gl.set_value('MU', 0)

# Train_and_evaluate
def train_and_evaluate(input_data, config, source, target, random_state=0):

    ###get input data
    PPMI_s = input_data['PPMI_S']
    PPMI_t = input_data['PPMI_T']
    X_s = input_data['attrb_S']
    X_t = input_data['attrb_T']
    X_n_s = input_data['attrb_nei_S']
    X_n_t = input_data['attrb_nei_T']
    Y_s = input_data['label_S']
    Y_t = input_data['label_T']
    Y_t_o = np.zeros(np.shape(Y_t))   #observable label matrix of target network, all zeros

    ## Combine the node attributes and the neighbor attributes of the node to get the complete data of the node
    X_s_new = lil_matrix(np.concatenate((lil_matrix.toarray(X_s), X_n_s), axis=1))
    X_t_new = lil_matrix(np.concatenate((lil_matrix.toarray(X_t), X_n_t), axis=1))


    ## Num
    n_input = X_s.shape[1]
    num_class = Y_s.shape[1]
    num_nodes_S = X_s.shape[0]
    num_nodes_T = X_t.shape[0]

    ## model config
    clf_type = config['clf_type']
    dropout = config['dropout']
    num_epoch = config['num_epoch']
    batch_size = config['batch_size']
    n_hidden = config['n_hidden']
    n_emb = config['n_emb']
    l2_w = config['l2_w']
    net_pro_w = config['net_pro_w']
    emb_filename = config['emb_filename']
    label_filename = config['label_filename']
    domain_filename = config['domain_filename']
    model_filename = config['model_filename']
    lr_ini = config['lr_ini']


    whole_xs_xt_stt = utils.csr_2_sparse_tensor_tuple(vstack([X_s, X_t]))
    whole_xs_xt_stt_nei = utils.csr_2_sparse_tensor_tuple(vstack([X_n_s, X_n_t]))

    # Set random seed
    tf.set_random_seed(random_state)
    np.random.seed(random_state)
    model = M2CDNE(n_input, n_hidden, n_emb, num_class, clf_type, l2_w, net_pro_w, batch_size)
    merged_summary_op = tf.summary.merge_all()

    # saver:save model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Random initialize
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('graphs', sess.graph)
        gl.set_value('D_M', 0)
        gl.set_value('D_C', 0)
        gl.set_value('MU', 0)


        for cEpoch in range(num_epoch):
            S_batches = utils.batch_generator([X_s_new, Y_s], int(batch_size / 2),shuffle=True)  # [X_s_new, Y_s]：X_s|X_n_s,Y_s
            T_batches = utils.batch_generator([X_t_new, Y_t_o], int(batch_size / 2),shuffle=True)  # [X_t_new, Y_t_o]：X_t|X_n_t,Y_t_o

            num_batch = round(max(num_nodes_S / (batch_size / 2), num_nodes_T / (batch_size / 2)))

            # Adaptation param and learning rate schedule as described in the DANN paper
            p = float(cEpoch) / (num_epoch)
            lr = lr_ini / (1. + 10 * p) ** 0.75
            grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  #gradually change from 0 to 1

            D_M = gl.get_value('D_M')
            D_C = gl.get_value('D_C')
            MU = gl.get_value('MU')

            if D_M == 0 and D_C == 0 and MU == 0:
                MU = 0.5
            else:
                D_M = D_M / 5
                D_C = D_C / 5
                MU = 1 - D_M / (D_M + D_C)
                print(MU)
            D_M=0
            D_C=0
            gl.set_value('D_M', D_M)
            gl.set_value('D_C', D_C)
            gl.set_value('MU', MU)

            ##in each epoch, train all the mini batches
            for cBatch in range(num_batch):
                ### each batch, half nodes from source network, and half nodes from target network
                xs_ys_batch, shuffle_index_s = next(S_batches)
                xs_batch = xs_ys_batch[0]
                ys_batch = xs_ys_batch[1]

                xt_yt_batch, shuffle_index_t = next(T_batches)
                xt_batch = xt_yt_batch[0]
                yt_batch = xt_yt_batch[1]

                x_batch = vstack([xs_batch, xt_batch])
                batch_csr = x_batch.tocsr()
                xb = utils.csr_2_sparse_tensor_tuple(batch_csr[:, 0:n_input])
                xb_nei = utils.csr_2_sparse_tensor_tuple(batch_csr[:,-n_input:])
                yb = np.vstack([ys_batch, yt_batch])


                mask_L = np.array(np.sum(yb, axis=1) > 0,dtype=np.float)  # 1 if the node is with observed label, 0 if the node is without label
                domain_label = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]), np.tile([0., 1.], [batch_size // 2,1])])  # [1,0] for source, [0,1] for target

                ##topological proximity matrix between nodes in each mini-batch
                a_s, a_t = utils.batchPPMI(batch_size, shuffle_index_s, shuffle_index_t, PPMI_s, PPMI_t)
                _, tloss, summary_str,aa,bb = sess.run([model.train_op, model.total_loss, merged_summary_op,model.D_C,model.D_M],
                                                 feed_dict={model.X: xb, model.X_nei: xb_nei, model.y_true: yb,
                                                            model.d_label: domain_label, model.A_s: a_s, model.A_t: a_t,
                                                            model.mask: mask_L, model.learning_rate: lr,
                                                            model.Ada_lambda: grl_lambda, model.dropout: dropout})
                summary_writer.add_summary(summary_str, cEpoch * num_batch + cBatch)
                gl.set_value('D_M', aa)
                gl.set_value('D_C', bb)
            #Compute evaluation on test data by the end of each epoch
            pred_prob_xs_xt = sess.run(model.pred_prob,
                                       feed_dict={model.X: whole_xs_xt_stt, model.X_nei: whole_xs_xt_stt_nei,
                                                  model.Ada_lambda: 1.0,
                                                  model.dropout: 0.})
            pred_prob_xs = pred_prob_xs_xt[0:num_nodes_S, :]
            pred_prob_xt = pred_prob_xs_xt[-num_nodes_T:, :]


            print('epoch: ', cEpoch + 1,'  learning-rate:',lr)
            F1_s = utils.f1_scores(pred_prob_xs, Y_s)
            print('Source %s  micro-F1: %f, macro-F1: %f' % (source, F1_s[0], F1_s[1]))
            F1_t = utils.f1_scores(pred_prob_xt, Y_t)
            print('Target %s  testing micro-F1: %f, macro-F1: %f' % (target, F1_t[0], F1_t[1]))


            acc_s = utils.accuracy(pred_prob_xs, Y_s)
            acc_t = utils.accuracy(pred_prob_xt, Y_t)
            print('Target %s  testing accuracy: %f' % (target, acc_t))


        # save embedding features、predicted labels、predicted damain labels
        emb, label, domain = sess.run([model.emb, model.pred_prob, model.d_softmax],
                                      feed_dict={model.X: whole_xs_xt_stt, model.X_nei: whole_xs_xt_stt_nei,
                                                 model.Ada_lambda: 1.0, model.dropout: 0.})
        hs = emb[0:num_nodes_S, :]
        ht = emb[-num_nodes_T:, :]
        scipy.io.savemat(emb_filename + '_emb.mat', {'rep_S': hs, 'rep_T': ht})

        pred_prob_s = label[0:num_nodes_S, :]
        pred_prob_t = label[-num_nodes_T:, :]
        scipy.io.savemat(label_filename + '_label.mat', {'group_S': pred_prob_s, 'group_T': pred_prob_t})

        ds = domain[0:num_nodes_S, :]
        dt = domain[-num_nodes_T:, :]
        scipy.io.savemat(domain_filename + '_d.mat', {'d_rep_S': ds, 'd_rep_T': dt})

        save_path = saver.save(sess, model_filename, global_step=cEpoch + 1)
        print("Model saved in file: %s" % save_path)
        print('learning rate： ', lr)

