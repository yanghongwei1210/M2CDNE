import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from scipy.sparse import lil_matrix
import scipy.io
import scipy.io as sio

def get_prediction(source,target):
    #Load target data
    A_t, X_t, Y_t = utils.load_network('./input/'+str(target)+'.mat')
    # Compute PPMI
    Kstep = 3
    A_k_t = utils.AggTranProbMat(A_t, Kstep)
    PPMI_t = utils.ComputePPMI(A_k_t)
    n_PPMI_t = utils.MyScaleSimMat(PPMI_t)  # row normalized PPMI
    X_n_t = np.matmul(n_PPMI_t, lil_matrix.toarray(X_t))  # neibors' attribute matrix

    whole_xt_stt = utils.csr_2_sparse_tensor_tuple(X_t)
    whole_xt_stt_nei = utils.csr_2_sparse_tensor_tuple(X_n_t)

    model_path = './model/' + str(source) + '_' + str(target)
    meta_path = './model/' + str(source) + '_' + str(target) + '/model.ckpt-10.meta'
    filename = 'Label_Pred/' + str(source) + '_' + str(target) + '_predlabel.mat'

    with tf.Session() as sess:
        # restore the model
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        Ada_lambda = graph.get_tensor_by_name('Ada_lambda:0')
        dropout = graph.get_tensor_by_name('dropout:0')
        X_shape = graph.get_tensor_by_name('X/shape:0')
        X_values = graph.get_tensor_by_name('X/values:0')
        X_indices = graph.get_tensor_by_name('X/indices:0')

        X_nei_shape = graph.get_tensor_by_name('X_nei/shape:0')
        X_nei_values = graph.get_tensor_by_name('X_nei/values:0')
        X_nei_indices = graph.get_tensor_by_name('X_nei/indices:0')
        op_to_pred = graph.get_tensor_by_name('Node_Classifier/clf_pred_prob:0')
        pred_prob_xt = sess.run(op_to_pred,feed_dict={X_indices: whole_xt_stt[0], X_values: whole_xt_stt[1],
                                              X_shape: whole_xt_stt[2],
                                              X_nei_indices: whole_xt_stt_nei[0],
                                              X_nei_values: whole_xt_stt_nei[1], X_nei_shape: whole_xt_stt_nei[2],
                                              Ada_lambda: 1.0,
                                              dropout: 0.})

        scipy.io.savemat(filename, {'group_T': pred_prob_xt})
    print('Get classifier predicted labels！')
    return filename



target='MIR_pT2I'
dataset=['CLEF_pT2I','PASCAL_pT2I']
#'CLEF_pT2I','MIR_pT2I','PASCAL_pT2I','NUS_pT2I'
source_dataset = [d for d in dataset if d != target]
for source in source_dataset:
    get_prediction(source, target)

