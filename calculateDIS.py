import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from scipy.sparse import vstack
import scipy.io as sio
from scipy.sparse import lil_matrix
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA



def calculateEMDDIS(source,target):
    print("start calculate EMD：")
    Kstep = 3

    path1=r'./Train_result/'+ str(source) +'_'+str(target)+'_emb.mat'
    source_feat = sio.loadmat(path1)
    X_s = source_feat['rep_S']
    X_t = source_feat['rep_T']

    d1 = X_s
    d2 = X_t
    [num, fs1] = X_s.shape
    if fs1 > 1:
        pca = PCA(n_components=1)
        d1 = pca.fit_transform(X_s)[:, 0]
        # print('d1.shape:', d1.shape)
    [num, fs2] = X_t.shape
    if fs2 > 1:
        pca = PCA(n_components=1)
        d2 = pca.fit_transform(X_t)[:, 0]
        # print('d2.shape:', d2.shape)
    dis = wasserstein_distance(d1, d2)

    print('END!')
    print(source, '-', target, ' distance：', dis)
    return dis