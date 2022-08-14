#Step 1: Use each source domain and target domain for training, train multiple classifiers and domain discriminators, and save each model


import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from scipy.sparse import lil_matrix
from trainModel import train_and_evaluate

tf.set_random_seed(0)
np.random.seed(0)

source='PASCAL_pT2I'
target='MIR_pT2I'
#'CLEF_pT2I','MIR_pT2I','PASCAL_pT2I','NUS_pT2I'


emb_filename='Train_result/'+str(source)+'_'+str(target) # Save the node representation obtained by the model
label_filename='Train_result/'+str(source)+'_'+str(target) # Save the predicted node label obtained by the model
domain_filename='Train_result/'+str(source)+'_'+str(target) # Save the predicted domain label obtained by the model
model_filename='model/'+str(source)+'_'+str(target)+'/model.ckpt' # Save the model
Kstep=3


# Load source data
A_s, X_s, Y_s= utils.load_network('./input/'+str(source)+'.mat')
# Compute PPMI
A_k_s=utils.AggTranProbMat(A_s, Kstep)
PPMI_s=utils.ComputePPMI(A_k_s)
n_PPMI_s=utils.MyScaleSimMat(PPMI_s)    # row normalized PPMI
X_n_s=np.matmul(n_PPMI_s,lil_matrix.toarray(X_s)) #neibors' attribute matrix


# Load target data
A_t, X_t, Y_t = utils.load_network('./input/'+str(target)+'.mat')
# Compute PPMI
A_k_t=utils.AggTranProbMat(A_t, Kstep)
PPMI_t=utils.ComputePPMI(A_k_t)
n_PPMI_t=utils.MyScaleSimMat(PPMI_t)   # row normalized PPMI
X_n_t=np.matmul(n_PPMI_t,lil_matrix.toarray(X_t)) #neibors' attribute matrix


##input data
input_data=dict()
input_data['PPMI_S']=PPMI_s
input_data['PPMI_T']=PPMI_t
input_data['attrb_S']=X_s
input_data['attrb_T']=X_t
input_data['attrb_nei_S']=X_n_s
input_data['attrb_nei_T']=X_n_t
input_data['label_S']=Y_s
input_data['label_T']=Y_t

##model config
config=dict()
config['clf_type'] = 'multi-label'
config['dropout'] = 0.5
config['num_epoch'] = 10 ## maximum training iteration
config['batch_size'] = 100
config['n_hidden'] = [128,128] # dimensionality for each k-th hidden layer of FE1 and FE2
config['n_emb'] = 512  #embedding dimension
config['l2_w'] = 1e-3 #weight of L2-norm regularization
config['net_pro_w'] = 1e-3 #weight of pairwise constraint
config['lr_ini'] = 1e-3 #initial learning rate
config['emb_filename'] =emb_filename #output file name to save node representations
config['label_filename'] =label_filename #output file name to save node labels
config['domain_filename'] =domain_filename #output file name to save domain labels
config['model_filename'] =model_filename #output file name to save model



random_state=0
print ('source and target networks:',str(source),'--->',str(target))

#Train
train_and_evaluate(input_data, config,source,target,random_state)
