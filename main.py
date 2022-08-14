from calculateDIS import calculateEMDDIS
import utils
import scipy.io as sio
import numpy as np
import scipy.io


# Calculate the distance between each source domain and target domain
def load_distances(source_dataset,target):
    EMDdis = []
    sourceslen = len(source_dataset)
    for i in range(sourceslen):
        EMDdis.append(calculateEMDDIS(source_dataset[i], target))
    print('distance：',EMDdis)
    return EMDdis

# Calculate the weight
def dist2weight(distances):
    distances = np.array([-d ** 2 / 2 for d in distances])
    weights = [np.exp(d) for d in distances]

    return weights



target='MIR_pT2I'
dataset=['CLEF_pT2I','PASCAL_pT2I']
#'CLEF_pT2I','MIR_pT2I','PASCAL_pT2I','NUS_pT2I'
source_dataset = [d for d in dataset if d != target]
print(source_dataset)

path='./input/' + str(target) + '.mat'
target_net = sio.loadmat(path)
Y_true=target_net['group']

# Measuring the distance between source domain and target domain by EMD
distances = load_distances(source_dataset,target)
weights = dist2weight(distances)
print('weights：',weights)

# Send target domain data to each trained model to obtain classification results
list=[]
for source in source_dataset:
    filename = 'Label_Pred/' + str(source) + '_' + str(target) + '_predlabel.mat'
    net = sio.loadmat(filename)
    pred=net['group_T']
    list.append(pred)

multi_w=[]
final_pred=np.zeros(Y_true.shape)
for i in range(len(list)):
    _pred=list[i] # Represents the classification results of each model for the target domain
    w=weights[i]
    multi_w.append(w*_pred)
for j in range(len(multi_w)):
    final_pred+=multi_w[j]


F1_t = utils.f1_scores(final_pred, Y_true)  # Calculate F1_score
print('Final-----Target %s testing micro-F1: %f, macro-F1: %f' % (target,F1_t[0], F1_t[1]))
acc_t = utils.accuracy(final_pred, Y_true)
print('Target %s  testing accuracy: %f' % (target, acc_t))

result_save_path='./Final_Label/'+target+'.mat'
scipy.io.savemat(result_save_path, {'group_T': F1_t[2]})