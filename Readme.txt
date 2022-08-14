-CODE
    -Final_Label：Store the node label prediction results of the final target domain
    -Label_Pred：Store the node label prediction results obtained by feeding the target domain into multiple trained model classifiers
    -model:Store each trained model
    -Train_result：Store the node feature representation, domain representation, and label prediction results
    -calculate.py：Calculate the EMD distance between the target domain and each source domain
    -trainModel.py：Main code for model training
    -flip_gradient.py：Code for gradient descent
    -globavar.py:File for handling global variables
    -M2CDNE_model.py：Model
    -main.py：Evaluate the effect of the whole model on target domain
    -S1_train.py：Use each source domain and target domain for training, train multiple classifiers and domain discriminators, and save each model
    -S2_getPrediction.py：Feed the target domain into each classifier to obtain individual predictions
    -utils.py：utility function


Step 1:run S1_train.py to train the model(Contains running M2CDNE_model.py and trainModel.py)
Step 2:run S2_getPrediction.py to feed the target domain into each classifier to obtain individual predictions
Step 3:run main.py（Contains running calculateDIS.py to get weights）to get the final classification result of the weighted target domain


