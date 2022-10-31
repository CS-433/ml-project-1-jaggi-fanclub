import io, os, sys, types
import numpy as np
# Useful starting lines
import numpy as np
from implementations import *
from proj1_helpers import *
from cross_validation import *
from cleanup_helpers import *

dataset_variables = ["DER_mass_MMC",
 "DER_mass_transverse_met_lep", 
 "DER_mass_vis",
 "DER_pt_h", 
 "DER_deltaeta_jet_jet", 
 "DER_mass_jet_jet", 
 "DER_prodeta_jet_jet", 
 "DER_deltar_tau_lep", 
 "DER_pt_tot",
 "DER_sum_pt", 
 "DER_pt_ratio_lep_tau", 
 "DER_met_phi_centrality", 
 "DER_lep_eta_centrality", 
 "PRI_tau_pt",
 "PRI_tau_eta", 
 "PRI_tau_phi",
 "PRI_lep_pt", 
 "PRI_lep_eta",
 "PRI_lep_phi", 
 "PRI_met", 
 "PRI_met_phi",
 "PRI_met_sumet", 
 "PRI_jet_num", 
 "PRI_jet_leading_pt", 
 "PRI_jet_leading_eta", 
 "PRI_jet_leading_phi",
 "PRI_jet_subleading_pt", 
 "PRI_jet_subleading_eta",
 "PRI_jet_subleading_phi", 
 "PRI_jet_all_pt"]

#Index of the jet_num variable, useful later
jet_num_index = dataset_variables.index('PRI_jet_num')

print("Loading data.")
#Loads train data
data_train_path = 'data/train.csv' 
y_train, tx_train, ids_train = load_csv_data(data_train_path)

#Loads test data
data_test_path = 'data/test.csv' 
_, tx_test, ids_test = load_csv_data(data_test_path)

jets = [0, 1, 2, 3]

#New datasets for y, tx, ids all split in 4
print("Splitting the subsets into jet subsets.")
y_split, tx_split, ids_split = prepare_into_jet_subsets(y_train, tx_train, ids_train,jets, jet_num_index)

#Remove the data features where every entry is nan, saves the removed feature's indices in del_indices.
print("Removing nan features.")
del_indices = {}
for jet in jets:
    id_nans = np.where(np.isnan(tx_split[jet]))
    
    nan_index, nan_counts = np.unique(id_nans[1], return_counts=True)
    indices_todel = nan_index[nan_counts == len(tx_split[jet][:,0])]
    
    del_indices[jet] = indices_todel
    
    tx_split[jet] = np.delete(tx_split[jet], indices_todel, axis=1)

#Turns every remaining nan into the mean of the variable in that jet subset.
print("Setting the remaining nans to the mean of the feature.")
for jet in jets:
        tx_split[jet] = nans_to_mean(tx_split, jet)

#Runs ridge regression, once for each jet
print("Starting ridge regression.")
best_degrees = {}
w_preds = {}
for jet in jets:
    degrees = np.arange(2,4)
    k_fold = 3
    lambdas = np.logspace(-4, 0, 30)
    #Computes the best degree and lambda
    best_degree, best_lambda, _ = best_degree_selection(tx_split[jet], y_split[jet],
                                                        degrees, k_fold, lambdas, seed = 1)
    #Saves the degree for predictions
    best_degrees[jet] = best_degree
    
    poly_tr = build_poly(tx_split[jet], best_degree)
    w_pred, _ = ridge_regression(y_split[jet], poly_tr, best_lambda)

    w_preds[jet] = w_pred
    print("Ridge done for jet subset nr", jet)

print("Processing the test data.")
#Splits the data for testing in the same way as for the training data
_, te_tx_split, te_ids_split = prepare_into_jet_subsets(np.array(0), tx_test, ids_test, jets, jet_num_index)

#Removes the same data features as for the train data
for jet in jets:
    te_tx_split[jet] = np.delete(te_tx_split[jet], del_indices[jet], axis=1)

#Sets the remaing nans to the mean of the variable
for jet in jets:
    te_tx_split[jet] = nans_to_mean(te_tx_split, jet)

#Predicts the labels, using the weights predicted for the corresponding jet
print("Predicting the labels.")
predicted_labels = {}
for jet in jets:
    poly_te = build_poly(te_tx_split[jet], best_degrees[jet])
    predicted_labels[jet] = predict_labels(w_preds[jet], poly_te)

#Concatenates and saves the results
print("Saving the results.")
pred_labels_final = np.concatenate((predicted_labels[0], predicted_labels[1], predicted_labels[2], predicted_labels[3]))
ids_final = np.concatenate((te_ids_split[0], te_ids_split[1], te_ids_split[2], te_ids_split[3]))
create_csv_submission(ids_final, pred_labels_final, "prediction.csv")
