#Standardizdation process for the data
import numpy as np



def standardize(x):
    centered_data = x - np.nanmean(x, axis=0)
    std_data = centered_data / np.nanstd(centered_data, axis=0)
    
    return std_data

#Method that splits the dataset accordin to the jet number
def prepare_into_jet_subsets(y, tx, ids, jets, jet_num_index):
    tx[np.where(tx == -999)] = np.nan
    std_tx = standardize(tx)
    std_tx[:, jet_num_index] = tx[:, jet_num_index]
    
    y_split_temp = {}
    tx_split_temp = {}
    split_ids_temp = {}

    #Splits the data into 4 subsets, one for each jet number.
    for jet in jets:
        jet_ids = np.where(std_tx[:, jet_num_index] == jet)
        split_ids_temp[jet] = ids[jet_ids]
        if(y.shape != ()):
            y_split_temp[jet] = y[jet_ids]
        tx_split_temp[jet] = std_tx[jet_ids]
    return y_split_temp, tx_split_temp, split_ids_temp


#Sets any remaining nan variable to the mean of the feature.
def nans_to_mean(tx_split_ntm, jet):
    #Initialize variables
    tx_jet = tx_split_ntm[jet]
    means_without_outliers = np.zeros(tx_jet.shape[1])
    counts_without_outliers = np.zeros(tx_jet.shape[1])
    
    #Counts the number of valid entries and sums their value
    for entry in tx_jet:
        for id_point, point in enumerate(entry):
            if(not np.isnan(point)):
                means_without_outliers[id_point] += point
                counts_without_outliers[id_point] += 1
            
    #Computes the mean of valid entries
    means_without_outliers /= counts_without_outliers
    
    tx_jet_no_nan = tx_jet.copy()
    bad_variables = np.where(np.isnan(tx_jet))

    #Replaces every nan variable by the mean of that variable
    for entry, point in zip(bad_variables[0], bad_variables[1]):
        tx_jet_no_nan[entry, point] = means_without_outliers[point]
    return tx_jet_no_nan