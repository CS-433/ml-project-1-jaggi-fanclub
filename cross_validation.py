import numpy as np
from implementations import *

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    """
    poly = x
    
    for i in range(2,degree+1):
       
        poly = np.c_[poly,x**i]
    return poly



def cross_validation(x, y, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """

    # ***************************************************
    # INSERT YOUR CODE HERE

    indice_test_k = k_indices[k]
    x_test = x[indice_test_k,:]
    y_test = y[indice_test_k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_train = y[tr_indice]
    x_train = x[tr_indice]

    # ***************************************************
   
    # ***************************************************
    # INSERT YOUR CODE HERE
    poly_tr = build_poly(x_train, degree)
    poly_te = build_poly(x_test, degree)

    # ***************************************************
   
    # ***************************************************
    # INSERT YOUR CODE HERE
    w_star,_ = ridge_regression(y_train,poly_tr,lambda_)
    
    # ***************************************************

    # ***************************************************
    # INSERT YOUR CODE HERE
   
    loss_tr = np.sqrt(2*compute_loss(y_train, poly_tr, w_star))
    loss_te = np.sqrt(2*compute_loss(y_test, poly_te, w_star))
    # ***************************************************
    
    return loss_tr, loss_te



from plots_lab4 import cross_validation_visualization

def cross_validation_demo(x,y,degree, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 1
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    for lambda_ in lambdas:
        rmse_tr_k = []
        rmse_te_k = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(x, y, k_indices, k, lambda_, degree)
            rmse_tr_k.append(loss_tr)
            rmse_te_k.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_k))
        rmse_te.append(np.mean(rmse_te_k))
    # ***************************************************
    
    ind_best_rmse = np.where(rmse_te<=np.min(rmse_te))
    best_rmse = np.min(rmse_te)
    best_lambda = lambdas[ind_best_rmse]
    #cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    #print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse


def best_degree_selection(x,y,degrees, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        

    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    best_lamba_all_degree = []
    best_rmse_all_degree = []
    for degree in degrees:
        best_lambda_degree, best_rmse_degree = cross_validation_demo(x,y,degree, k_fold, lambdas)
        best_lamba_all_degree.append(best_lambda_degree)
        best_rmse_all_degree.append(best_rmse_degree)
    
    ind_best_rmse = np.argmin(best_rmse_all_degree)
    
    best_rmse = np.min(best_rmse_all_degree)
    best_lambda = best_lamba_all_degree[ind_best_rmse] 
    best_degree = degrees[ind_best_rmse]
    # ***************************************************
  
    
    return best_degree, best_lambda, best_rmse


def split_data(x, y, ratio, seed):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
        

    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    l = int(np.floor(len(y)*ratio))
    y = np.random.permutation(y)
    np.random.seed(seed)
    x = np.random.permutation(x)
    
    return (x[0:l],x[l:],y[0:l],y[l:])
    
    
    # ***************************************************
    raise NotImplementedError
