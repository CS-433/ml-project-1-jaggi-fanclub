import numpy as np


#
#Function1
def compute_mse(e: np.ndarray) -> np.float64:
    """Computes the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.float64:
    """Computes the loss.
    """
    e = y - tx.dot(w)
    return compute_mse(e)

    
def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.float64]:
    """Gradient descent algorithm."""
    w = initial_w
    if max_iters == 0:
        loss = compute_loss(y, tx, w)
        
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = compute_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
    grad, err = compute_gradient(y, tx, w)
    loss = compute_mse(err)
    return w,loss

##########################
#Function2
                          
def batch_iter(y: np.ndarray, tx: np.ndarray, batch_size, num_batches: int = 1, shuffle: bool = True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

                          
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss

    w = initial_w
    batch_size = 1 #SGD == minibatch method batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # compute loss
            loss = compute_loss(y, tx, w)
            # store w and loss

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

##########################
#Function3
                          
def least_squares(y, tx):
    """compute the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    
    #Computes the optimal w
    w = np.linalg.inv(tx.T@tx)@tx.T@y
    mse = compute_loss(y, tx, w)
    return w, mse

##########################
#Function4

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.


    """
    shape = (tx.T@tx).shape
    shape_y = y.shape
    a = tx.T@tx + 2*shape_y[0]*lambda_*np.eye(shape[0])
    b = tx.T@y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)                   
    return w,loss
    
    
##########################

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return np.exp(t)/(1+np.exp(t))

def compute_loss_reg(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    N = y.shape[0]
    sig_txw = sigmoid(tx@w)
    
    loss = y.T@(np.log(sig_txw)) + (1 - y).T@(np.log(1 - sig_txw))

    return -np.sum(loss)/N

def compute_gradient_reg(y, tx, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)

    """
    #Applies the formula as proven in class
    return 1/y.shape[0]*tx.T@(sigmoid(tx@w)-y)

#Function5
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do max_iters steps of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1) 
        max_iters: int
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1) 
    """
   
    w = initial_w
    for iter in range(max_iters):
        #Update w
        w = w - gamma* compute_gradient_reg(y, tx, w)
    loss = compute_loss_reg(y,tx,w)
    
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """Returns the loss and gradient for logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    """

    loss = compute_loss_reg(y, tx, w) + lambda_ * float(w.T@w)
    gradient = compute_gradient_reg(y, tx, w) + 2 * lambda_*w

    return loss, gradient

#Function6                          
def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """
    Do max_iters steps of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar
        initial_w:  shape=(D, 1)
        max_iters: int
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        #Compute loss and update w.
        _, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        
    #Compute loss without penalization ???? TODO: check if bug in tests
    loss = compute_loss_reg(y, tx, w)
    
    return w, loss                       
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                
                         
                  
