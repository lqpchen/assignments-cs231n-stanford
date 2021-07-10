from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_rows = X.shape[0] # number of rows
    
    for i in range(num_rows):
        score_i = X[i].dot(W)
        score_i -= np.max(score_i)
        exp_score_i = np.exp(score_i)
        loss += -1*np.log(exp_score_i[y[i]]/exp_score_i.sum())
        
        dW_inc = X[[i]].T@(exp_score_i/exp_score_i.sum()).reshape(1, -1)
        dW_inc[:, y[i]] = (exp_score_i[y[i]]/exp_score_i.sum() - 1)*X[i]
        dW += dW_inc
    loss/=num_rows
    loss += reg*np.sum(W*W)
    dW/=num_rows
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_rows = X.shape[0]
    XW = X@W
    XW -= XW.max(axis=1).reshape(-1, 1)
    expXW = np.exp(XW)
    expXW_max = expXW[np.arange(num_rows), y]
    Ls = -1*np.log(expXW_max/expXW.sum(axis=1))
    loss = Ls.sum()/num_rows
    loss += reg*np.sum(W*W)    

    mat = expXW/expXW.sum(axis=1).reshape(-1, 1)
    mat[np.arange(num_rows), y] -= 1
    dW = X.T@mat/num_rows + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
