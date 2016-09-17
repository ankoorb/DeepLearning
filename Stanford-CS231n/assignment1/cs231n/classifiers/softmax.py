import numpy as np
from random import shuffle

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
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    probs = np.exp(scores)/np.sum(np.exp(scores))
    correct_class_prob = probs[y[i]]
    loss -= np.log(correct_class_prob)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += (probs[y[i]] - 1) * X[i]
      else:
        dW[:, j] += probs[j] * X[i]
            
  # Average loss
  loss /= num_train
  dW /= num_train
  
  # Add regularization loss
  loss += 0.5 * reg * np.sum(W * W)      
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  
  max_scores = np.amax(scores, axis=1)
  scores -= max_scores[:, np.newaxis]
  scores_exp = np.exp(scores)
  scores_exp_sum = np.sum(scores_exp, axis=1)
  probs = scores_exp/scores_exp_sum[:, np.newaxis]
  correct_class_probs = probs[np.array(xrange(num_train)), y]
  loss -= np.sum(np.log(correct_class_probs))
  
  # Gradient 
  mask = np.zeros(probs.shape)
  mask[np.array(xrange(probs.shape[0])), y] = 1.0
  probs -= mask
  dW = X.T.dot(probs)
  #d_probs = probs
  #d_probs[np.array(xrange(num_train)), y] -= 1.0
  #dW = X.T.dot(d_probs)
  
  # Average loss
  loss /= num_train
  dW /= num_train
  
  # Add regularization loss
  loss += 0.5 * reg * np.sum(W * W)      
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

