import numpy as np
import random as rnd


# Create matrix X
def createX(x, m):
    X = np.zeros(shape=((len(x), m + 1)))
    row_index = 0
    for x_elem in x:
        xrow = []
        for i in range(m + 1):
            xrow.append(pow(x_elem, i))
        X[row_index:] = xrow
        row_index += 1
        # print 'rank: ', np.rank(np.mat(X))
    return np.mat(X)  # return matrix instead of ndarray


# Compute RMSE
def computeCost(X, y, w):
    n = y.size # Number of sample
    predictions = X * w # Predicition of y given current w
    err = predictions - y # error prediction
    # RMSE
    sqErr = np.power(err, 2)
    sqErrSum = sqErr.sum()
    J = (1.0 / (2 * n)) * sqErrSum
    return J


# Standard Regression
def standReg(x, y, m):
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    XTX = X.T * X
    if np.linalg.det(XTX) == 0.0:
        print "The XTX matrix is singular, cannot do inverse"
        return
    w = XTX.I * X.T * y
    print 'w_stand: ', w
    J = computeCost(X, y, w)
    return w, J


# Gradient Descent Regression
def gradDescent(x, y, m, num_iters=1500, alpha=0.0001):
    """
    Regression using Gradient Descent

    x = X point's sample
    y = Y point's sample
    m = Order
    num_iters = number of iteration, default 1500
    alpha = learning rate alpha, default 0.0001
    """
    n = y.size # Number of sample, also the same with do x.size
    X = createX(x, m)  # create X
    y = np.mat(y) # convert y to matrix data type
    y = y.T  # prepare y
    w = np.mat(np.zeros(shape=(m + 1, 1))) # Prepare w with ALL ZERO value
    E_history = np.zeros(shape=(num_iters, 1)) # Store error history for each iteration
    prevE = float("inf") # assign initial previous error as Infinite (very large error)
    for i in range(num_iters): # Loop for 'num_iters' iterations
        E = computeCost(X, y, w) # compute current error of predicition given w
        if E < prevE: # Make sure current error is convergent (always smaller)
            # Some of following expression can be optimized because in computeCost() we have compute some of the value
            predictions = X * w # calculate current prediction
            diff = predictions - y # error of prediction
            for j in range(m + 1): # iteration to update all of the w values
                error = diff.T * X[:, j] # compute prediction error of each order
                # w_i' = w_i - alpha * (sum of regression errors of orde i-th) * X_i
                w[j][0] = w[j][0] - alpha * (1.0 / n) * error.sum() # update w, TODO: why 1/n * sum of errors?
            E_history[i, 0] = E  # computeCost(X, y, w), store error history
            prevE = E # update previous error as current error
        else: # If the error increased, halt process and throws error
            print 'diverged! try using smaller alpha'
            break # break loop
    return w, E_history


# Stochastic Gradient Descent using 1 random sample
def stocGradDescent(x, y, m, num_iters=1500, alpha=0.0001):
    """
    Regression using Stochastic Gradient Descent with one random sample choosen

    x = X point's sample
    y = Y point's sample
    num_iters = number of iterations, default 1500
    alpha = learning rate alpha, default 0.0001
    """
    n = y.size
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    w = np.mat(np.zeros(shape=(m + 1, 1))) # initialize w with all ZERO value
    E_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        dataIndex = rnd.randint(0, n - 1) # choose ONE random index
        E = computeCost(X[dataIndex], y[dataIndex], w) # compute prediction error using the choosen index
        # Some of following expression can be optimized because in computeCost() we have compute some of the value
        predictions = X[dataIndex] * w # the prediction value of prediction
        diff = predictions - y[dataIndex] # prediction error
        for j in range(m + 1): # update all of w values
            error = diff.T * X[dataIndex, j] # compute prediction error of each order using choosen sample
            w[j][0] = w[j][0] - alpha * (1.0 / n) * error
        # we can't MAINTAIN THE ERROR to be CONVERGENT, because we ONLY look at SOME data sample, NOT ALL the data!
        E_history[i, 0] = E  # computeCost(X, y, w)
    return w, E_history


# Stochastic Gradient Descent using more than one sample
def stocGradDescent2(x, y, m, num_iters=1500, alpha=0.0001, sample_size=4):
    """
    Regression using Stochastic Gradient Descent with more than one random sample choosen

    x = X point's sample
    y = Y point's sample
    num_iters = number of iterations, default 1500
    alpha = learning rate alpha, default 0.0001
    sample_size = random sample choosen, default 4 random sample
    """
    n = y.size
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    w = np.mat(np.zeros(shape=(m + 1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        randIndex = rnd.sample(range(len(X)), sample_size)
        E = computeCost(X[randIndex], y[randIndex], w)
        predictions = X[randIndex] * w
        diff = predictions - y[randIndex]
        for j in range(m + 1):
            error = diff.T * X[randIndex, j]
            w[j][0] = w[j][0] - alpha * (1.0 / n) * error.sum()

        E_history[i, 0] = E  # computeCost(X, y, w)
    return w, E_history


# TODO: EXPERIMENTS ONLY
def gradDescentWRand(x, y, m, num_iters=1500, alpha=0.0001):
    n = y.size # Number of sample, also the same with do x.size
    X = createX(x, m)  # create X
    y = np.mat(y) # convert y to matrix data type
    y = y.T  # prepare y
    # randW = np.mat(np.random.normal(size=m+1))
    # w = randW.T
    w = np.mat(np.zeros(shape=(m + 1, 1))) # Prepare w with ALL ZERO value

    # randomize intial w
    randW = rnd.randint(0, n - 1) / n
    for i in range(m+1):
        if i == 0:
            continue
        w[i] = np.power(randW, i)

    E_history = np.zeros(shape=(num_iters, 1)) # Store error history for each iteration
    prevE = float("inf") # assign initial previous error as Infinite (very large error)
    for i in range(num_iters): # Loop for 'num_iters' iterations
        E = computeCost(X, y, w) # compute current error of predicition given w
        if E < prevE: # Make sure current error is convergent (always smaller)
            # Some of following expression can be optimized because in computeCost() we have compute some of the value
            predictions = X * w # calculate current prediction
            diff = predictions - y # error of prediction
            for j in range(m + 1): # iteration to update all of the w values
                error = diff.T * X[:, j] # compute prediction error of each order
                # w_i' = w_i - alpha * (sum of regression errors of orde i-th) * X_i
                w[j][0] = w[j][0] - alpha * (1.0 / n) * error.sum() # update w, TODO: why 1/n * sum of errors?
            E_history[i, 0] = E  # computeCost(X, y, w), store error history
            prevE = E # update previous error as current error
        else: # If the error increased, halt process and throws error
            print 'diverged! try using smaller alpha'
            break # break loop
    return w, E_history

# Create Model
def createModel(x, w):
    # print 'X = ', x
    y = np.zeros(len(x))
    y = np.mat(y)
    y = y.T  # prepare y
    pwr = np.arange(len(w))
    for wi, p in zip(w, pwr):
        accum = np.mat(wi * (x ** p))
        y = y + accum.T
    return np.squeeze(np.asarray(y))




