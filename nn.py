import numpy as np



def generate_layer_theta(X,Y, hiden_layers):
    thetas = []
    errs = []
    input_size = len(X[0])
    output_size = len(Y[0])
    thetas.append(np.random.randn(hiden_layers[0], input_size + 1))
    errs.append(np.zeros((hiden_layers[0], input_size + 1)))
    for i in range(1, len(hiden_layers)):
        thetas.append(np.random.randn(hiden_layers[i], hiden_layers[i - 1] + 1))
        errs.append(np.zeros((hiden_layers[i], hiden_layers[i - 1] + 1)))
    thetas.append(np.random.randn(output_size, hiden_layers[-1] + 1))
    errs.append(np.zeros((output_size, hiden_layers[-1] + 1)))
    return thetas, errs

def generate_layer_structure(input_shape, hidden_shape, output_shape):
    layers = []
    input_layer = (hidden_shape[0],input_shape + 1)
    layers.append(input_layer)
    for i in range(1, len(hidden_shape)):
        layers.append((hidden_shape[i], layers[i-1][0] + 1))
    layers.append((output_shape, layers[-1][0] + 1))
    return layers


def forward_propagate(x, thetas):
    neurons = []
    a1 = np.insert(x, 0, [1]) # add bias term
    neurons.append(a1.reshape(a1.shape[0], 1))
    for i in range(0, len(thetas)):
        z = thetas[i].dot(neurons[i])
        res = 1 / (1 + np.exp(-z))
        if i < len(thetas) - 1:
            res = np.insert(res, 0, [1]) # add bias term for all layers except output layer
        neurons.append(res.reshape(res.shape[0], 1))
    return neurons


def aggregate_errs(agg_errs, forward_neurons, backward_errs):
    for i in range(0, len(backward_errs)):
        if i < len(backward_errs) - 1:
            agg_errs[i] += backward_errs[i][1:].dot(forward_neurons[i].T)
        else:
            agg_errs[i] += backward_errs[i].dot(forward_neurons[i].T)
    return agg_errs


def backward_propagate(y, thetas,forward_neurons):
    neuron_errs = []
    layer_err = forward_neurons[-1] - y
    total_layer = len(thetas)
    neuron_errs.append(layer_err)
    for i in range(0, total_layer - 1):
        idx = total_layer - 1 - i
        dfz = np.multiply(forward_neurons[idx], (1 - forward_neurons[idx]))
        if i == 0:
            err = thetas[idx].T.dot(neuron_errs[i])
        else:
            err = thetas[idx].T.dot(neuron_errs[i][1:])
        neuron_err = np.multiply(err, dfz)
        neuron_errs.append(neuron_err)
    neuron_errs.reverse()
    return neuron_errs


def update_theta(m,thetas, agg_errs,learning_rate, C):
    for i, theta in enumerate(thetas):
        derr = agg_errs[i]/m
        regularized_term = C*(np.hstack((theta[:,1:],np.zeros((theta.shape[0],1)))))
        D = derr + regularized_term
        thetas[i] = theta - learning_rate*D
    return thetas


# X: train data, Y: train label, steps: max steps for gradient descent, learning_rate: learning rate for gradient decent
# C: regularized term

def train_mlp(X,Y,lshape,steps=3000,learning_rate=0.001, C=0.0001):
    assert isinstance(lshape, tuple)
    layer_structure = generate_layer_structure(len(X[0]), lshape, len(Y[0]))
    thetas = [np.random.randn(layer[0],layer[1]) for layer in layer_structure]
    agg_errs = [np.zeros(layer) for layer in layer_structure]
    prev_cost = 0
    for stp in range(0, steps):
        total_cost = 0
        for i, x in enumerate(X):
            forward_neurons = forward_propagate(x, thetas)
            backward_errs = backward_propagate(Y[i], thetas, forward_neurons)
            agg_errs = aggregate_errs(agg_errs, forward_neurons, backward_errs)
            cost = Y[i][0]*np.log(forward_neurons[-1][0]) + (1-Y[i][0])*np.log(1-forward_neurons[-1][0])
            total_cost += cost[0]
        print("[" +str(stp) + "]" + "Cost : " + str(total_cost))
        if abs(total_cost - prev_cost) <= 0.0001 or total_cost == 'na':
            break
        else:
            prev_cost = total_cost
        thetas = update_theta(len(X),thetas,agg_errs,learning_rate,C)
    return thetas

def test(thetas):
    correct = 0
    tX,tY = generate_sample()
    for j, x in enumerate(tX):
        fn = forward_propagate(x, thetas)[-1][0][0]
        mf = 0 if fn < 0.5 else 1
        if mf == tY[j][0]: correct += 1
        print("Predict : " + str(mf) + ",True: " + str(tY[j][0]))
    print("Accuracy Score: " + str(correct/len(X)))




# def train(X,Y):
#     init_theta1 = np.random.randn(4,3)
#     init_theta2 = np.random.randn(1,4)
#     err_theta1 = np.zeros((4, 3))
#     err_theta2 = np.zeros((1, 4))
#     for stp in range(0,200):
#
#         for i, x in enumerate(X):
#             Z2 = init_theta1.dot(x)
#             A2 = 1/(1+np.exp(-Z2))
#             Z3 = init_theta2.dot(A2)
#             A3 = 1/(1+np.exp(-Z3))
#             err_A3 = A3 - Y[i]
#             dfa2 = np.multiply(A2,(1-A2))
#             err_A2 = np.multiply(init_theta2.T.dot(err_A3),dfa2)
#             err_theta2 += err_A3.dot(A2.reshape(1,4))
#             err_theta1 += err_A2.reshape(4,1).dot(x.reshape(1,3))
#         init_theta1 =init_theta1 - 0.005*err_theta1/len(X)
#         init_theta2 =init_theta2 - 0.005*err_theta2/len(X)
#
#     correct = 0
#     tX = np.random.randn(500,3)
#     theta_true = [1,2,1]
#     tY = [[0] if mm < 0.5 else [1] for mm in 1/(1+np.exp(-tX.dot(theta_true)))]
#     for j in range(0, len(tX)):
#         mm = 1 / (1 + np.exp(-init_theta1.dot(tX[j])))
#         ff = 1 / (1 + np.exp(-init_theta2.dot(mm)))
#         mf = 0 if ff < 0.5 else 1
#         if mf == tY[j][0] : correct += 1
#         print("Predict : " + str(mf) + ",True: " + str(tY[j][0]))
#     print("Accuracy : " + str(correct/len(tY)))

def generate_sample():
    X = np.random.randn(sample_shape[0], sample_shape[1])
    Y = [[0] if mm < 0.5 else [1] for mm in 1 / (1 + np.exp(-X.dot(theta_true)))]
    return X,Y



if __name__=="__main__":
    sample_shape = (500,4)
    theta_true = [1,2,1,2]
    X,Y = generate_sample()
    model = train_mlp(X,Y,(4,4))
    test(model)
    # train(X,Y)
    # print(y_True)
