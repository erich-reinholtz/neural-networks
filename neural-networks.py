import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from scipy import special

def sigmoid(a):
    siga = 1/(1 + special.expit(-a))
    return siga
    

class nn_one_layer():
    def __init__(self, input_size, hidden_size, output_size):
        #define the input/output weights W1, W2
        # self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        # self.W2 = 0.1 * np.random.randn(hidden_size, output_size)

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / 128)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / 64)
        
        self.f = sigmoid
    
    def forward(self, u):
        z = np.matmul(u, self.W1)
        h = self.f(z)
        v = np.matmul(h, self.W2)
        return v, h, z


# de la greg
def load_data():
    mndata = MNIST('./sets')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_onehot = []
    test_onehot = []
    for label in train_labels:
        one_hot = np.zeros(10)
        one_hot[:] = 0.01
        one_hot[label] = 0.99
        train_onehot.append(one_hot)
    for label in test_labels:
        one_hot = np.zeros(10)
        one_hot[:] = 0.01
        one_hot[label] = 0.99
        test_onehot.append(one_hot)

    train_images = (np.array(train_images)*(0.99/255))+0.1
    test_images = (np.array(test_images)*(0.99/255))+0.1
    
    return train_images, np.array(train_onehot), test_images, np.array(test_onehot)

# tot de la greg
def generate_batch(inputs, targets, batch_size):
    #randomly choose batch_size many examples; note there will be
    #duplicate entries when batch_size > len(dataset) 
    rand_inds = np.random.randint(0, len(inputs), batch_size)
    inputs_batch = inputs[rand_inds]
    targets_batch = targets[rand_inds]
    
    return inputs_batch, targets_batch



#loss function as defined above
def loss_mse(preds, targets):
    loss = np.sum((preds - targets)**2)
    return 0.5 * loss

#derivative of loss function with respect to predictions
def loss_deriv(preds, targets):
    dL_dPred = preds - targets
    return dL_dPred


#derivative of the sigmoid function
def sigmoid_prime(a):
    dsigmoid_da = sigmoid(a)*(1-sigmoid(a))
    return dsigmoid_da

#compute the derivative of the loss wrt network weights W1 and W2
#dL_dPred is (precomputed) derivative of loss wrt network prediction
#X is (batch) input to network, H is (batch) activity at hidden layer
def backprop(W1, W2, dL_dPred, U, H, Z):
    #hints: for dL_dW1 compute dL_dH, dL_dZ first.
    #for transpose of numpy array A use A.T
    #for element-wise multiplication use A*B or np.multiply(A,B)
    
    dL_dW2 = np.matmul(H.T, dL_dPred)
    dL_dH = np.matmul(dL_dPred, W2.T)
    dL_dZ = np.multiply(sigmoid_prime(Z), dL_dH)
    dL_dW1 = np.matmul(U.T, dL_dZ) 
    
    return dL_dW1, dL_dW2



#train the provided network with one batch according to the dataset
#return the loss for the batch
def train_one_batch(nn, inputs, targets, lr):
    preds, H, Z = nn.forward(inputs)

    loss = loss_mse(preds, targets)

    dL_dPred = loss_deriv(preds, targets)
    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= lr * dL_dW1
    nn.W2 -= lr * dL_dW2
    
    return loss

# #test the network on a given dataset
# def test(nn, dataset):
#     # inputs, targets = generate_batch(dataset, batch_size=200)

#     # inputs, targets, _, _ = load_data()


#     # preds, H, Z = nn.forward(inputs) 
#     preds, _, _ = nn.forward(inputs) 
#     loss = loss_mse(preds, targets)
#     return loss


# chosen_dataset = dataset_xor

input_size = 784
hidden_size = 50
output_size = 10

batch_size = 10 #number of examples per batch
nbatches = 10000 #number of batches used for training
# lr = 0.1 #learning rate
lr = 0.05

inputs, targets, _, _ = load_data()

nn = nn_one_layer(input_size, hidden_size, output_size)

losses = [] #training losses to record
for i in range(nbatches):
    inputBatch, targetBatch = generate_batch(inputs, targets, batch_size)
    loss = train_one_batch(nn, inputBatch, targetBatch, lr)
    losses.append(loss)


plt.plot(np.arange(1, nbatches+1), losses)
plt.xlabel("# batches")
plt.ylabel("training MSE")
plt.show()

