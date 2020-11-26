import numpy as np
import random
#import mnist_loader
import neural_net
from datetime import datetime
import sys
import os
import warnings

def getTimestamp():
    return datetime.now().strftime("%H:%M:%S")

logTraining = True
logSaving = True
productionCode = False

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def train(self, training_data, iterations, mini_batch_size, lr, savingRate,
            test_data=None): #minibatch stochastic gradient descent

        if test_data: n_test = len(test_data)
        n = len(training_data)

        if mini_batch_size == -1:
            mini_batch_size = n

        count = 0
        status = 0
        if logTraining:
            print(f"Started training with {iterations} iterations at ", getTimestamp())
        for i in range(iterations):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #die geshuffelte Trainigsliste wird in Minibatches unterteilt
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), n_test))
            else:
                #print("Epoch {0} complete".format(j))
                pass

            if count == savingRate:
                count = 0
                self.save()

            count += 1

            if (i / iterations * 100) - status > 0.1:
                status = round( i / iterations * 100, 2)
                if logTraining:
                    print(f"Status: {status}% at ", getTimestamp())
                #self.doTest()

        self.save()

    def update_mini_batch(self, mini_batch, lr):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x, y)
            gradient_b = [b+delta_b for b, delta_b in zip(gradient_b, delta_gradient_b)] #für mini batch updaten
            gradient_w = [w+delta_w for w, delta_w in zip(gradient_w, delta_gradient_w)]

        lr2 = lr/len(mini_batch)
        self.weights = [w-lr2*nw #gesamt updaten
                        for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b-lr2*nb
                       for b, nb in zip(self.biases, gradient_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            #print("shape a", np.shape(a), "\nshape w", np.shape(w))
            a = sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y) #1/2 * 2(out-y)^1

    def save(self):
        if productionCode:
            id = str(datetime.now().timestamp())
            newPath = neural_net.savePath + id
            os.mkdir(newPath)
            np.save(newPath+"\\weights.npy", self.weights, allow_pickle=True)
            np.save(newPath+"\\biases.npy", self.biases, allow_pickle=True)
            print(id)
        else:
            np.save(neural_net.savePath + "weights.npy", self.weights, allow_pickle=True)
            np.save(neural_net.savePath + "biases.npy", self.biases, allow_pickle=True)
            np.save(neural_net.savePath + "weights.npy -- Backup.npy", self.weights, allow_pickle=True)
            np.save(neural_net.savePath + "biases.npy -- Backup.npy", self.biases, allow_pickle=True)
            if logSaving:
                print("saved network " + getTimestamp())

    def load(self, id=-1):
        if productionCode:
            self.weights = np.load(neural_net.savePath +str(id)+ "\\weights.npy", allow_pickle=True)
            self.biases = np.load(neural_net.savePath +str(id)+ "\\biases.npy", allow_pickle=True)
            return

        self.weights = np.load(neural_net.savePath+"weights.npy", allow_pickle=True)
        self.biases = np.load(neural_net.savePath+"biases.npy", allow_pickle=True)
        if logSaving:
            print("loaded network "+getTimestamp())


    def save2(self):
        np.savetxt(neural_net.savePath+"weights.csv", self.weights, delimiter=",")
        np.savetxt(neural_net.savePath+"biases.csv", self.biases, delimiter=",")
        if logSaving:
            print("saved network")

    def load2(self):
        self.weights = np.loadtxt(neural_net.savePath+"weights.csv", delimiter=",")
        self.biases = np.loadtxt(neural_net.savePath+"biases.csv", delimiter=",")
        if logSaving:
            print("loaded network")

    def doTest(self):
        testpath = neural_net.savePath + "training\\data\\apfel\\"
        p = testpath + "1.jpg"
        if logTraining:
            print("Testing : ", p)
            print(self.feedforward(neural_net.getArrayData(p, np.zeros((5, 5)))[0]))


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def main2():
    global logSaving, logTraining, productionCode

    run = False
    train = False

    args = sys.argv
    formel = "2*x"
    try:
        if args[1] == "train":
            formel = args[2]
            log = False
            logTraining = False
            logSaving = False
            train = True
        elif args[1] == "run":
            logTraining = False
            logSaving = False
            run = True
        productionCode = True
    except Exception as e:
        print("ERROR", str(e))

    log = False

    sizes = [1, 10, 10, 1]  # [2700,500,100, 10, 2]
    # sizes = [3, 5,3, 1]
    net = Network(sizes)
    if not productionCode:
        try:
            net.load()
        except:
            net.save()

    training_data = neural_net.loadTrainingDataNumbers(formel)
    #training_data = [(np.array([[0, 0, 1]]).T, np.array([[0]])), (np.array([[0, 1, 1]]).T, np.array([[1]])), (np.array([[1, 0, 1]]).T, np.array([[1]])), (np.array([[0, 1, 0]]).T, np.array([[1]])), (np.array([[1, 0, 0]]).T, np.array([[1]])), (np.array([[1, 1, 1]]).T,np.array([[0]])), (np.array([[0, 0, 0]]).T, np.array([[0]]))]

    if log:
        print(training_data)

        print("weights")
        print(net.weights)

        print("biases")
        print((net.biases))

    if train:
        net.train(training_data, 10, 50, 2, 50)
    elif run:
        net.load(id=args[2])
        print(net.feedforward(np.array([[float(args[3]) / neural_net.NUMBERLIMIT]]))[0][0] * neural_net.NUMBERLIMIT)

    if log:
        tests = 5
        for x in range(0, 10000, int(1000/tests)):
            print("Input: ")
            print(x)
            print("Correct: ")
            print(neural_net.trainingFunction(x, formel))
            print("Prediction: ")
            print(net.feedforward(np.array([[x/neural_net.NUMBERLIMIT]]))[0][0]*neural_net.NUMBERLIMIT)
            print("===================")

    exit()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main2()

    log = False
    sizes = [2700, 5, 2] #[2700,500,100, 10, 2]
    net = Network(sizes)
    try:
        net.load()
    except:
        net.save()
        pass
    #net = Network([3,5, 1])

    training_data = neural_net.loadTrainingData()
    #training_data = [(np.array([[0, 0, 1]]).T, np.array([[0]])), (np.array([[0, 1, 1]]).T, np.array([[1]])), (np.array([[1, 0, 1]]).T, np.array([[1]])), (np.array([[0, 1, 0]]).T, np.array([[1]])), (np.array([[1, 0, 0]]).T, np.array([[1]])), (np.array([[1, 1, 1]]).T,np.array([[0]])), (np.array([[0, 0, 0]]).T, np.array([[0]]))]

    if log:
        print(training_data)

        print("weights")
        print(net.weights)

        print("biases")
        print((net.biases))

    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.train(training_data, 300000, 20, 20.0, 50)
    #print(net.feedforward(np.array([[1,0,0]]).T))
    #testpath = neural_net.savePath + "training\\data\\banane\\"
    #print(net.feedforward(neural_net.getArrayData(testpath+"22.jpg", np.zeros((5,5)))[0]))
    net.doTest()
