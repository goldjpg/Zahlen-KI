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
            test_data=None, formel=""): #minibatch stochastic gradient descent

        if test_data: n_test = len(test_data)
        n = len(training_data)

        #n_test = sum(1 for _ in test_data)
        #n = sum(1 for _ in training_data)

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
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                #print("Epoch {0} complete".format(j))
                pass

            if count == savingRate:
                count = 0
                self.save()

            count += 1

            if (i / iterations * 100) - status >= 0.1:
                status = round( i / iterations * 100, 2)
                if logTraining:
                    print(f"Status: {status}% at ", getTimestamp())
                #self.doTest()
                if productionCode:
                    print(status, flush=True)

        self.save(formel)

    def update_mini_batch(self, mini_batch, lr):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        cost = 0
        usematrices = False
        if usematrices:
            xmat = np.hstack([x] for x,y in mini_batch)
            ymat = np.hstack([y] for x,y in mini_batch)
            delta_gradient_b, delta_gradient_w, c = self.backpropagation(xmat, ymat)
        else:
            for x, y in mini_batch:
                #print("shape:")
                #print(np.shape(x))
                #print(np.shape(y))

                delta_gradient_b, delta_gradient_w, c = self.backpropagation(x, y)
                gradient_b = [b+delta_b for b, delta_b in zip(gradient_b, delta_gradient_b)] #für mini batch updaten
                gradient_w = [w+delta_w for w, delta_w in zip(gradient_w, delta_gradient_w)]
                cost += c

            lr2 = lr / len(mini_batch)#lr * len(mini_batch)# #len(mini_batch) / lr wäre logischer xD #official
            #lr2 = lr * len(mini_batch)# #len(mini_batch) / lr wäre logischer xD #own
            self.weights = [w-lr2*nw #gesamt update
                            for w, nw in zip(self.weights, gradient_w)]
            self.biases = [b-lr2*nb
                           for b, nb in zip(self.biases, gradient_b)]

        if False:
            cost = cost / (2*len(mini_batch))
            print("Cost: ", cost)

    def backpropagation(self, x, y): #x: input, y: perfect output
        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        gradient_weight = [np.zeros(w.shape) for w in self.weights]

        # Berechnung der Ausgabewerte und Speicherug aller Aktivierungen sowie z Werte
        activation = x
        activations = [x] #alle Aktivierungen, Schicht für Schicht
        zs = [] #  alle z Vektoren, Schicht für Schicht

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)

        #Cost errechnen

        cost = (np.linalg.norm(activations[-1] - y)) ** 2

        # Backpropagation
        # Fehlerwert letzte Schicht
        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1]) # Anwendung des Hadamard Produkts

        # Ableitung nach Bias Units und Gewichtungen
        gradient_bias[-1] = delta
        gradient_weight[-1] = np.dot(delta, activations[-2].T)

        # rückwärts die Werte für alle Schichten berechnen
        for l in range(2, self.num_layers):
            z = zs[-l]

            # Fehlerwert für alle weiteren Schichten
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_derivative(z) # Multiiplikation mit dem Fehlerwert der Schicht danach und anschließend Anwendung des Hadamard Produkts für die Multiplikation mit der Ableitung der Sigmoidfunktion

            # Ableitung nach Bias Units und Gewichtungen
            gradient_bias[-l] = delta
            gradient_weight[-l] = np.dot(delta, activations[-l-1].T)

        return (gradient_bias, gradient_weight, cost)

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, formel=""):
        if productionCode:
            id = str(datetime.now().timestamp())
            newPath = neural_net.savePath + id
            os.mkdir(newPath)
            np.save(newPath+"/weights.npy", self.weights, allow_pickle=True)
            np.save(newPath+"/biases.npy", self.biases, allow_pickle=True)
            if formel != "":
                with open(newPath+"/formel.txt", "w") as f:
                    f.write(formel)
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
            self.weights = np.load(neural_net.savePath +str(id)+ "/weights.npy", allow_pickle=True)
            self.biases = np.load(neural_net.savePath +str(id)+ "/biases.npy", allow_pickle=True)
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

    def doTest(self, image=False):
        testpath = neural_net.savePath + "training/mnistdata/apfel/"

        p = testpath + "1.jpg"
        if logTraining:
            if image:
                p = neural_net.savePath + "out2/w1.png"
                print(p)
                print("Testing : ", p)
                print(self.feedforward(neural_net.getArrayData(p, np.zeros((5, 5)))[0]))
            else:
                print("Testing : ", p)
                print(self.feedforward(neural_net.getArrayData(p, np.zeros((5, 5)))[0]))


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def main2():
    global logSaving, logTraining, productionCode

    run = False
    train = False
    log = False


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


    sizes = [1,8, 5, 5, 5, 1]  # sizes = [1, 8, 5, 1]
    # sizes = [3, 5,3, 1]
    net = Network(sizes)
    if not productionCode:
        try:
            net.load()
        except:
            net.save()
    #print(net.weights[1])
    #print(net.biases[0])
    training_data = neural_net.loadTrainingDataNumbers(formel)
    #print(training_data)
    #print(training_data)
    #training_data = [(np.array([[0, 0, 1]]).T, np.array([[0]])), (np.array([[0, 1, 1]]).T, np.array([[1]])), (np.array([[1, 0, 1]]).T, np.array([[1]])), (np.array([[0, 1, 0]]).T, np.array([[1]])), (np.array([[1, 0, 0]]).T, np.array([[1]])), (np.array([[1, 1, 1]]).T,np.array([[0]])), (np.array([[0, 0, 0]]).T, np.array([[0]]))]

    if log:
        print(training_data)

        print("weights")
        print(net.weights)

        print("biases")
        print((net.biases))

    if train:
        ITERATIONS = 100
        net.train(training_data, ITERATIONS, 20, 2, ITERATIONS, formel=formel)
    elif run:
        id = args[2]
        x = float(args[3])
        net.load(id=id)
        p = neural_net.savePath + f"/{id}"
        formel = ""
        with open(p + "/formel.txt", "r") as f:
            formel = f.read()
        #print(formel)
        #print(neural_net.trainingFunction(x, formel), formel)
        print(str(net.feedforward(np.array([[x / neural_net.NUMBERLIMIT]]))[0][0] * neural_net.NUMBERLIMIT)+" Richtig: "+str(neural_net.trainingFunction(x,formel)))

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

def main3():
    import mnist_loader
    trainingdata, validationdata, testdata = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    trainingdata = (trainingdata)

    trd = []
    for t in trainingdata:
        trd.append(t)

    td = []
    for t in testdata:
        td.append(t)
    print(testdata)
    print("training..")
    net.train(trd, 30,10,3.0,30,test_data=td)
    exit()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main2()
    log = False
    sizes = [3,5,2]#[2700, 5, 2] #[2700,500,100, 10, 2]
    net = Network(sizes)
    try:
        net.load()
    except:
        net.save()
        pass
    #net = Network([3,5, 1])
    training_data = neural_net.loadTrainingData()
    print(training_data)
    #training_data = [(np.array([[0, 0, 1]]).T, np.array([[0]])), (np.array([[0, 1, 1]]).T, np.array([[1]])), (np.array([[1, 0, 1]]).T, np.array([[1]])), (np.array([[0, 1, 0]]).T, np.array([[1]])), (np.array([[1, 0, 0]]).T, np.array([[1]])), (np.array([[1, 1, 1]]).T,np.array([[0]])), (np.array([[0, 0, 0]]).T, np.array([[0]]))]

    if log:
        print(training_data)

        print("weights")
        print(net.weights)

        print("biases")
        print((net.biases))

    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.train(training_data, 10000, 20, 100, 10000)
    #print("finished")
    #net.train(training_data, 300000, 20, 20.0, 5000)
    #print(net.feedforward(np.array([[1,0,0]]).T))
    #testpath = neural_net.savePath + "training/mnistdata/banane/"
    #print(net.feedforward(neural_net.getArrayData(testpath+"22.jpg", np.zeros((5,5)))[0]))

    print("Finished Training")
    net.doTest(True)
