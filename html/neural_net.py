import numpy as np
import os

def getRandomWeights(anzahlInputsProNeuron, anzahlNeuronen):
    return 2 * np.random.random((anzahlInputsProNeuron, anzahlNeuronen)) - 1

savePath = "C:/Users/JulianEbeling/Documents/xampp/xampp/htdocs/Objekt-KI/html/"#AI Test Data\\"
#savePath = "saveTest\\"

training_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
#training_inputs = 0
#training_outputs = 0
log = False
log2 = False

NUMBERLIMIT = 10000

class Layer:

    def init(self, previous, next, first=False, last=False, inputs=0, outputs=0):
        self.previous = previous
        self.next = next
        self.weights = getRandomWeights(self.anzahlInputsProNeuron, self.anzahlNeuronen)
        self.biases = np.random.rand(self.anzahlNeuronen, 1)
        self.adjustments = 0
        self.error = 0
        self.delta = 0
        self.isLast = last
        self.isFirst = first

        self.input = inputs
        self.output = 0
        self.optimal_output = outputs

    def __init__(self, anzahlNeuronen, anzahlInputsProNeuron):
        self.anzahlNeuronen = anzahlNeuronen
        self.anzahlInputsProNeuron = anzahlInputsProNeuron

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def sigmoid_derivitave(self, x):
        return x * (1 - x)

    def calculateOutputs(self):

        if not self.isFirst: #beim ersten gibt es ja kein previous
            self.input = self.previous.output
        if log:
            print(f"calculating outputs of {self.ID}")
            print(f"inputs: {self.input}")
            print(f"weights: {self.weights}")
        self.output = self.sigmoid(np.dot(self.input, self.weights))######################### hier dann die biase
        #weightedSum = np.dot(self.input, self.weights)
        if log:
            print(f"output: {self.output}")

        #self.biases.resize(weightedSum.shape)
        #print(self.biases)
        #self.output = self.sigmoid(weightedSum+self.biases)


    def calculateErrorAndDelta(self):
        #error
        if self.isLast:
            self.error = self.optimal_output - self.output
        else:
            self.error = self.next.delta.dot(self.next.weights.T)

        #delta
        self.delta = self.error * self.sigmoid_derivitave(self.output)
        if log2:
            print(f"error: {self.error}")
            print(f"delta: {self.delta}")

    def calculateAdjustments(self, autoApply=True):

        if self.isFirst:
            self.adjustments = training_inputs.T.dot(self.delta)
            #print("inputs.T", training_inputs.T)
        else:
            self.adjustments = self.previous.output.T.dot(self.delta)

        if autoApply:
            self.applyAdjustments()

        if log2:
            print(f"adjustments: {self.adjustments}")

    def applyAdjustments(self):
        self.weights += self.adjustments

    def save(self):
        file = open(savePath+self.ID, "w")
        weigthsString = "np.array(["
        for w1 in self.weights:
            weigthsString += "["
            for w2 in w1:
                weigthsString += f"{w2},"
            weigthsString += "],"

        weigthsString += "])"
        weigthsString = weigthsString.replace(",]", "]")

        filecontent = f"weights={weigthsString}\n\nbiases=[]"
        file.write(filecontent)
        file.flush()
        file.close()
        print(f"Saved weigths in {self.ID}")


    def laod(self):
        file = open(savePath + self.ID, "r")
        filecontent = file.read()
        self.weights = eval(filecontent.split("\n\n")[0].replace("weights=", ""))
        print(f"Loaded weigths from {self.ID}")
        file.close()


class NeuralNetwork:

    def __init__(self, training_inputs, training_outputs):
        self.layers = []
        self.number_of_layers = 0
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs

    def train(self, iterations):
        self.layers[0].input = self.training_inputs
        status = 0
        count = 0
        print(f"Started training with {iterations} iterations.")
        for i in range(iterations):

            for layer in self.layers: #calculate Outputs
                layer.calculateOutputs()

            for x in range(1, self.number_of_layers+1): #calculate error and delta
                self.layers[-x].calculateErrorAndDelta()

            for layer in self.layers: #calculate adjustments
                layer.calculateAdjustments()

            if (i / iterations * 100) - status > 1.0:
                status = int( i / iterations * 100)
                print(f"Status: {status}%")

            if count == 5000:
                count = 0
                self.saveAll()

            count += 1

        print("finished training")

    def think(self, input):
        self.layers[0].input = input
        for layer in self.layers: #calculate Outputs
            layer.calculateOutputs()
        return self.layers[-1].output #von der letzten layer

    def add_layer(self, anzahlNeuronen):
        self.number_of_layers += 1

        if self.number_of_layers == 1: #is first created layer
            anzahlInputsProNeuron = self.training_inputs.shape[1]
        else:
            anzahlInputsProNeuron = self.layers[self.number_of_layers-2].anzahlNeuronen

        layer = Layer(anzahlNeuronen, anzahlInputsProNeuron)
        layer.ID = f"layer{self.number_of_layers}.txt"

        self.layers.append(layer)

    def init(self):
        count = 0

        for layer in self.layers:
            count += 1

            if count == 1:
                print("erste initialisiert")
                layer.init(None, self.layers[count], first=True, inputs=self.training_inputs)

            elif count == self.number_of_layers:
                print("letzte initialisiert")
                layer.init(self.layers[count-2], None, last=True, outputs=self.training_outputs)

            else:
                layer.init(self.layers[count-2], self.layers[count])

    def saveAll(self):
        print("saving all layers")

        for layer in self.layers:
            layer.save()
        print("saved all layers")


    def loadAll(self):
        for layer in self.layers:
            layer.laod()
        print("loaded all layers")

def loadByteArray(path):
    inputs = []
    with open(path, "rb") as f:
        while (byte := f.read(1)):
            b = int.from_bytes(byte, "big")/255
            inputs.append(b)
    return inputs

def loadByteArray2(path):
    return [loadByteArray(path)]

def trainingFunction(x, formel):

    #y = 2*x
    y = eval(formel)
    return y

def loadTrainingDataNumbers(formel):
    training_data = []

    ins = range(NUMBERLIMIT)#range(int(100))
    outs = []
    for x in ins:
        outs.append(trainingFunction(x, formel))

    for x,y in zip(ins, outs):
        training_data.append((np.array([[x/NUMBERLIMIT]]), np.array([[y/NUMBERLIMIT]])))

    return training_data

def loadTrainingArray():
    global training_inputs, training_outputs

    training_inputs = []
    training_outputs = []
    path = savePath+"training\\data\\apfel\\"
    for p in os.listdir(path):
        if "." in p:
            #is file
            training_inputs.append(loadByteArray(path+p))
            training_outputs.append([1, 0])
            print(f"loaded data apfel: {p}")

    path = savePath + "training\\data\\banane\\"
    for p in os.listdir(path):
        if "." in p:
            # is file
            training_inputs.append(loadByteArray(path + p))
            training_outputs.append([0, 1])
            print(f"loaded data banane: {p}")

    training_inputs = np.array(training_inputs)
    training_outputs = np.array(training_outputs)
    print(f"Training_inputs: {training_inputs}")
    print(f"Training_outputs: {training_outputs}")

    return training_inputs, training_outputs

def getArrayData(path, output):
    return (np.array(loadByteArray2(path)).T, output.T)

def loadTrainingData():
    training_data = []

    path = savePath + "training\\data\\apfel\\"
    for p in os.listdir(path):
        if "." in p:
            # is file
            training_data.append(getArrayData(path+p, np.array([[1, 0]])))
            print(f"loaded data apfel: {p}")

    path = savePath + "training\\data\\banane\\"
    for p in os.listdir(path):
        if "." in p:
            # is file
            training_data.append(getArrayData(path+p, np.array([[0, 1]])))
            print(f"loaded data banane: {p}")

    return training_data


def main():
    network = NeuralNetwork(training_inputs, training_outputs)
    network.add_layer(3)
    network.add_layer(1)
    network.init()
    try:
        network.loadAll()
    except:
        network.saveAll()
    print(f"Training_inputs: {training_inputs}")
    print(f"Training_outputs: {training_outputs}")
    network.train(1)
    network.saveAll()

    while True:
        A = float(input("Input 1 > "))
        B = float(input("Input 2 > "))
        C = float(input("Input 3 > "))
        data = np.array([A,B,C])
        print(f"Input Data = {data}")
        print("Output Data: ")
        print(network.think(data))

if __name__ == '__main__':
    main()
    #print(loadByteArray("saveTest\\training\\data\\apfel\\1.jpg"))

#training_inputs, training_outputs = loadTrainingArray()
