import numpy as np
from abc import ABC, abstractmethod

# Network objects represent feed forward neural networks. These objects are initialize using a list of the 
# amount of neurons in each layer. For example, Network([1, 2, 3, 4]) will initialize a neural network with 4 layers
# an input layer of 1 neuron, 2 hidden layers with 2 and 3 neurons respectively, and an output layer with 4 neurons.
# The main data a Network object will store is a list of "Layer" objects. The network class will then provide multiple
# functions for feed forward, learning, and other helpful abstractions a neural network would need.
class Network:
    # initialize a list of Layer objects based on the list passed to the constructor
    def __init__(self, network_init):
        self.layer_list = self.init_new_network(network_init) # initialize a list of Layer objects
        
        # having initialized all layers, use the dimensions to initialize weight arrays for each layer besides the input layer
        for layer in self.layer_list: 
            layer.initialize_weights()

    # This function is called only in the constructor of a network and serves the role of initializing Layer objects for the layer list.
    # One important thing to note which will be described a bit more later, is that the layers are also a doubly-linked list so they have
    # information about their neighboring layers
    def init_new_network(self, network_init):
        return_list = [None] * len(network_init)

        # iterate over empty return list and populate it with layers. the 0th and final index will get an InputLayer and OutputLayer respectively. The
        # importance of different layer classes will be seen later in the code.
        for i in range(len(network_init)):
            if i == 0:
                return_list[i] = InputLayer(network_init[i])
            elif i != len(network_init) - 1:
                return_list[i] = HiddenLayer(network_init[i])
                return_list[i].set_prev(return_list[i-1])
                return_list[i-1].set_next(return_list[i])
            else:
                return_list[i] = OutputLayer(network_init[i])
                return_list[i].set_prev(return_list[i-1])
                return_list[i-1].set_next(return_list[i])

        return return_list
    
    """
    The rest of the functions in the Network class will serve as the functional portions of networks and leraning
    """

    # feedforward will start at the head of the layers list by taking the 0th element of the list, then it will recursively
    # feed forward the provided data with functions inside of the Layer objects. print_result is an optional parameter that
    # prints the output layers activation if definded
    def feedforward(self, data, print_result = None):
        data = np.array(data)
        
        # call forward after setting input layers activation to be data
        self.layer_list[0].set_activation(data)
        self.layer_list[0].forward()

        # print output layer activation if print_result is defined
        if print_result != None:
            print(self.layer_list[len(self.layer_list) - 1].activation_vector)

    # this function will perform stochastic_gradient_descent on the network. The steps it will follow to do so are:
    # for each epoch, sample a random x, y input, expected output pair and feed it forward into the network, then use
    # the backprop_error function defined in the layers to find the error of the output layer, then recursively find
    # the error of previous layers. Finally the errors will be used to populate a gradient array to be added to the 
    # weight arrays of the layers, and a bias vector for each layer.
    def stochastic_gradient_descent(self, training_data_tuple, epochs, learning_rate, batch_size):
        for i in range(epochs):
            # sample from training data tuple a random minibatch
            rng = np.random.default_rng()
            random_integers = rng.choice(np.arange(0, len(training_data_tuple)), size=batch_size, replace=False)

            # reset networks bias and weight gradient vectors
            self.layer_list[len(self.layer_list) - 1].set_bias_gradient()
            self.layer_list[len(self.layer_list) - 1].set_weight_gradient()

            for j in range(batch_size):
                sample = training_data_tuple[random_integers[j]]

                # feed forward sample from random minibatch
                self.feedforward(sample[0])

                # use expected output to find the error of each neuron in each layer recursively
                self.layer_list[len(self.layer_list) - 1].backprop_error(sample[1])

                # having calculated all we need, apply to bias gradient the current samples prefered bias gradient,
                # does not change bias until all mini batch apply bias gradients
                self.layer_list[len(self.layer_list) - 1].change_bias_gradient()

                # apply to weight gradient the current samples prefered weght gradient, does not change wegihts
                # until all mini batches apply gradient descent
                self.layer_list[len(self.layer_list) - 1].change_weight_gradient()

                
            # apply the bias gradient vector to each layer
            self.layer_list[len(self.layer_list) - 1].apply_bias_gradient(learning_rate)
            # apply the weight gradient matrix to each layer
            self.layer_list[len(self.layer_list) - 1].apply_weight_gradient(learning_rate)



     
    # simple testing function that returns amount of correct identifications in a provided data tuple
    def evaluate(self, data):
        correct = 0
        for x, y in data:
            self.feedforward(x)
            prediction = np.argmax(self.layer_list[-1].activation_vector)
            if prediction == y:
                correct += 1
        print(correct)


            
# Layer is the abstract class which will define some features a layer will have, children Layers will simply be initialized
# with a number representing the neuron count.
class Layer(ABC):
    def __init__(self, neuron_count):
        self.neuron_count = neuron_count

        # initialize bias vector is done in the constructor, because it simply will have the dimensions
        # of the neuron_count. the weight arrays will be initialized after all layers are made, because
        # the dimensions need to be known of the previous layer.
        self.bias_vector = self.initialize_biases()

    # initialize weights populates arrays of the dimensions: current layer neuron count rows by previous layer
    # neuron count columns. The values of the entries will initialy be randomly distributed about 0 until learning
    # takes place.
    def initialize_weights(self):
        weight_list = np.zeros((self.neuron_count, self.prev_layer.neuron_count))
        for x in range(self.neuron_count):
            for y in range(self.prev_layer.neuron_count):
                weight_list[x][y] = np.random.default_rng().standard_normal()

        self.weight_array = np.array(weight_list)

    # initialize biases will create a vector that has neuron_count entries which are randomly distributed about 0 until
    # learning takes place.
    def initialize_biases(self):
        return_vector = []
        for i in range(self.neuron_count):
            return_vector.append(np.random.default_rng().standard_normal())

        return np.array(return_vector)

    def set_next(self, next_layer):
        self.next_layer = next_layer

    def set_prev(self, prev_layer):
        self.prev_layer = prev_layer

    # abstract pass function that is implemented in children Layer objects
    @abstractmethod
    def forward(self):
        pass

    # abstract pass function that is implemented in children Layer objects
    @abstractmethod
    def backprop_error(self):
       pass

    # reset bias gradient to zero vector
    def set_bias_gradient(self):
        self.bias_gradient = np.zeros_like(self.bias_vector)
        self.prev_layer.set_bias_gradient()

    # reset weight gradient to zero array 
    def set_weight_gradient(self):
        self.weight_gradient = np.zeros_like(self.weight_array)
        self.prev_layer.set_weight_gradient()

    # this will be called for each mini batch in a sample, and will adjust a gradient without actually applying
    # it to the networks bias vector
    def change_bias_gradient(self):
        self.bias_gradient += self.error_vector
        self.prev_layer.change_bias_gradient()

    # this will finnaly apply the bias gradient to the layers bias vectors recursively
    def apply_bias_gradient(self, learning_rate):
        self.bias_vector -= learning_rate*self.bias_gradient
        self.prev_layer.apply_bias_gradient(learning_rate)

    # error of current layer * transpose activation of previous layer = new weight gradient
    # this will be called for each mini batch in a sample, and will adjust a gradient without actually applying
    # it to the networks weight matrix
    def change_weight_gradient(self):
        self.weight_gradient += np.reshape(self.error_vector, (len(self.error_vector), 1)) @ np.reshape(self.prev_layer.activation_vector, (1, len(self.prev_layer.activation_vector))) 
        self.prev_layer.change_weight_gradient()

    # this will finnaly apply the weight gradient to the layers weight arrays recursively
    def apply_weight_gradient(self, learning_rate):
        self.weight_array -= learning_rate*self.weight_gradient
        self.prev_layer.apply_weight_gradient(learning_rate)


# input layer is pretty simple, it will construct an object with neuron_count neurons, then it will pass
# the weight and bias initialization because the input layer does not need weights nor biases.
# the activation of the input layer neurons will be set by feed forward
class InputLayer(Layer):
    # consturct input layer
    def __init__(self, neuron_count):
        super().__init__(neuron_count)
        self.prev_layer = None

    # pass weight intilization
    def initialize_weights(self):
        pass

    # pass bias initialization
    def initialize_biases(self):
        pass

    # setter for activation
    def set_activation(self, activation):
        self.activation_vector = activation

    # start the recursive feed forward process by calling forward on the .next_layer
    def forward(self):
        self.next_layer.forward()

    '''
    a lot of functions in the input layer will simply be the base case to recursive calls through
    the network, and as a result just pass. This is seen below. Most are overiding something in the
    parent Layer object
    '''

    def backprop_error(self):
        pass

    def set_bias_gradient(self):
        pass

    def set_weight_gradient(self):
        pass

    def change_bias_gradient(self):
        pass

    def apply_bias_gradient(self, learning_rate):
        pass

    def change_weight_gradient(self):
        pass

    def apply_weight_gradient(self, learning_rate):
        pass

# hidden layers will be performing some linear algebra for feeding forward and backpropagating. They are
# simply constructed by a neuron count, and perform a couple of functions to help the network learn.
class HiddenLayer(Layer):
    def __init__(self, neuron_count):
        super().__init__(neuron_count)

    # forward will calculate the weight array multiplied by the previous layers activation and add this layers
    # biases then it will call the sigmoid activation, and finally continue the recursive forward calls.
    def forward(self):
        self.z_vector = self.weight_array @ self.prev_layer.activation_vector + self.bias_vector
        self.activation_vector = sigmoid(self.z_vector)
        self.next_layer.forward()

    # find the error of hidden layer using formula: weights of next layer transposed times errors of next layer * sigmoid prime of z.
    # where * is inner product
    def backprop_error(self):
        self.error_vector = (np.transpose(self.next_layer.weight_array) @ self.next_layer.error_vector) * sigmoid_prime(self.z_vector)
        # recursively backprop error
        self.prev_layer.backprop_error()

# output layers in feed forward are just like a hidden layer in the sense they will perform linear algebra.
# they also will play a role in backpropagation that is different to hidden layers.
class OutputLayer(Layer):
    def __init__(self, neuron_count):
        super().__init__(neuron_count)
        self.next_layer = None

    # final call of forward but same math of weight times previous activation + bias, then sigmoid activation.
    def forward(self):
        self.z_vector = self.weight_array @ self.prev_layer.activation_vector + self.bias_vector
        self.activation_vector = sigmoid(self.z_vector)
        
    # get layers quadratic error prime given expected value
    def quad_error_prime(self, expected_value):
        expected_vector = np.zeros_like(self.activation_vector)
        expected_vector[expected_value] = 1
        return self.activation_vector - expected_vector

    # find the error of the output layer using formula: quadratic error prime of activations * sigmoid prime of z vector, where * is hadamard product
    def backprop_error(self, expected_value):
        self.error_vector = self.quad_error_prime(expected_value) * sigmoid_prime(self.z_vector)
        # recursively backprop error
        self.prev_layer.backprop_error()

# sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# derrivitive of sigmoid activation function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

'''
End of Module
'''
