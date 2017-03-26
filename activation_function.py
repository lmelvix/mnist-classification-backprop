import read_mnist
import numpy
import math


class ActivationFunction(object):

    def __init__(self, feature_size, sample_size, layer1_size, layer2_size):
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.feature_size = feature_size
        self.sample_size = sample_size
        self.prev_gradient_1 = numpy.asmatrix(numpy.zeros((self.feature_size,self.layer1_size)))
        self.prev_gradient_2 = numpy.asmatrix(numpy.zeros((self.layer1_size, self.layer2_size)))

    def initialize_weights(self):
        # Xavier Initialization
        weight_1 = numpy.random.randn(self.feature_size,self.layer1_size)
        weight_2 = numpy.random.randn(self.layer1_size, self.layer2_size)
        return ((1/math.sqrt(self.feature_size)) * numpy.asmatrix(weight_1)), \
               ((1/math.sqrt(self.layer1_size)) * numpy.asmatrix(weight_2))

    def sigmoid(self, input):
        # Apply sigmoid 1/(1+exp(-x)) on given input
        return 1.0/(1.0 + numpy.exp(-input))

    def tanh(self, weight, inp):
        # Apply 1.7159 * (tanh(2/3 * x))
        dot_product = numpy.dot(inp, weight)
        tanh_output = 1.7159 * (numpy.tanh((2./3.) * dot_product))
        return tanh_output

    def tanh_gradient(self, weight, inp):
        # Gradient : 1.1439 - (0.5828*z^2)
        tanh_output = self.tanh(weight, inp)
        return 1.1439 - (0.5828 * numpy.multiply(tanh_output, tanh_output))


    def logistic_regression(self, layer1_input, weight_1):
        # Squash w.x to 1/(1+exp(-w.x)) for Logistic layer output
        dot_product = numpy.dot(layer1_input, weight_1)
        return self.sigmoid(dot_product)

    def softmax_regression(self, layer2_input, weight_2):
        dot_product = numpy.exp(numpy.dot(layer2_input, weight_2))
        tot_exp = dot_product.sum(axis=1)
        return dot_product / tot_exp

    def logistic_gradient(self, weight_1, layer1_input):
        logit_output = self.logistic_regression(layer1_input, weight_1)
        return numpy.multiply(logit_output,(1 - logit_output))

    def gradient_update(self, gradient, weight, learning_rate, input, regularize, momentum):
        # Compute gradient for the current iteration
        curr_gradient = numpy.dot(gradient.T, input).T

        if weight.shape == self.prev_gradient_1.shape:
            prev_gradient = self.prev_gradient_1
        else:
            prev_gradient = self.prev_gradient_2

        # Update weight vector
        gradient_to_update = (learning_rate*(curr_gradient+(2*regularize*weight)))
        weight += gradient_to_update + (momentum*prev_gradient)

        # Update previous gradient
        if weight.shape == self.prev_gradient_1.shape:
            self.prev_gradient_1 = gradient_to_update
        else:
            self.prev_gradient_2 = gradient_to_update
        return weight

    def threshold_output(self, my_output):
        threshold_output = my_output.copy()
        threshold_output[numpy.where(threshold_output==numpy.max(threshold_output,axis=1))] = 1
        threshold_output[numpy.where(threshold_output!=numpy.max(threshold_output,axis=1))] = 0
        return threshold_output



