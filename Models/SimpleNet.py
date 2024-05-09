import numpy as np
import scipy
class neuralNetworkSingleHidden:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.residuals=[]
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    def train(self, inputs_list, targets_list, epochs=10):
        for epoch in range(epochs):
            sum_errors = 0
            for inputs, targets in zip(inputs_list, targets_list):
                inputs = np.array(inputs, ndmin=2).T
                targets = np.array(targets, ndmin=2).T

                hidden_1_inputs = np.dot(self.wih, inputs)
                hidden_1_outputs = self.activation_function(hidden_1_inputs)

                final_inputs = np.dot(self.who, hidden_1_outputs)
                final_outputs = self.activation_function(final_inputs)

                output_errors = targets - final_outputs
                sum_errors += np.sum(output_errors)  #accumulate sum of errors

                hidden_1_errors = np.dot(self.who.T, output_errors) 

                self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_1_outputs))
                self.wih += self.lr * np.dot((hidden_1_errors * hidden_1_outputs * (1.0 - hidden_1_outputs)), np.transpose(inputs))

            self.residuals.append(sum_errors) 
        pass

    
    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_1_inputs = np.dot(self.wih, inputs)
        hidden_1_outputs = self.activation_function(hidden_1_inputs)
        
        
        final_inputs = np.dot(self.who, hidden_1_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs