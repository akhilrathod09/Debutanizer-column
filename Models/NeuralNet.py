import numpy as np
import scipy
class neuralNetworkMultiHidden:
    
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.residuals=[]
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.whh = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.who = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        # learning rate
        self.lr = learningrate
        
        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    def train(self, inputs_list, targets_list, epochs=10):
        for epoch in range(epochs):
            sum_errors = 0
            for inputs, targets in zip(inputs_list, targets_list):
                inputs = np.array(inputs_list, ndmin=2).T
                targets = np.array(targets_list).T
                
                hidden_1_inputs = np.dot(self.wih, inputs)
                hidden_1_outputs = self.activation_function(hidden_1_inputs)
                
                hidden_2_inputs = np.dot(self.whh, hidden_1_outputs)
                hidden_2_outputs = self.activation_function(hidden_2_inputs)
                
                final_inputs = np.dot(self.who, hidden_2_outputs)
                final_outputs = self.activation_function(final_inputs)
                
                output_errors = targets - final_outputs
                sum_errors += np.sum(output_errors)
                # print(np.shape(output_errors))
                # self.residuals=output_errors.T
                hidden_2_errors = np.dot(self.who.T, output_errors) 
                hidden_1_errors = np.dot(self.whh.T, hidden_2_errors) 
                
                self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_2_outputs))
                
                self.whh += self.lr * np.dot((hidden_2_errors * hidden_2_outputs * (1.0 - hidden_2_outputs)), np.transpose(hidden_1_outputs))
                
                self.wih += self.lr * np.dot((hidden_1_errors * hidden_1_outputs * (1.0 - hidden_1_outputs)), np.transpose(inputs))
            # print(sum_errors)
            self.residuals.append(sum_errors)
        pass


    
    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_1_inputs = np.dot(self.wih, inputs)
        hidden_1_outputs = self.activation_function(hidden_1_inputs)
        
        hidden_2_inputs = np.dot(self.whh, hidden_1_outputs)
        hidden_2_outputs = self.activation_function(hidden_2_inputs)
        
        final_inputs = np.dot(self.who, hidden_2_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs