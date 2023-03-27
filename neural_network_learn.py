import numpy as np
import scipy.special
class neuralNetwork():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_func=lambda x:scipy.special.expit(x)

    def train(self,inputs_list,targets_list):
        inputs=np.array(inputs_list,ndim=2).T
        targets=np.array(targets_list,ndim=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors)
        self.who+=self.lr*np.dot(output_errors*final_outputs*(1.0-final_outputs),np.transpose(hidden_outputs))
        self.wih+=self.lr*np.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),np.transpose(inputs))

    def query(self,inputs_list):
        inputs=np.array(inputs_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_func(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_func(final_inputs)
        return final_outputs

input_nodes=3
output_nodes=3
hidden_nodes=3
learning_rate=0.5
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
n.query([1.0,0.5,-1.5])

