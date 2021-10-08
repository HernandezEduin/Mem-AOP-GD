# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 20:32:43 2021

@author: Eduin Hernandez
"""
#------------------------------------------------------------------------------
'Libraries'
#Time
import time
from datetime import datetime

#Storage
import argparse
import pickle

#Math
import numpy as np

#DNN
import torch
import torch.nn as nn
from utils.custom_nn import LinearMem
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#Preprocessing
from utils.energy_preprocessing import prepare_data, split, normalize

#Plotting
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
'Argparser'
def str2bool(string):
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Temperature Training')

    'Model Details'
    parser.add_argument('--batch-size', type=int, default=144, help='Batch Size for Training')
    parser.add_argument('--epoch-num', type=int, default=100, help='Total Epochs used for Training')
    
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for the model')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    parser.add_argument('--nesterov', type=str2bool, default='False', help='Whether to use Nesterov Momentum')
       
    'Device'
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Either cpu or cuda for gpu')

    'Compression'
    parser.add_argument('--compression', type=int, default=32, help='Number of partitions to keep.')
    parser.add_argument('--compression-type', type=str, default='randk', help='Erasure Type')
    parser.add_argument('--memory-decay-rate', type=float, default=1.0, help='Memory Decay Rate')
    
    'Simulations'
    parser.add_argument('--simulation-num', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--memory-state', type=list, default=True, help='Either True, False, or None')
    
    'Print and Plot'
    parser.add_argument('--plot-state', type=str2bool, default='True', help='Whether to plot the results')
    parser.add_argument('--verbose-state', type=str2bool, default='True', help='Whether to print the results')
    
    'Save Details'
    parser.add_argument('--save-state', type=str2bool, default='False', help='Whether to save the results of the training')
    parser.add_argument('--folder-path', type=str, default='./results/energy/', help='Folder Save file path for Accuracy')
    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------
args = parse_args()

assert 576 % args.batch_size == 0, 'Batch size must be a multiple of 576'

if args.compression_type not in ['topk', 'randk', 'weightedk']:
    assert False
#-----------------------------------------------------------------------------
if args.device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

print('Device:', device.type)
#----------------------------------------------------------------
'Model Declaraction'
class Net(nn.Module):
    def __init__(self, mem_state: bool):
        super(Net, self).__init__()        
        self.dense = LinearMem(16,1, memory_state = mem_state, bias=True)
    
    def forward(self, x):        
        y = self.dense(x)
        return y
#------------------------------------------------------------------------------
'Modify Gradients of a Layer'
def register_hook(model):
    model.dense.register_backward_hook(memorygrad)

def calculate_matrix_norm(A, B):
    norms = torch.square(A.norm(dim=1, p='fro')*B.norm(dim=1, p='fro')) #Should be the same as squaring once from the outside
    norms = norms/norms.sum()
    return norms, norms.argsort(descending=True)

def memorygrad(self, grad_input, grad_output):
    if self.__class__ == class_dense:                
        if self.memory_state == True:
            """Manually Recalculating the Gradients with Memory"""
            A = self.memA + self.input*np.sqrt(args.learning_rate)
            B = self.memB + grad_output[0]*np.sqrt(args.learning_rate)
            
            grad_b = torch.einsum('ij -> j', grad_output[0])*args.learning_rate
            grad_i = None
            
            if args.compression_type == 'topk':
                _, ind = calculate_matrix_norm(A, B)
                ind = ind[:args.compression]
            elif args.compression_type == 'weightedk':
                prob, _ = calculate_matrix_norm(A, B)
                ind = np.random.choice(args.batch_size, args.compression, replace=False, p=prob.cpu().numpy())
            elif args.compression_type == 'randk':
                ind = np.random.choice(args.batch_size, args.compression, replace=False)
            
            grad_w = torch.einsum('ji, jk -> ik', A[ind], B[ind])
            A[ind] = 0      #Remove what was transmitted
            B[ind] = 0      #Remove what was transmitted

            
            self.memA = torch.clone(A)*np.sqrt(args.memory_decay_rate)
            self.memB = torch.clone(B)*np.sqrt(args.memory_decay_rate)
            
            grad_input = (grad_b, grad_i, grad_w)
        elif self.memory_state == False:
            """Manually Recalculating the Gradients without Memory"""
            A = self.input*np.sqrt(args.learning_rate)
            B = grad_output[0]*np.sqrt(args.learning_rate)

            grad_b = torch.einsum('ij -> j', grad_output[0])*args.learning_rate
            grad_i = None
            
            if args.compression_type == 'topk':
                _, ind = calculate_matrix_norm(A, B)
                ind = ind[:args.compression]
            elif args.compression_type == 'weightedk':
                prob, _ = calculate_matrix_norm(A, B)
                ind = np.random.choice(args.batch_size, args.compression, replace=False, p=prob.cpu().numpy())
            elif args.compression_type == 'randk':
                ind = np.random.choice(args.batch_size, args.compression, replace=False)
            
            grad_w = torch.einsum('ji, jk -> ik', A[ind], B[ind])
            
            grad_input = (grad_b, grad_i, grad_w)
        else:
            grad_input = (grad_input[0]*args.learning_rate, grad_input[1], grad_input[2]*args.learning_rate)

    return grad_input
#-----------------------------------------------------------------------------
'Data Preparation'
data = np.genfromtxt("./data/EnergyEfficiency/data.csv", delimiter=",", skip_header=1)

input_data, output_data = prepare_data(data)
x_train, y_train, x_test, y_test = split(input_data, output_data)
x_train, x_test = normalize(x_train, x_test)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

dataset_train = TensorDataset(x_train, y_train)
dataset_test = TensorDataset(x_test, y_test)

train_size = len(x_train)
test_size = len(x_test)

trainloader = DataLoader(dataset_train, batch_size = args.batch_size, shuffle=True)
testloader = DataLoader(dataset_test, batch_size = test_size, shuffle=False)
#------------------------------------------------------------------------------
'Training'
print('Energy Efficieny - ' + args.compression_type  + ' - Memory: ' + str(args.memory_state) +  ' - K/D = ' + str(args.compression) + '/' + str(args.batch_size) +
      ' - Start Time: ' + datetime.now().strftime("%H:%M:%S"))

simulation_start_time = time.time()

losses = np.zeros((args.simulation_num, args.epoch_num))
losses_val = np.zeros((args.simulation_num, args.epoch_num))

speed = np.zeros(args.simulation_num)

for sim_num in range(args.simulation_num):
    'Model Initialization'
    net = Net(args.memory_state).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=args.momentum, nesterov=args.nesterov)
    
    class_dense = net.dense.__class__
    register_hook(net)
    start_time = time.time()
    for epoch in range(args.epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss_val = 0.0

        net.train(True)
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            'Foward Propagation'
            output = net(inputs)
            loss = criterion(output, labels)
    
            'Backward Propagation'
            #Automatically calculates the gradients for trainable weights, access as weight.grad
            loss.backward()
        
            #Performs the weight Update
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
    

        net.train(False)
        for input_val, labels_val in testloader:
            input_val, labels_val = input_val.to(device), labels_val.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            'Foward Propagation'
            output_val = net(input_val)
            loss = criterion(output_val, labels_val)
            
            # print statistics
            running_loss_val += loss.item()
    

        losses[sim_num, epoch] = running_loss / (train_size//args.batch_size)
        
        losses_val[sim_num, epoch] = running_loss_val

    speed[sim_num] = time.time() - start_time
    if args.verbose_state:
        print('Model: %d, Loss: %.5f, Val_Loss: %.5f, Time: %d s' % (sim_num + 1,
               losses[sim_num,-1], losses_val[sim_num,-1], speed[sim_num]))

if args.save_state:
    data = {'loss': losses,
            'loss_val': losses_val,
            'time': speed,
            'batch_sizes': args.batch_size,
            'epochs': args.epoch_num,
            'learning_rate': args.learning_rate,
            'simulation_num': args.simulation_num,
            'compression': args.compression,
            'compression_type': args.compression_type}
    
    if args.memory_state == None:
        pickle.dump(data, open(args.folder_path + 'Energy_Efficiency_epochs_' +
               str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) +
               '_compression' + str(args.compression) + '.p', "wb" ) )
    elif args.memory_state == True:
        pickle.dump(data, open(args.folder_path + 'Energy_Efficiency_epochs_' + args.compression_type + '_withMem_epochs' +
               str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) +
               '_compression' + str(args.compression) +
               '_decay' + str(args.memory_decay_rate) + '.p', "wb" ) )
    else:
        pickle.dump(data, open(args.folder_path + 'Energy_Efficiency_epochs_' + args.compression_type + '_epochs' +
               str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) +
               '_compression' + str(args.compression) +
               '_decay' + str(args.memory_decay_rate) + '.p', "wb" ) )

if args.plot_state:
    plt.figure()
    plt.title('Loss')
    plt.plot(np.sqrt(losses.mean(axis=0)), color='C2', label='Train Loss')
    plt.plot(np.sqrt(losses_val.mean(axis=0)), color='C3',label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.legend()
    plt.grid()    
    
    plt.figure()
    plt.title('Prediction for Validation Data - Last Simulation')
    plt.plot(output_val.cpu().detach().numpy(), color='C0', label='Prediction')
    plt.plot(labels_val.cpu().detach().numpy(), color='C2',label='Label')
    plt.xlabel('Heating Load')
    plt.ylabel('#th Case')
    plt.legend()
    plt.grid()
print('Models: %d, Loss: %.5f, Val_Loss: %.5f, Avg. Time: %d s' % (args.simulation_num,
        losses[:,-1].mean(), losses_val[:,-1].mean(), speed.mean()))

print('Energy Efficieny  - ' + args.compression_type + ' - Memory: ' + str(args.memory_state) + ' - K/D = ' + str(args.compression) + '/' + str(args.batch_size) +
      ' - End Time: ' + datetime.now().strftime("%H:%M:%S"))

print('Elapsed Time: %d m' % ((time.time() - simulation_start_time)/60))
