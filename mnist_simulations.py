# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 14:32:49 2021

@author: Eduin Hernandez
Summary: For plotting comparison of various schemes.
"""
#-----------------------------------------------------------------------------
'Libraries'
#Time
import time
from datetime import datetime

#Storage
import argparse
from utils.parser_utils import str2bool
import pickle

#Math
import numpy as np

#DNN
import torch
import torch.nn as nn
from utils.custom_nn import LinearMem
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms

#Plotting
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for MNIST/KMNIST/FashionMNIST Training')

    'Model Details'
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size for Training')
    parser.add_argument('--epoch-num', type=int, default=30, help='Total Epochs used for Training')
    
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for the model')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    parser.add_argument('--nesterov', type=str2bool, default='False', help='Whether to use Nesterov Momentum')
       
    'Dataset'
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to Use. Can be MNIST, KMNIST, or FashionMNIST')
    
    'Device'
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Either cpu or cuda for gpu')

    'Compression'
    parser.add_argument('--compression', type=int, default=32, help='Number of vectors/batches to keep.')
    parser.add_argument('--compression-type', type=str, default='randk', help='Compression scheme type: topk, weightedk, or randk')
    parser.add_argument('--memory-decay-rate', type=float, default=1.0, help='Memory Decay Rate')
    
    'Simulations'
    parser.add_argument('--simulation-num', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--memory-state', type=str2bool, default='True', help='Either True, False, or None where None is the baseline')
    
    'Print and Plot'
    parser.add_argument('--plot-state', type=str2bool, default='True', help='Whether to plot the results')
    parser.add_argument('--verbose-state-sim', type=str2bool, default='True', help='Whether to print the results per simulation')
    parser.add_argument('--verbose-state', type=str2bool, default='True', help='Whether to print the results at end of code')
    
    'Save Details'
    parser.add_argument('--save-state', type=str2bool, default='True', help='Whether to save the results of the training')
    parser.add_argument('--folder-path', type=str, default='./results/mnist/', help='Folder Save file path for statistics')
    
    args = parser.parse_args()
    return args

#------------------------------------------------------------------------------
'Model/Network Declaraction'
class Net(nn.Module):
    def __init__(self, mem_state: bool):
        super(Net, self).__init__()
        
        self.flat = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        
        self.dense = LinearMem(784,10, memory_state = mem_state, bias=True)
        
        
    def forward(self, x):                
        x = self.flat(x)
        x = self.softmax(self.dense(x))
        
        return x
#------------------------------------------------------------------------------
'Modify Gradients of a Layer'
def register_hook(model):
    model.dense.register_backward_hook(memorygrad)

def calculate_matrix_norm(A, B): 
    norms = torch.square(A.norm(dim=1, p='fro')*B.norm(dim=1, p='fro'))
    norms = norms/norms.sum()
    return norms, norms.argsort(descending=True)

def memorygrad(self, grad_input, grad_output):
    global args
    if self.__class__ == class_dense:                
        if self.memory_state == True: #With Memory Schemes
            """Manually Recalculating the Gradients with Memory"""
            A = self.memA + self.input*np.sqrt(args.learning_rate)
            B = self.memB + grad_output[0]*np.sqrt(args.learning_rate)
            
            grad_b = torch.einsum('ij -> j', grad_output[0])*args.learning_rate #Bias gradient
            grad_i = None
            
            'Compression Schemes'
            if args.compression_type == 'topk':
                _, ind = calculate_matrix_norm(A, B)
                ind = ind[:args.compression]
            elif args.compression_type == 'weightedk':
                prob, _ = calculate_matrix_norm(A, B)
                ind = np.random.choice(args.batch_size, args.compression, replace=False, p=prob.cpu().numpy())
            elif args.compression_type == 'randk':
                ind = np.random.choice(args.batch_size, args.compression, replace=False)
            
            grad_w = torch.einsum('ji, jk -> ik', A[ind], B[ind]) #Use only K vectors for the multiplication
            A[ind] = 0      #Remove what was used for calculation
            B[ind] = 0      #Remove what was used for calculation

            
            self.memA = torch.clone(A)*np.sqrt(args.memory_decay_rate)
            self.memB = torch.clone(B)*np.sqrt(args.memory_decay_rate)
            
            grad_input = (grad_b, grad_i, grad_w)
        elif self.memory_state == False: #Without Memory Schemes
            """Manually Recalculating the Gradients without Memory"""
            A = self.input*np.sqrt(args.learning_rate)
            B = grad_output[0]*np.sqrt(args.learning_rate)

            grad_b = torch.einsum('ij -> j', grad_output[0])*args.learning_rate #Bias gradient
            grad_i = None
            
            'Compression Schemes'
            if args.compression_type == 'topk':
                _, ind = calculate_matrix_norm(A, B)
                ind = ind[:args.compression]
            elif args.compression_type == 'weightedk':
                prob, _ = calculate_matrix_norm(A, B)
                ind = np.random.choice(args.batch_size, args.compression, replace=False, p=prob.cpu().numpy())
            elif args.compression_type == 'randk':
                ind = np.random.choice(args.batch_size, args.compression, replace=False)
            
            grad_w = torch.einsum('ji, jk -> ik', A[ind], B[ind]) #Use only K vectors for the multiplication
            
            grad_input = (grad_b, grad_i, grad_w) #Reassign the gradients
        else: #Baseline Scheme, no compression
            grad_input = (grad_input[0]*args.learning_rate, grad_input[1], grad_input[2]*args.learning_rate)

    return grad_input
#-----------------------------------------------------------------------------
def run_experiment(arg):
    global args
    args = args
    #-----------------------------------------------------------------------------
    'Preparing and Checking Parser'    
    if args.compression_type not in ['topk', 'randk', 'weightedk']:
        assert False, 'Compression type must be topk, randk, or weightedk!'
    
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    #-----------------------------------------------------------------------------
    'Data Preparation'
    transformTrain = transforms.Compose( [transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    transformTest = transforms.Compose( [transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    
    
    if args.dataset == 'MNIST':
       mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transformTrain) 
       mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transformTest)
    elif args.dataset == 'KMNIST':
       mnist_trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transformTrain) 
       mnist_testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transformTest)
    elif args.dataset == 'FashionMNIST':
       mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformTrain) 
       mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformTest)
    else:
        assert False, 'Selected the wrong dataset'
    
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch_size,
                                             shuffle=False)
    
    train_size = len(mnist_trainset)
    test_size = len(mnist_testset)
    #------------------------------------------------------------------------------
    'Training'
    if args.verbose_state:
        print('Device:', device.type)
        
        if args.memory_state == None:
            print(args.dataset + ' - Baseline - Start Time: ' + datetime.now().strftime("%H:%M:%S"))
        else:
            print(args.dataset + ' - ' + args.compression_type  + ' - Memory: ' + str(args.memory_state) +  ' - K/D = ' + str(args.compression) + '/' + str(args.batch_size) +
                  ' - Start Time: ' + datetime.now().strftime("%H:%M:%S"))
    
    simulation_start_time = time.time()
    
    losses = np.zeros((args.simulation_num, args.epoch_num))
    losses_val = np.zeros((args.simulation_num, args.epoch_num))
    
    acc = np.zeros((args.simulation_num, args.epoch_num))
    acc_val = np.zeros((args.simulation_num, args.epoch_num))
    
    speed = np.zeros(args.simulation_num)

        
    for sim_num in range(args.simulation_num):
        'Model Initialization'
        net = Net(args.memory_state).to(device)
        
        'Loss and Optimizer'
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1, momentum=args.momentum, nesterov=args.nesterov) #learning rate modified in memory_grad
            
        'Using hook to modify gradients'
        global class_dense
        class_dense = net.dense.__class__
        register_hook(net)
        start_time = time.time()
        
        'Train and Validate Network'
        for epoch in range(args.epoch_num):  # loop over the dataset multiple times
            correct = 0
            correct_val = 0
            running_loss = 0.0
            running_loss_val = 0.0
                
            'Training Phase'
            net.train(True)
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)
                if args.batch_size != labels.size()[0]:
                    'Skip last batch if it is of unequal size'
                    continue
            
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
        
                #statistics
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
            
            net.train(False)
            for input_val, labels_val in testloader:
                input_val, labels_val = input_val.to(device), labels_val.to(device)
                if args.batch_size != labels_val.size()[0]:
                    'Skip last batch if it is of unequal size'
                    continue
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                'Foward Propagation'
                output = net(input_val)
                loss = criterion(output, labels_val)
                
                #statistics
                running_loss_val += loss.item()
                _, predicted = torch.max(output.data, 1)
                correct_val += (predicted == labels_val).sum().item()
                
            #Statistics   
            acc[sim_num, epoch] = correct/train_size
            losses[sim_num, epoch] = running_loss / (train_size//args.batch_size)
            acc_val[sim_num, epoch] = correct_val/test_size
            losses_val[sim_num, epoch] = running_loss_val / (test_size//args.batch_size)
                
        speed[sim_num] = time.time() - start_time
        'Printing Statistics per Simulation'
        if args.verbose_state_sim:
            print('Model: %d, Acc: %.5f, Loss: %.5f, Val_Acc: %.5f, Val_Loss: %.5f, Time: %d s' % (sim_num + 1,
                   acc[sim_num,-1], losses[sim_num,-1], acc_val[sim_num,-1], losses_val[sim_num,-1], speed[sim_num]))
        
        #------------------------------------------------------------------------------
        'Storing Results'
        if args.save_state:
            data = {'acc': acc,
                    'acc_val': acc_val,
                    'loss': losses,
                    'loss_val': losses_val,
                    'time': speed,
                    'batch_sizes': args.batch_size,
                    'epochs': args.epoch_num,
                    'learning_rate': args.learning_rate,
                    'simulation_num': args.simulation_num,
                    'compression': args.compression,
                    'compression_type': args.compression_type}

            if args.memory_state == None:
                pickle.dump(data, open(args.folder_path + args.dataset + '_epochs_' +
                       str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) +
                        '.p', "wb" ) )
            elif args.memory_state == True:
                pickle.dump(data, open(args.folder_path  + args.dataset + '_' + args.compression_type + '_withMem_epochs' +
                       str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) +
                       '_compression' + str(args.compression) +
                       '_decay' + str(args.memory_decay_rate) + '.p', "wb" ) )
            else:
                pickle.dump(data, open(args.folder_path + args.dataset + '_' + args.compression_type + '_epochs' +
                       str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) +
                       '_compression' + str(args.compression) +
                       '_decay' + str(args.memory_decay_rate) + '.p', "wb" ) )
    
    #------------------------------------------------------------------------------
    'Plotting Results'
    if args.plot_state:
        plt.figure()
        plt.title('Loss')
        plt.plot(losses.mean(axis=0), color='C2', label='Train Loss')
        plt.plot(losses_val.mean(axis=0), color='C3',label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Categorical Cross Entropy')
        plt.legend()
        plt.grid()    
        
        plt.figure()
        plt.title('Acc')
        plt.plot(acc.mean(axis=0), color='C2', label='Train Acc')
        plt.plot(acc_val.mean(axis=0), color='C3',label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.grid()    
    #------------------------------------------------------------------------------
    'Printing Overall Statistics'
    if args.verbose_state:
        print('Models: %d, Avg. Acc: %.5f, Avg. Loss: %.5f, Avg. Val Acc: %.5f, Avg. Val Loss: %.5f, Time: %d s' % (args.simulation_num,
                   acc[:,-1].mean(), losses[:,-1].mean(), acc_val[:,-1].mean(), losses_val[:,-1].mean(), speed.mean()))
        if args.memory_state == None:
            print(args.dataset + ' - Baseline - End Time: ' + datetime.now().strftime("%H:%M:%S"))
        else:
            print(args.dataset + ' - ' + args.compression_type + ' - Memory: ' + str(args.memory_state) + ' - K/D = ' + str(args.compression) + '/' + str(args.batch_size) +
              ' - End Time: ' + datetime.now().strftime("%H:%M:%S"))
        
    print('Elapsed Time: %d m' % ((time.time() - simulation_start_time)/60))

if __name__ == '__main__':
    args = parse_args()

    run_experiment(args)        
