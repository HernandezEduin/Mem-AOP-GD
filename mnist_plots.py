# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:23:05 2021

@author: Eduin Hernandez
Summary: For plotting comparison of various schemes.
"""
#Storage
import argparse
from utils.parser_utils import str2bool
import pickle

#Math
import numpy as np

#Plotting
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------        
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for MNIST Plotting')
    
    'Model Details'
    parser.add_argument('--epoch-num', type=int, default=30, help='End Iterations for Training')
    
    'Dataset'
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to Use. Can be MNIST, KMNIST, or FashionMNIST')
    
    'Compression'
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size used')
    parser.add_argument('--compression', type=int, default=32, help='Number of vectors/batches to kept')
    parser.add_argument('--compressions-type', type=list, default=['baseline', 'topk', 'weightedk', 'randk'], help='Compression scheme type: topk, weightedk, or randk')
    parser.add_argument('--memory-states', type=list, default=['True', 'False'], help='Either True, False, or None where None is the baseline')
    parser.add_argument('--memory-decay-rate', type=float, default=1.0, help='Memory Decay Rate')
    
    'Simulations'
    parser.add_argument('--simulation-num', type=int, default=10, help='Number of simulations to run')
    
    'Load Details'
    parser.add_argument('--folder-path', type=str, default='./results/mnist/', help='Folder Save file path for statistics')
    
    args = parser.parse_args()
    return args

#------------------------------------------------------------------------------
def get_styles(pet):
    linestyle = 'dashed'
    color = 'C0'
    if 'mem' in pet or 'baseline' in pet:
        linestyle = 'solid'
    
    if 'topk' in pet:
        color = 'C1'
    elif 'weightedk' in pet:
        color = 'C2'
    elif 'randk' in pet:
        color = 'C3'
    return color, linestyle

def run_plot(args):
    data = {}
    for pet in args.compressions_type:
        if pet == 'baseline':
            data['baseline'] = pickle.load(open(args.folder_path + args.dataset + '_epochs' + str(args.epoch_num) +
                                                '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) + '.p', "rb" ) )
        else:
            for mem_state in args.memory_states:
                mem_state = str2bool(mem_state)
                if mem_state == True:
                    data[pet + ' with mem'] = pickle.load(
                        open(args.folder_path + args.dataset + '_' + pet + '_withMem_epochs' + str(args.epoch_num) +
                             '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) + '_compression' + str(args.compression) +
                             '_decay' + str(args.memory_decay_rate) + '.p', "rb" ))
                elif mem_state == False:
                    data[pet] = pickle.load(
                        open(args.folder_path + args.dataset + '_' + pet + '_epochs' + str(args.epoch_num) +
                             '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) + '_compression' + str(args.compression) +
                             '_decay' + str(args.memory_decay_rate) + '.p', "rb" ))
                else:
                    assert False
    
    keys = list(data.keys())
    
    plt.close('all')    
    for pet in keys:
        color, linestyle = get_styles(pet)
        plt.figure(1)
        plt.plot(data[pet]['loss'].mean(axis=0), linestyle=linestyle, color=color, label= pet)
        
        plt.figure(2)
        plt.plot(data[pet]['loss_val'].mean(axis=0), linestyle=linestyle, color=color, label= pet)
        
        plt.figure(3)
        plt.plot(data[pet]['acc'].mean(axis=0), linestyle=linestyle, color=color, label= pet)
        
        plt.figure(4)
        plt.plot(data[pet]['acc_val'].mean(axis=0), linestyle=linestyle, color=color, label= pet)
    
    plt.figure(1)
    plt.title('Train Loss - Efficient Energy' + '\nK/D= ' + str(args.compression) + '/' + str(args.batch_size) + ' - $\\alpha$ = ' + str(args.memory_decay_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.figure(2)
    plt.title('Validation Loss - Efficient Energy' + '\nK/D= ' + str(args.compression) + '/' + str(args.batch_size) + ' - $\\alpha$ = ' + str(args.memory_decay_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.figure(3)
    plt.title('Train Acc - Efficient Energy' + '\nK/D= ' + str(args.compression) + '/' + str(args.batch_size) + ' - $\\alpha$ = ' + str(args.memory_decay_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid()
    
    plt.figure(4)
    plt.title('Validation Acc - Efficient Energy' + '\nK/D= ' + str(args.compression) + '/' + str(args.batch_size) + ' - $\\alpha$ = ' + str(args.memory_decay_rate))
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid()
    
if __name__ == '__main__':
    args = parse_args()
    run_plot(args)
