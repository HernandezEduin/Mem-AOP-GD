# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:19:34 2021

@author: Eduin Hernandez
Summary: For training on all schemes and ploting for comparison.
"""
import argparse
import os
from utils.parser_utils import str2bool
from mnist_simulations import run_experiment
from mnist_plots import run_plot
#------------------------------------------------------------------------------
'Argparser'
def parse_args(): #Parser with Default Values
    parser = argparse.ArgumentParser(description='Variables for MNIST Training')

    'Model Details'
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size for Training')
    parser.add_argument('--epoch-num', type=int, default=30, help='Total Epochs used for Training')
    
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for the model')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    parser.add_argument('--nesterov', type=str2bool, default='False', help='Whether to use Nesterov Momentum')
    parser.add_argument('--train-state', type=str2bool, default='True', help='Whether to train the networks. If false, only plots the results.')
    
    'Dataset'
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to Use. Can be MNIST, KMNIST, or FashionMNIST')
    
    'Device'
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. Either cpu or cuda for gpu')

    'Compression'
    parser.add_argument('--compression', type=int, default=32, help='Number of vectors/batches to keep.')
    parser.add_argument('--compressions-type', type=list, default=['topk', 'weightedk', 'randk'], help='Compression scheme type: topk, weightedk, and/or randk.')
    parser.add_argument('--memory-decay-rate', type=float, default=1.0, help='Memory Decay Rate')
    
    'Simulations'
    parser.add_argument('--simulation-num', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--memory-states', type=list, default=['True', 'False'], help='Either True, False, or None where None is the baseline')
    
    'Print and Plot'
    parser.add_argument('--plot-state', type=str2bool, default='False', help='Whether to plot the results')
    parser.add_argument('--plot-comparison', type=str2bool, default='True', help='Whether to plot the comparison for all the schemes.')
    parser.add_argument('--verbose-state-sim', type=str2bool, default='False', help='Whether to print the results per simulation')
    parser.add_argument('--verbose-state', type=str2bool, default='True', help='Whether to print the results at end of code')
    
    'Save Details'
    parser.add_argument('--save-state', type=str2bool, default='True', help='Whether to save the results of the training')
    parser.add_argument('--folder-path', type=str, default='./results/mnist/', help='Folder Save file path for statistics')

    'Leave these empty'
    parser.add_argument('--compression-type', type=str, default='randk', help='Leave empty!')
    parser.add_argument('--memory-state', type=str2bool, default='False', help='Leave empty!')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.train_state:
        'Training Baseline if it does not exist'
        if not os.path.isfile(args.folder_path + args.dataset + '_epochs' + 
                       str(args.epoch_num) + '_runs' + str(args.simulation_num) + '_vectors' + str(args.batch_size) + '.p'):
            args.memory_state = str2bool('None')
            run_experiment(args)
        
        'Training all other schemes'
        for comp_type in args.compressions_type:
            args.compression_type = comp_type
            for mem_state in args.memory_states:
                args.memory_state = str2bool(mem_state)
                run_experiment(args)

    if args.plot_comparison:
        args.compressions_type.insert(0,'baseline')
        run_plot(args)