# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:56:05 2021

@author: Eduin Hernandez
Summary: For training on all schemes and ploting for comparison.
"""
import argparse
from utils.parser_utils import str2bool
from energy_efficiency_simulations import run_experiment
from energy_efficiency_plots import run_plot
#------------------------------------------------------------------------------
'Argparser'
def parse_args(): #Parser with Default Values
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
    parser.add_argument('--compression', type=int, default=3, help='Number of partitions to keep.')
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
    parser.add_argument('--folder-path', type=str, default='./results/energy/', help='Folder Save file path for statistics')

    'Leave these empty'
    parser.add_argument('--compression-type', type=str, default='randk', help='Leave empty!')
    parser.add_argument('--memory-state', type=str2bool, default='False', help='Leave empty!')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if 'None' in args.memory_states:
        args.memory_states.remove('None')
        args.memory_state = str2bool('None')
        run_experiment(args)
    
    for comp_type in args.compressions_type:
        args.compression_type = comp_type
        for mem_state in args.memory_states:
            args.memory_state = str2bool(mem_state)
            run_experiment(args)

    if args.plot_comparison:
        args.compressions_type.insert(0,'baseline')
        run_plot(args)