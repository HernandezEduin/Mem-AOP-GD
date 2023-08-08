# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:18:26 2021

@author: Eduin Hernandez
Summary: Plots figure 3.b of paper. K/D = 16/64.
"""

from mnist_plots import run_plot, parse_args

args = parse_args()

args.dataset = 'MNIST'
args.epoch_num = 30
args.batch_size = 64
args.compression = 16
args.compressions_type = ['baseline', 'topk', 'weightedk', 'randk']
args.memory_states = ['True', 'False']
args.memory_decay = 1.0
args.simulation_num = 10

run_plot(args)
