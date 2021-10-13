# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:27:14 2021

@author: Eduin Hernandez
Summary: Plots figure 2.b of paper. K/D = 9/144.
"""

from energy_efficiency_plots import run_plot, parse_args

args = parse_args()

args.epoch_num = 100
args.batch_size = 144
args.compression = 9
args.compressions_type = ['baseline', 'topk', 'weightedk', 'randk']
args.memory_states = ['True', 'False']
args.memory_decay = 1.0
args.simulation_num = 10

run_plot(args)