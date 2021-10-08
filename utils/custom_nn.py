# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:41:21 2021

@author: Eduin Hernandez
"""
import torch
from torch import Tensor
import torch.nn as nn

class LinearMem(nn.Linear):
    def __init__(self, in_features: int, out_features: int, memory_state: bool = True, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        
        self.input = None
        self.memA = None
        self.memB = None
        self.init_state = True
        self.memory_state = memory_state
    
    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        self.input = input
        
        'Must Improve'
        if self.init_state and self.memory_state:
            self.memA = torch.zeros_like(input)
            self.memB = torch.zeros_like(output)
            
            self.init_state = False
        return output