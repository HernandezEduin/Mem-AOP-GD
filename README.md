# Approximate Outer Product Gradient Descent with Memory
Code for the numerical experiment of the paper [Speeding-Up Back-Propagation in DNN: Approximate Outer Product with Memory](https://arxiv.org/abs/2110.09164).

## Environment
To use the code, install Anaconda with the following libraries:
* conda install scikit-image
* conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -c pytorch

For a preferred cuda version, use one of the following instead:
* conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch
* conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
* conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
* conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

## Reproducing Results
To reproduce any of the results, run one of the following codes for their corresponding plot in the paper:
* plot_figure_2a.py
* plot_figure_2b.py
* plot_figure_2c.py
* plot_figure_3a.py
* plot_figure_3b.py
* plot_figure_3c.py

## Training Models
To train a single layered network on any of the schemes, use any of the following with the desired argument parameters:
* mnist_simulations.py
* energy_efficiency_simulatons.py

To train a single layered network on all the schemes in one go, use any of the following with the desired argument paramters:
* mnist_run.py
* energy_efficiency_run.py


## Plotting Results
To plot the results from simulations, enter the argument parameters in the following codes:
* mnist_plots.py
* energy_efficiency_plots.py

## Reference
If you use this code, please cite the following [paper](https://arxiv.org/abs/2110.09164):

    @article{hernandez2021speeding,
          title={Speeding-Up Back-Propagation in DNN: Approximate Outer Product with Memory},
          author={Hernandez, Eduin E and Rini, Stefano and Duman, Tolga M},
          journal={arXiv preprint arXiv:2110.09164},
          year={2021}
    }
