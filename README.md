## Paper
Bo Zhao\*, Iordan Ganev\*, Robin Walters, Rose Yu, Nima Dehmamy (\*equal contribution) [Symmetries, Flat Minima, and the Conserved Quantities of Gradient Flow](https://arxiv.org/abs/2210.17216). *International Conference on Learning Representations (ICLR)*, 2023.

## Abstract
Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Our framework uses equivariances of the activation functions and can be applied to different layer architectures. To generalize this framework to nonlinear neural networks, we introduce a novel set of nonlinear, data-dependent symmetries. These symmetries can transform a trained model such that it performs similarly on new samples, which allows ensemble building that improves robustness under certain adversarial attacks. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability.

## Requirement 
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)


## Reproducing experiments in the paper
Dynamics of conserved quantities in gradient descent (Figure 3):

```
python Q_dynamics_elementwise.py
```

Distribution of conserved quantities for two-layer neural networks under Xavier initialization (Figure 4, 5):

```
python Q_distribution.py
```

Convergence rate of two-layer networks with elementwise activations, initialized with different conserved quantity values (Figure 6):

```
python Q_convergence_elementwise.py
```

Convergence rate of two-layer networks with radial activations, initialized with different conserved quantity values (Figure 7):

```
python Q_convergence_radial.py
```


Eigenvalues of the Hessian from trained models initialized with different conserved quantity values (Figure 9):

```
python Q_hessian_eigenvalues.py
```

The ensemble and adversarial experiments (Figure 10, 11) can be found in Jupyter Notebooks `ensemble_CIFAR.ipynb` and `adversarial_CIFAR.ipynb`. All results are saved in the directory `figures/`.

## Cite
```
@article{zhao2023symmetries,
  title={Symmetries, Flat Minima, and the Conserved Quantities of Gradient Flow},
  author={Bo Zhao and Iordan Ganev and Robin Walters and Rose Yu and Nima Dehmamy},
  journal={International Conference on Learning Representations},
  year={2023}
}
```
