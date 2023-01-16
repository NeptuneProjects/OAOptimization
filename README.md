# Ocean Acoustics Optimization

This package contains backend code for research on Bayesian optimization for estimation of ocean acoustic parameters.
The package is under active development.

While there are several libraries available for conducting Bayesian optimization, this package makes use of three open source libraries, two of which are maintained by Meta Research.
The libraries, listed according to increasing levels of abstraction, are:
1. [GPyTorch](https://gpytorch.ai), developed and maintained by researchers from multiple universities and Meta.
2. [BoTorch](https://botorch.org) by [Meta Open Source](https://code.facebook.com/projects/).
3. [Adaptive Experimentation Platform (Ax)](https://ax.dev) by [Meta Open Source](https://code.facebook.com/projects/).
All three of these libraries are built on [PyTorch](https://pytorch.org), thus enabling GPU acceleration of some computational tasks.
