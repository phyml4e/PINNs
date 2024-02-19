# PINNs

In this repository, you will find the codes for the application of deep learning models (especially neural networks) to solve different ODEs and PDEs which mainly occur in different engineering problems. The focus is mainly on solids and also the mechanics of materials at the microscale. The main theory behind this approach is based on the idea of physics-informed neural networks. The main platform to implement the routines here is the Sciann package, but certainly, the ideas can be taken to any other platforms as well.

1- How to Get Started: a simple code for solving a first-order ODE using Physics-Informed Neural Networks (PINN).
Additional details can be found in the Google Colab file and the appendix of the following article:"
https://www.sciencedirect.com/science/article/abs/pii/S0045782522005722 (arxiv version: https://arxiv.org/abs/2206.13103)
![ODE](https://github.com/phyml4e/PINNs/assets/109462709/6b2484fa-05c2-492a-ba14-8ce25f6c6757)


2- To employ the mixed formulation for solving a mechanical equilibrium and steady-state thermal diffusion problem within a two-phase microstructure, please refer to:
https://www.sciencedirect.com/science/article/abs/pii/S0045782522005722

![alt-text](https://github.com/phyml4e/PINNs/blob/main/mixformulation.gif)


3- For problems occurring in multiphysics environments involving the coupling of at least two different physics, please see https://arxiv.org/abs/2302.04954. This code is tailored specifically for a mixed-PINN formulation addressing a thermo-mechanical problem within the microstructure of a two-phase material. Future updates will include additional investigations involving other physics.
![multi_phy](https://github.com/phyml4e/PINNs/assets/109462709/a6e4379e-f604-4c8f-b2e1-4713d5284d3e)


4- To address coupled nonlinear relations in solid constitutive laws, please refer to https://arxiv.org/abs/2304.06044.
![comm-pinn](https://github.com/phyml4e/PINNs/assets/109462709/d2addf6a-f4ea-4bfe-86b9-cb6c010046ff)
