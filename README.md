# pyPCEnT
Python implementation of the nonadiabatic PCEnT theory. A detailed description of the theory can be found in Ref. \[1\], the calculation is based on the following equation, 

$$k_{\rm PCEnT} = \frac{1}{4\pi^2\hbar^2}|V_{\rm el}|^2\sum_{\mu,\nu}P_{\mu}|S_{\mu\nu}|^2 \iint\mathrm{d}\omega_1\,\mathrm{d}\omega_2\,L_{\rm D,em}(\omega_1-\omega_{\mu 0}^{\rm eq,D}) K(\omega_1-\omega_2) L_{\rm A,abs}(\omega_2-\omega_{0\nu}^{\rm eq,A})$$

where $V_{\rm el}$ is the electronic coupling between reactant and product electronic states, $P_{\mu}$ is the Boltzmann population of the reactant vibronic states, $S_{\mu\nu}$ is the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $L_{\rm D,em}(\omega_1-\omega_{\mu 0}^{\rm eq,D})$ and $L_{\rm A,abs}(\omega_2-\omega_{0\nu}^{\rm eq,A})$ are normalized donor emission and adsorption line shape functions, and $K(\omega_1-\omega_2)$ is the convolution kernel which describes the reorganization of common vibrational modes shared by the donor and the acceptor.  


## Installation 
To use the kinetic model, simply download the code and add it to your `$PYTHONPATH` variable.

## Documentation

### Initialization

#### Required Quantities
To calculate the PCEnT rate constant using the nonadiabatic PCEnT theory, we need the following physical quantities: 

1. `GSProtonPot` (2D array or function): proton potential of the ground state
2. `ReacProtonPot` (2D array or function): proton potential of the reactant state
3. `ProdProtonPot` (2D array or function): proton potential of the product state
4. `DonorEmLineshape` (2D array or function): line shape function for donor emission
5. `AcceptorAbsLineshape` (2D array or function): line shape function for acceptor absorption

The input of proton potentials and line shape functions can be either a 2D array or a callable function. 

If these inputs are 2D array, a fitting will be performed to create a callable function for subsequent calculations. By default, the proton potentials will be fitted to an 8th-order polynormial, while the line shape functions will be fitted as a sum of Gaussian functions. The 2D array should have shape (N, 2), for proton potentials, the first row is the proton position in Angstrom, the second row is the potential energy in eV, for line shape functions, the first row is $\hbar\omega$ in eV, the second row is the lins shape function in eV<sup>-1</sup>. 

If these inputs are functions, they must only take one argument, for proton potentials, it is the proton position in Angstrom, for line shape functions it is $\hbar\omega$ in eV. The unit of the proton potentials should be eV. The unit of the line shape functions should be eV<sup>-1</sup>

6. `ConvolKernel` (2D array or function): convolution kernel which describes the reorganization of common vibrational modes shared by the donor and the acceptor, default = None

The default setting `ConvolKernel = None` indicates that there is no reorganization of common modes. One important example of such case is intermolecular PCEnT process, where there is no common mode. 

7. `Vel (float)`: electronic coupling between reactant and product electronic states in eV, default = 0.0434 eV = 1 kcal/mol

Since PCEnT is a non-radiative process, the electronic ground state does not participate in the process. However, the proton potential of the ground state is essential to calculate the donor emission and acceptor absorption spectra for analyzing the spectral overlap of the system. 



## Citation
If you find this kinetic model helpful, please cite the following paper: 
1. Cui, K.; Hammes-Schiffer, S.; Theory for Proton-Coupled Energy Transfer, *J. Chem. Phys.* **2024**, *161*, 034113. DOI:[10.1063/5.0217546](https://doi.org/10.1063/5.0217546)
