# pyPCEnT
Python implementation of the nonadiabatic PCEnT theory. A detailed description of the theory can be found in Ref. \[1\], the calculation is based on the following equation, 

```math
k_{\rm PCEnT} = \frac{1}{4\pi^2\hbar^2}|V_{\rm el}|^2\sum_{\mu,\nu}P_{\mu}|S_{\mu\nu}|^2 \iint\mathrm{d}\omega_1 \mathrm{d}\omega_2\ L_{\rm D,em}(\omega_1-\omega_{\mu 0}^{\rm D}) K(\omega_1-\omega_2) L_{\rm A,abs}(\omega_2-\omega_{0\nu}^{\rm A})
```

where $`V_{\rm el}`$ is the electronic coupling between reactant and product electronic states, $`P_{\mu}`$ is the Boltzmann population of the reactant vibronic states, $`S_{\mu\nu}`$ is the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $`L_{\rm D,em}(\omega_1-\omega_{\mu 0}^{\rm D})`$ and $`L_{\rm A,abs}(\omega_2-\omega_{0\nu}^{\rm A})`$ are normalized donor emission and adsorption line shape functions, and $`K(\omega_1-\omega_2)`$ is the convolution kernel which describes the reorganization of common vibrational modes shared by the donor and the acceptor. 

For simplicity, we may assume a Gaussian line shape for donor emission, acceptor absorption, and the convolution kernel, which corresponds to a harmonic bath at high temperature. The expressions of the Gaussian line shape functions are given by: 

```math
L_{\rm D,em}(\omega-\omega_{\mu 0}^{\rm D}) = \frac{2\pi\hbar}{\sqrt{2\pi s_{\rm D} k_{\rm B}T}}\exp\left(-\frac{\hbar^2(\omega-\omega_{\mu 0}^{\rm D})^2}{2 s_{\rm D} k_{\rm B}T}\right)
```

```math
L_{\rm A,abs}(\omega-\omega_{0 \nu}^{\rm A}) = \frac{2\pi\hbar}{\sqrt{2\pi s_{\rm A} k_{\rm B}T}}\exp\left(-\frac{\hbar^2(\omega-\omega_{0 \nu}^{\rm A})^2}{2 s_{\rm A} k_{\rm B}T}\right)
```

```math
K(\omega_1-\omega_2) = \frac{2\pi\hbar}{\sqrt{2\pi s_{\rm com} k_{\rm B}T}}\exp\left(-\frac{(\hbar(\omega_1-\omega_2) - s_{\rm com}/2)^2}{2 s_{\rm com} k_{\rm B}T}\right)
```

where $`\omega_{\mu 0}^{\rm D}`$ and $`\omega_{0 \nu}^{\rm A}`$ should be calculated as the vertical transition frequencies (with all nuclei other than the transferring proton fixed), $`k_{\rm B}T`$ is the thermal energy, and $`s_D`$ and $`s_A`$ are the Stokes shift of the transition, which defines the width of the line shape function. For harmonic surfaces, they are also related to the reorganization energy $`\lambda`$ of the transition via $`s = 2\lambda`$. $`s_{\rm com}`$ can also be viewed as a Stokes shift term, but it only includes the reorganzation of the common modes. When $`s_{\rm com} \rightarrow 0`$, $`K(\omega_1-\omega_2) \rightarrow 2\pi\delta(\omega_1-\omega_2)`$. 

## Installation 
To use the pyPCEnT module, simply download the code and add it to your `$PYTHONPATH` variable.

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

If these inputs are 2D arrays, a fitting will be performed to create a callable function for subsequent calculations. By default, the proton potentials will be fitted to an 8th-order polynormial, while the line shape functions will be fitted as a sum of Gaussian functions. The 2D array should have shape (N, 2), for proton potentials, the first row is the proton position in Angstrom, the second row is the potential energy in eV, for line shape functions, the first row is $`\hbar\omega`$ in eV, the second row is the lins shape function in eV<sup>-1</sup>. 

If these inputs are functions, they must only take one argument, for proton potentials, it is the proton position in Angstrom, for line shape functions it is $`\hbar\omega`$ in eV. The unit of the proton potentials should be eV. The unit of the line shape functions should be eV<sup>-1</sup>.

6. `ConvolKernel` (2D array or function): convolution kernel which describes the reorganization of common vibrational modes shared by the donor and the acceptor, default = None

The default setting `ConvolKernel = None` indicates that there is no reorganization of common modes. One important example of such case is intermolecular PCEnT process, where there is no common mode. 

7. `Vel` (float): electronic coupling between reactant and product electronic states in eV, default = 0.0434 eV = 1 kcal/mol

Since PCEnT is a non-radiative process, the electronic ground state does not participate in the process. However, the proton potential of the ground state is essential to calculate the total donor emission and acceptor absorption spectra for analyzing the spectral overlap of the system. 

#### Example
The following code set up an `pyPCEnT` object for rate constant calculation. 
```python
import numpy as np
from pyPCEnT import pyPCEnT
from pyPCEnT.functions import gen_Gaussian_lineshape, fit_poly8 

# define electronic coupling and temperature
Vel = 0.034
T = 77

# proton potentials for ground, reactant, and product states calculated using TDDFT
rp = np.array([-0.614,-0.550,-0.485,-0.420,-0.356,-0.291,-0.226,-0.162,-0.097,-0.032,0.032,0.097,0.162,0.226,0.291,0.356,0.420,0.485,0.550,0.614])
E_GS = np.array([2.208,1.454,0.977,0.706,0.581,0.548,0.562,0.590,0.604,0.589,0.534,0.441,0.317,0.180,0.060,0.000,0.062,0.334,0.943,2.065])
E_Reac = np.array([5.283,4.534,4.061,3.794,3.673,3.646,3.680,3.688,3.634,3.646,3.602,3.513,3.392,3.257,3.138,3.078,3.140,3.413,4.022,5.144])
E_Prod = np.array([4.847,4.134,3.706,3.495,3.452,3.509,3.620,3.801,3.949,4.073,4.157,4.194,4.189,4.157,4.128,4.144,4.270,4.597,5.250,6.411])

# use the fit_poly8 function defined in pyPCEnT.functions to generate callable functions as input quantities
GSProtonPot = fit_poly8(rp, E_GS)
ReacProtonPot = fit_poly8(rp, E_Reac) 
ProdProtonPot = fit_poly8(rp, E_Prod)

# parameters used to generate Gaussian line shapes
hbaromega_Dem = 2.83          # transition energy in eV  for donor emission between the two minima, without any ZPEs 
hbaromega_Aabs = 3.10         # transition energy in eV  for acceptor absorption between the two minima, without any ZPEs
S_Dem = 0.50                  # Stokes shift (2 x reorganization energy) in eV for donor emission
S_Aabs = 0.91                 # Stokes shift (2 x reorganization energy) in eV for acceptor absorption 

# use the gen_Gaussian_lineshape function defined in pyPCEnT.functions to generate callable functions as input quantities
Dem = gen_Gaussian_lineshape(hbaromega_Dem, S_Dem, T=T)
Aabs = gen_Gaussian_lineshape(hbaromega_Aabs, S_Aabs, T=T)

# set up system
system = pyPCEnT(GSProtonPot, ReacProtonPot, ProdProtonPot, Dem, Aabs, Vel=Vel)
```

#### Other Parameters
Other parameters that can be modified during the initialization are

8. `NStates` (int): number of proton vibrational states to be calculated, default = 10. One should test the convergence with respect to this quantity. 
9. `NGridPot` (int): number of grid points used for FGH calculation, default = 128
10. `NGridLineshape` (int): number of grip points used to calculate spectral overlap integral, defaut = 500
11. `FitOrder` (int): order of polynomial to fit the proton potential, default = 8, This is only useful when some of the proton potentials, `GSProtonPot`, `ReacProtonPot`, and `ProdProtonPot`, are provided as 2D arrays. Another possible value for this is 6.

When initializing, the program will automatically determine the proper ranges of $`\hbar\omega`$ or proton position to perform subsequent numerical calculations. Users could fine tune these ranges by parseing additional inputs `rmin`, `rmax`, `hbaromega_min`, and `hbaromega_max`. 

### Calculation
#### PCEnT Rate Constant
In a typical calculation of the nonadiabatic PCEnT rate constant, we first need to solve the 1D Schrödinger equations for the proton moving in the proton potentials associated with the reactant and product electronic states. This calculation yields the proton vibrational energy levels and wave functions, which in turn determine the Boltzmann population of the reactant vibronic states, $`P_{\mu}`$, the overlap integral between the proton vibrational wave functions associated with the reactant and product electronic states, $`S_{\mu\nu}`$, as well as the transition frequencies between all vibronic states, $`\omega_{\mu 0}^{\rm D}`$ and $`\omega_{0 \nu}^{\rm A}`$. We will calculate $`P_{\mu}`$, $`S_{\mu\nu}`$, $`\omega_{\mu 0}^{\rm D}`$, and $`\omega_{0 \nu}^{\rm A}`$ for all $`\mu`$ and $`\nu`$ from 0 to `NStates`. These quantities will be fed into the rate constant expression to give the final results. 

All these steps have been integrated in the method `pyPCEnT.calculate`.  This method takes two parameters, the mass of the particle, which should be set to the mass of the proton or deuterium, and the temperature of the system. It returns the calculated rate constant at the given condition. Follow by the previous example, we can calculate the PCEnT rate constant and the kinetic isotope effect (KIE) by: 
```python
from pyPCEnT.units import massH, massD

k_tot_H = system.calculate(massH, T)
k_tot_D = system.calculate(massD, T)
KIE = k_tot_H/k_tot_D
```

#### Analyze the Result
The calculated proton vibrational energy levels and wave functions can be accessed through
```python
Evib_GS, wfc_GS = system.get_ground_proton_states()
Evib_reactant, wfc_reactant = system.get_reactant_proton_states()
Evib_product, wfc_product = system.get_product_proton_states()
```
$`P_{\mu}`$, $`S_{\mu\nu}`$, and the spectral convolution integral, which is defined as

```math
I_{\mu\nu} = \iint\mathrm{d}\omega_1 \mathrm{d}\omega_2\ L_{\rm D,em}(\omega_1-\omega_{\mu 0}^{\rm D}) K(\omega_1-\omega_2) L_{\rm A,abs}(\omega_2-\omega_{0\nu}^{\rm A})
```

can be accessed using the methods
```python
Pu = system.get_reactant_state_distributions()
Suv = system.get_proton_overlap_matrix()
Iuv = system.get_spectral_overlap_matrix()
```

The contribution of a given $`(\mu,\nu)`$ pair to the total rate constant is given by

```math
k_{\mu\nu} = \frac{1}{4\pi^2\hbar^2}|V_{\rm el}|^2 P_{\mu}|S_{\mu\nu}|^2 I_{\mu\nu}
```

```math
{\rm \%\ Contrib.} = \frac{k_{\mu\nu}}{k_{\rm PCEnT}}
```

These quantities can be obtained from the code using
```python
kuv = system.get_kinetic_contribution_matrix()
k_tot = system.get_total_rate_constant()
percentage_contribution = kuv/k_tot
```

#### Total Donor Emission and Acceptor Absorption Spectra
In the expression of the nonadiabatic PCEnT rate constant, the line shape functions $L_{\rm D,em}$ and $L_{\rm A,abs}$ describe the transitions between **vibronic states**, whereas the experimentally measured donor emission and acceptor absorption spectra describe the transition between **electronic states**. The line shape functions corresponds to the experimental spectra, $`\tilde{L}_{\rm D,em}`$ and $`\tilde{L}_{\rm A,abs}`$, are given by

```math
\tilde{L}_{\rm D, em}(\omega) = \sum_{\mu,\sigma} P_{\mu} |S_{\mu\sigma}|^2 L_{\rm D, em}(\omega - \omega_{\mu \sigma}^{\rm D})
```
```math
\tilde{L}_{\rm A, abs}(\omega) = \sum_{\sigma',\nu} P_{\sigma'} |S_{\sigma'\nu}|^2 L_{\rm A, abs}(\omega - \omega_{\sigma'\nu}^{\rm A})
```

Note that these expressions involve different Boltzmann distributions and overlap integrals between proton vibtational wave functions. The parameters in the line shape functions $L_{\rm D,em}$ and $L_{\rm A,abs}$ are also different. Here $`\sigma`$ and $`\sigma'`$ label the proton vibrational states associated with the electronic ground state. Thus $`P_{\sigma'}`$ is the Boltzmann population of the vibronic states associated with the electronic ground state, $`S_{\mu\sigma}`$ and $`S_{\sigma'\nu}`$ are overlap integrals between the proton vibrational wave functions associated with the reactant/product state and the ground state. 

The total donor emission and acceptor absorption spectra can be calculated by calling
```python
total_Dem, total_Aabs = system.calc_total_spectra(massH, T)
```
This method also takes two parameters, the mass of the particle, which should be set to the mass of the proton or deuterium, and the temperature of the system. It returns two callable functions corresponds to $`\tilde{L}_{\rm D,em}`$ and $`\tilde{L}_{\rm A,abs}`$. One can make a plot of these line shape functions by
```python
import matplotlib.pyplot as plt

hbaromega = np.linspace(0, 6, 100)
plt.plot(hbaromega, total_Dem(hbaromega), 'b-', lw=2, label=r'$\tilde{L}_{\rm D,em}(\omega)$')
plt.plot(hbaromega, total_Aabs(hbaromega), 'r-', lw=2, label=r'$\tilde{L}_{\rm A,abs}(\omega)$')
```

The quantities $`P_{\sigma'}`$, $`S_{\mu\sigma}`$, and $`S_{\sigma'\nu}`$ can be accesses via
```python
Pw = system.get_ground_state_distributions()
Suw = system.get_proton_overlap_matrix_ReacGS()
Swv = system.get_proton_overlap_matrix_GSProd()
```

## Citation
If you used this code for your research, please cite the following paper: 
1. Cui, K.; Hammes-Schiffer, S.; Theory for Proton-Coupled Energy Transfer, *J. Chem. Phys.* **2024**, *161*, 034113. [DOI:10.1063/5.0217546](https://doi.org/10.1063/5.0217546)
