import numpy as np
import matplotlib.pyplot as plt
from pyPCEnT2 import pyPCEnT
from pyPCEnT2.functions import gen_Gaussian_lineshape, fit_poly8 
from pyPCEnT2.units import hbar, kcal2eV, massH, massD

# define conditions
T = 77
Vel = 0.034


# double well potentials calculated using TDDFT
rp = np.array([-0.614,-0.550,-0.485,-0.420,-0.356,-0.291,-0.226,-0.162,-0.097,-0.032,0.032,0.097,0.162,0.226,0.291,0.356,0.420,0.485,0.550,0.614])
E_GS = np.array([2.208,1.454,0.977,0.706,0.581,0.548,0.562,0.590,0.604,0.589,0.534,0.441,0.317,0.180,0.060,0.000,0.062,0.334,0.943,2.065])
E_LES = np.array([5.283,4.534,4.061,3.794,3.673,3.646,3.680,3.688,3.634,3.646,3.602,3.513,3.392,3.257,3.138,3.078,3.140,3.413,4.022,5.144])
E_LEPT = np.array([4.847,4.134,3.706,3.495,3.452,3.509,3.620,3.801,3.949,4.073,4.157,4.194,4.189,4.157,4.128,4.144,4.270,4.597,5.250,6.411])

GSProtonPot = fit_poly8(rp, E_GS)
ReacProtonPot = fit_poly8(rp, E_LES) 
ProdProtonPot = fit_poly8(rp, E_LEPT)


# parameters used to generate Gaussian line shapes
hbaromega_Dem = 2.83          # transition energy in eV  for donor emission between the two minima, without any ZPEs 
hbaromega_Aabs = 3.10         # transition energy in eV  for acceptor absorption between the two minima, without any ZPEs
S_Dem = 0.50            # Stokes shift (2 x reorganization energy) in eV for donor emission
S_Aabs = 0.91           # Stokes shift (2 x reorganization energy) in eV for acceptor absorption 

Dem = gen_Gaussian_lineshape(hbaromega_Dem, S_Dem, T=T)
Aabs = gen_Gaussian_lineshape(hbaromega_Aabs, S_Aabs, T=T)

dG = hbaromega_Aabs - hbaromega_Dem - (S_Dem + S_Aabs)/2
print(f'dG = {dG:.2f} eV')

# set up system and do a calculation
system = pyPCEnT(GSProtonPot, ReacProtonPot, ProdProtonPot, Dem, Aabs, Vel=Vel)
system.calculate(massH, T=T)


#========================================================
# Plot 1: proton vibrational wave functions 
#========================================================

fig = plt.figure(figsize=(5,7))
gs = fig.add_gridspec(2, hspace=0)
ax1,ax2 = gs.subplots(sharex=True, sharey=True)

Evib_GS, wfc_GS = system.get_ground_proton_states()
Evib_reactant, wfc_reactant = system.get_reactant_proton_states()
Evib_product, wfc_product = system.get_product_proton_states()
rp = system.rp

if Evib_product[0] < Evib_reactant[0]:
    dEr = 0
    dEp = -Evib_product[0] + Evib_reactant[0]
else:
    dEr = Evib_product[0] - Evib_reactant[0]
    dEp = 0

ax1.plot(-rp, ReacProtonPot(rp)+dEr, 'b', lw=2)
ax1.plot(-rp, ProdProtonPot(rp)+dEp, 'r', lw=2)
scale_wfc = 0.06        # we will plot wave functions and energies in the same plot, this factor scales the wave function for better visualization

for i, (Ei, wfci) in enumerate(zip(Evib_reactant[:1], wfc_reactant[:1])):
    # change the sign of the vibrational wave functions for better visualization
    # make the largest amplitude positive
    sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
    ax1.plot(-rp, Ei+dEr+scale_wfc*sign*wfci, 'b-', lw=1, alpha=(1-0.12*i))
    ax1.fill_between(-rp, Ei+dEr+scale_wfc*sign*wfci, Ei+dEr, color='b', alpha=0.4)

for i, (Ei, wfci) in enumerate(zip(Evib_product[:5], wfc_product[:5])):
    sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
    ax1.plot(-rp, Ei+dEp+scale_wfc*sign*wfci, 'r-', lw=1, alpha=(1-0.12*i))
    ax1.fill_between(-rp, Ei+dEp+scale_wfc*sign*wfci, Ei+dEp, color='r', alpha=0.4)

ax1.set_xlim(-0.8,0.8)
ax1.set_ylim(0,1.3)
ax1.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
ax1.set_ylabel(r'$E$ / eV', fontsize=16)
ax1.tick_params(labelsize=14)


ax2.plot(-rp, GSProtonPot(rp), 'k', lw=2)
scale_wfc = 0.06        # we will plot wave functions and energies in the same plot, this factor scales the wave function for better visualization

for i, (Ei, wfci) in enumerate(zip(Evib_GS[:1], wfc_GS[:1])):
    sign = 1 if np.abs(np.max(wfci)) > np.abs(np.min(wfci)) else -1
    ax2.plot(-rp, Ei+scale_wfc*sign*wfci, 'k-', lw=1, alpha=(1-0.12*i))
    ax2.fill_between(-rp, Ei+scale_wfc*sign*wfci, Ei, color='k', alpha=0.4)

ax2.set_xlim(-0.65,0.75)
ax2.set_ylim(0,1.3)
ax2.set_xlabel(r'$r_{\rm p}\ /\ \rm\AA$', fontsize=16)
ax2.set_ylabel(r'$E$ / eV', fontsize=16)
ax2.set_xticks(np.arange(-0.6,0.8,0.2))
ax2.tick_params(labelsize=14)

plt.tight_layout()
plt.show()


#========================================================
# Plot 2: spectra and their overlap
#========================================================


fig = plt.figure(figsize=(5,7))
gs = fig.add_gridspec(2, hspace=0)
ax1,ax2 = gs.subplots(sharex=True, sharey=True)

alpha = 0.4
hbaromega = system.hbaromega
total_Dem, total_Aabs = system.calc_total_spectra(T=T)

ax1.plot(hbaromega, total_Dem(hbaromega), 'b-', lw=2, label=r'$\tilde{L}_{\rm D,em}(\omega)$')
ax1.plot(hbaromega, total_Aabs(hbaromega), 'r-', lw=2, label=r'$\tilde{L}_{\rm A,abs}(\omega)$')

ax1.fill_between(hbaromega, np.min( np.array([total_Dem(hbaromega), total_Aabs(hbaromega)]), axis=0), color=(0.7,0,1), alpha=alpha)

ax1.legend(fontsize=15,frameon=False, loc=1, bbox_to_anchor=(1.03,1.03))
ax1.set_xlim(2.0,5.0)
ax1.set_ylim(0,50)
ax1.set_xlabel(r'$\hbar\omega$ / eV', fontsize=16)
ax1.set_ylabel(r'$\tilde{L}(\omega)/\hbar$', fontsize=16)
ax1.tick_params(labelsize=14)


ax2.plot(hbaromega, Dem(hbaromega-Evib_reactant[0]+Evib_GS[0]), 'b-', lw=2, label=r'$L_{\rm D,em}(\omega-\omega_{\mu 0}^{\rm D})$')
ax2.plot(hbaromega, Aabs(hbaromega-Evib_product[0]+Evib_GS[0]), 'r-', lw=2, label=r'$L_{\rm A,abs}(\omega-\omega_{0 \nu}^{\rm A})$')
ax2.fill_between(hbaromega, np.min( np.array([Dem(hbaromega-Evib_reactant[0]+Evib_GS[0]), Aabs(hbaromega-Evib_product[0]+Evib_GS[0])]), axis=0), color=(0.7,0,1), alpha=alpha)

for i in range(1,4):
    ax2.plot(hbaromega, Dem(hbaromega-Evib_reactant[i]+Evib_GS[0]), 'b-', lw=2, alpha=0.6-0.15*i)
    ax2.plot(hbaromega, Aabs(hbaromega-Evib_product[i]+Evib_GS[0]), 'r-', lw=2, alpha=0.6-0.15*i)


ax2.legend(fontsize=15,frameon=False, loc=1, bbox_to_anchor=(1.03,1.03))
ax2.set_xlim(2.0,5.0)
ax2.set_ylim(0,60)
ax2.set_xlabel(r'$\hbar\omega$ / eV', fontsize=16)
ax2.set_ylabel(r'$L(\omega)/\hbar$', fontsize=16)
ax2.set_xticks(np.arange(2.0,5.5,0.5))
ax2.set_yticks(np.arange(0,60,10))
ax2.tick_params(labelsize=14)

plt.tight_layout()
plt.show()


#========================================================
# Calculation: rate constant for H and D 
#========================================================

k_tot_H = system.calculate(massH, T)
k_tot_D = system.calculate(massD, T)

print(f'At {T:d}K, k_tot(H) = {k_tot_H:.2e} s^-1')
print(f'At {T:d}K, k_tot(D) = {k_tot_D:.2e} s^-1')
print(f'At {T:d}K, KIE = {k_tot_H/k_tot_D:.2f}')
