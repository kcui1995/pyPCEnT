import numpy as np
import numpy.linalg as la
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.interpolate import interp1d
from .functions import *
from .units import *
from .FGH_1D import FGH_1D


class pyPCEnT(object):
    """
    This class set up the calculation for PCEnT rate constant, using Eq.(29) in Ref. 1
        [1] Cui and Hammes-Schiffer, J. Chem. Phys. 2024, 161, 034113 
    There are two important assumptions:
        1. The shape of the vibronic potential energy surface is solely determined by the electronic state, so the functional form of the line shape functions is independent of the (u,v) states. 
        2. The vibronic coupling V_uv = V_el * S_uv

    The input parameters are:
        GSProtonPot (2D array or function): proton potential of the ground state
        ReacProtonPot (2D array or function): proton potential of the reactant
        ProdProtonPot (2D array or function): proton potential of the product
        DonorEmLineshape (2D array or function): line shape function for donor emission
        AcceptorAbsLineshape (2D array or function): line shape function for acceptor absorption
        ConvolKernel (2D array or function): convolution kernel which describes the common vibrational modes shared by the donor and the acceptor
        Vel (float): electronic coupling between reactant and product states in eV, default = 0.0434 eV = 1 kcal/mol
        NStates (int): number of proton vibrational states to be calculated, default = 10
        NGridPot (int): number of grid points used for FGH calculation, default = 128
        NGridLineshape (int): number of grip points used to calculate spectral overlap integral, defaut = 500
        FitOrder (int): order of polynomial to fit the proton potential, default = 8 

    The program will automatically determine the ranges of \hbar\omega or proton position to perform subsequent calculations
    Users could fine tune these ranges by parseing additional inputs 'rmin', 'rmax', 'hbaromega_min', 'hbaromega_max'
    """
    def __init__(self, GSProtonPot, ReacProtonPot, ProdProtonPot, DonorEmLineshape, AcceptorAbsLineshape, ConvolKernel=None, Vel=0.0434, NStates=10, NGridPot=256, NGridLineshape=500, FitOrder=8, **kwargs):
        """
        *** Initialization ***
        The input of proton potentials and line shape functions can be either a 2D array or a callable function

        If these inputs are 2D array, a fitting will be performed to create a callable function for subsequent calculations
        By default, the proton potentials will be fitted to an 8th-order polynormial, while the line shape functions will be fitted as a sum of Gaussian functions
        The 2D array should have shape (N, 2),
        for proton potentials, the first row is the proton position in Angstrom, the second row is the potential energy in eV
        for line shape functions, the first row is \hbar\omega in eV, the second row is the lins shape function in eV^-1

        If these inputs are functions, they must only take one argument,
        for proton potentials, it is the proton position in Angstrom
        for line shape functions it is \hbar\omega in eV
        The unit of the proton potentials should be eV
        The unit of the line shape functions should be eV^-1

        ConvolKernel describes the common intramolecular vibration modes
        ConvolKernel == None means there is no common modes
        """
        if callable(GSProtonPot):
            self.GSProtonPot = GSProtonPot
        elif isarray(GSProtonPot) and len(GSProtonPot) == 2:
            r = GSProtonPot[0]
            pot = GSProtonPot[1]
            rmin0 = np.min(r)
            rmax0 = np.max(r)
            if FitOrder == 8:
                self.GSProtonPot = fit_poly8(r, pot)
            elif FitOrder == 6:
                self.GSProtonPot = fit_poly6(r, pot)
            else:
                raise ValueError("FitOrder must be 6 or 8")
        else:
            raise TypeError("'GSProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        if callable(ReacProtonPot):
            self.ReacProtonPot = ReacProtonPot
        elif isarray(ReacProtonPot) and len(ReacProtonPot) == 2:
            r = ReacProtonPot[0]
            pot = ReacProtonPot[1]
            rmin1 = np.min(r)
            rmax1 = np.max(r)
            if FitOrder == 8:
                self.ReacProtonPot = fit_poly8(r, pot)
            elif FitOrder == 6:
                self.ReacProtonPot = fit_poly6(r, pot)
            else:
                raise ValueError("FitOrder must be 6 or 8")
        else:
            raise TypeError("'ReacProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        if callable(ProdProtonPot):
            self.ProdProtonPot = ProdProtonPot
        elif isarray(ProdProtonPot) and len(ProdProtonPot) == 2:
            r = ProdProtonPot[0]
            pot = ProdProtonPot[1]
            rmin2 = np.min(r)
            rmax2 = np.max(r)
            if FitOrder == 8:
                self.ProdProtonPot = fit_poly8(r, pot)
            elif FitOrder == 6:
                self.ProdProtonPot = fit_poly6(r, pot)
            else:
                raise ValueError("FitOrder must be 6 or 8")
        else:
            raise TypeError("'ProdProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        if callable(DonorEmLineshape):
            self.DonorEmLineshape = DonorEmLineshape
        elif isarray(DonorEmLineshape) and len(DonorEmLineshape) == 2:
            hbaromega = DonorEmLineshape[0]
            lineshape = DonorEmLineshape[1]
            hbaromega_min1 = np.min(hbaromega)
            hbaromega_max1 = np.max(hbaromega)
            self.DonorEmLineshape = fit_Gaussian(hbaromega, lineshape)
        else:
            raise TypeError("'DonorEmLineshape' must be a 2D array with shape (N, 2) or a callable function")

        if callable(AcceptorAbsLineshape):
            self.AcceptorAbsLineshape = AcceptorAbsLineshape
        elif isarray(AcceptorAbsLineshape) and len(AcceptorAbsLineshape) == 2:
            hbaromega = AcceptorAbsLineshape[0]
            lineshape = AcceptorAbsLineshape[1]
            hbaromega_min2 = np.min(hbaromega)
            hbaromega_max2 = np.max(hbaromega)
            self.AcceptorAbsLineshape = fit_Gaussian(hbaromega, lineshape)
        else:
            raise TypeError("'AcceptorAbsLineshape' must be a 2D array with shape (N, 2) or a callable function")

        if callable(ConvolKernel):
            self.ConvolKernel = ConvolKernel
        elif isarray(ConvolKernel) and len(ConvolKernel) == 2:
            dhbaromega = ConvolKernel[0]
            lineshape = ConvolKernel[1]
            self.ConvolKernel = fit_Gaussian(dhbaromega, lineshape)
        elif ConvolKernel == None:
            self.ConvolKernel = ConvolKernel
        else:
            raise TypeError("'ConvolKernel' must be a 2D array with shape (N, 2) or a callable function or None")


        # determine the ranges frequency and proton position for subsequent calculations
        if 'rmin0' in locals() and 'rmin1' in locals() and 'rmin2' in locals():
            rmin = np.min([rmin0, rmin1, rmin2])
        elif 'rmin' in kwargs.keys():
            rmin = kwargs['rmin']
        else:
            rmin = -0.8
        if 'rmax0' in locals() and 'rmax1' in locals() and 'rmax2' in locals():
            rmax = np.max([rmax0, rmax1, rmax2])
        elif 'rmax' in kwargs.keys():
            rmax = kwargs['rmax']
        else:
            rmax = 0.8
        self.rp = np.linspace(rmin, rmax, NGridPot)


        if 'hbaromega_min1' in locals() and 'hbaromega_min2' in locals():
            hbaromega_min = np.min([hbaromega_min1, hbaromega_min2])
        elif 'hbaromega_min' in kwargs.keys():
            hbaromega_min = kwargs['hbaromega_min']
        else:
            hbaromega_min = 0
        if 'hbaromega_max1' in locals() and 'hbaromega_max2' in locals():
            hbaromega_max = np.max([hbaromega_max1, hbaromega_max2])
        elif 'hbaromega_max' in kwargs.keys():
            hbaromega_max = kwargs['hbaromega_max']
        else:
            hbaromega_max = 6
        self.hbaromega = np.linspace(hbaromega_min, hbaromega_max, NGridLineshape)

        self.check_normalization()

        self.Vel = Vel
        self.NStates = NStates

        # Create the matrices and vectors used for calculation
        # Suv: overlap matrix of proton vibrational wave functions associated with the reactant and product states
        # Suw: overlap matrix of proton vibrational wave functions associated with the reactant and ground states
        # Swv: overlap matrix of proton vibrational wave functions associated with the ground and product states
        # Iuv: matrix of spectral overlap integral
        # Pu: Boltzmann distribution of proton states on the reactant side
        # Pw: Boltzmann distribution of proton states on the ground state
        # total rate constants is \sum_u\sum_v Pu*|Suv|^2*Iuv
        self.Suv = np.zeros((NStates, NStates))
        self.Suw = np.zeros((NStates, NStates))
        self.Swv = np.zeros((NStates, NStates))
        self.Iuv = np.zeros((NStates, NStates))
        self.Pu = np.zeros(NStates)
        self.Pw = np.zeros(NStates)

    def check_normalization(self, err=5e-3, renormalize=True):
        """
        Compute the definite integral of line shape functions
        If they are properly normalized this integral should equal to 2\pi
        The function returns three Boolen values, corresponding to the donor emission, acceptor absorption, convolution kernel, respectively
        if renormalize == True, the unnormalized functions will be normalized
        """
        norm_donor_emission = simps(self.DonorEmLineshape(self.hbaromega), self.hbaromega)
        if np.abs(norm_donor_emission - 2*np.pi) >= err:
            normalized_donor_emission = False
            if renormalize == True:
                tmp_func = copy_func(self.DonorEmLineshape)
                scale = 2*np.pi/norm_donor_emission
                def func(x):
                    return tmp_func(x)*scale
                self.DonorEmLineshape = func 
        else:
            normalized_donor_emission = True

        norm_acceptor_absorption = simps(self.AcceptorAbsLineshape(self.hbaromega), self.hbaromega)
        if np.abs(norm_acceptor_absorption - 2*np.pi) >= err:
            normalized_acceptor_absorption = False
            if renormalize == True:
                tmp_func = copy_func(self.AcceptorAbsLineshape)
                scale = 2*np.pi/norm_acceptor_absorption
                def func(x):
                    return tmp_func(x)*scale
                self.AcceptorAbsLineshape = func 
        else:
            normalized_acceptor_absorption = True

        if self.ConvolKernel != None:
            dhbaromega = self.hbaromega - (self.hbaromega[-1]+self.hbaromega[0])/2
            norm_convol = simps(self.ConvolKernel(dhbaromega), dhbaromega)
            if np.abs(norm_convol - 2*np.pi) >= err:
                normalized_convol = False
                if renormalize == True:
                    tmp_func = copy_func(self.ConvolKernel)
                    scale = 2*np.pi/norm_convol
                    def func(x):
                        return tmp_func(x)*scale
                    self.ConvolKernel = func 
            else:
                normalized_convol = True
        else:
            normalized_convol = None

        return normalized_donor_emission, normalized_acceptor_absorption, normalized_convol

    def calc_proton_vibrational_states(self, mass=massH):
        """
        This function calculates the proton vibrational states (energies and wave functions)
        corresponding to GSProtonPot, ReacProtonPot, and ProdProtonPot respectively. 
        The FGH_1D code implemented by Maxim Secor is used
        """
        # calculate proton potentials on a grid, self.rp
        # the FGH code requires atomic units, so the units are converted
        rp_in_Bohr = self.rp*A2Bohr
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid/(ngrid-1)
        E_GS_in_Ha = self.GSProtonPot(self.rp)*eV2Ha
        E_reac_in_Ha = self.ReacProtonPot(self.rp)*eV2Ha
        E_prod_in_Ha = self.ProdProtonPot(self.rp)*eV2Ha

        # calculate the proton vibrational energies and wave fucntions for the ground state
        eigvals_GS, eigvecs_GS = FGH_1D(ngrid, sgrid, E_GS_in_Ha, mass)
        self.GSProtonEnergyLevels = eigvals_GS[:self.NStates]*Ha2eV

        # the output wave functions are normalized such that \sum_i \Psi_i^2 = 1 where i is the index of grid points
        # the correct normalization is that \int \Psi^2 dr = 1
        # the normalized wave functions has unit of A^-1/2
        unnormalized_wfcs_GS = np.transpose(eigvecs_GS)[:self.NStates]
        normalized_wfcs_GS = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_GS])
        self.GSProtonWaveFunctions = normalized_wfcs_GS

        # calculate the proton vibrational energies and wave fucntions for the reactant
        eigvals_reac, eigvecs_reac = FGH_1D(ngrid, sgrid, E_reac_in_Ha, mass)
        self.ReacProtonEnergyLevels = eigvals_reac[:self.NStates]*Ha2eV

        unnormalized_wfcs_reac = np.transpose(eigvecs_reac)[:self.NStates]
        normalized_wfcs_reac = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_reac])
        self.ReacProtonWaveFunctions = normalized_wfcs_reac

        # calculate the proton vibrational energies and wave fucntions for the product
        eigvals_prod, eigvecs_prod = FGH_1D(ngrid, sgrid, E_prod_in_Ha, mass)
        self.ProdProtonEnergyLevels = eigvals_prod[:self.NStates]*Ha2eV

        unnormalized_wfcs_prod = np.transpose(eigvecs_prod)[:self.NStates]
        normalized_wfcs_prod = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_prod])
        self.ProdProtonWaveFunctions = normalized_wfcs_prod

    def calc_ground_state_distributions(self, T=298):
        Boltzmann_factors = np.exp(-self.GSProtonEnergyLevels/kB/T)
        partition_func = np.sum(Boltzmann_factors)
        self.Pw = Boltzmann_factors/partition_func
        return self.Pw


    def calc_reactant_state_distributions(self, T=298):
        Boltzmann_factors = np.exp(-self.ReacProtonEnergyLevels/kB/T)
        partition_func = np.sum(Boltzmann_factors)
        self.Pu = Boltzmann_factors/partition_func
        return self.Pu

    def calc_proton_overlap_matrix(self):
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.Suv[u,v] = simps(self.ReacProtonWaveFunctions[u]*self.ProdProtonWaveFunctions[v], self.rp)
        return self.Suv

    def calc_proton_overlap_matrix_ReacGS(self):
        for u in range(self.NStates):
            for w in range(self.NStates):
                self.Suw[u,w] = simps(self.ReacProtonWaveFunctions[u]*self.GSProtonWaveFunctions[w], self.rp)
        return self.Suw

    def calc_proton_overlap_matrix_GSProd(self):
        for w in range(self.NStates):
            for v in range(self.NStates):
                self.Swv[w,v] = simps(self.GSProtonWaveFunctions[w]*self.ProdProtonWaveFunctions[v], self.rp)
        return self.Swv

    def calc_spectral_overlap_between_states(self, u, v):
        """
        This function calculate the spectral overlap integral between vibronic states Iu and IIv
        Depending whether there is common vibrational modes shared by the molecules, two different methods will be used
        Note that we are integrating over \hbar\omega instead of \omega
        The computed result is actually I_uv/\hbar instead of just I_uv, the missing \hbar need to be multiplied back
        """
        Evib_G0 = self.GSProtonEnergyLevels[0]
        Evib_u = self.ReacProtonEnergyLevels[u]
        Evib_v = self.ProdProtonEnergyLevels[v]
        LDem = self.DonorEmLineshape(self.hbaromega+Evib_G0-Evib_u)
        LAabs = self.AcceptorAbsLineshape(self.hbaromega+Evib_G0-Evib_v)

        if self.ConvolKernel == None:
            return 2*np.pi*hbar*simps(LDem*LAabs, self.hbaromega)
        else:
            LAabs_convoled = []
            for hbaromega1 in self.hbaromega:
                dhbaromega = hbaromega1-self.hbaromega
                K = self.ConvolKernel(dhbaromega)
                LAabs_convoled.append(simps(K*LAabs, self.hbaromega))
            LAabs_convoled = np.array(LAabs_convoled)
            return hbar*simps(LDem*LAabs_convoled, self.hbaromega)

    def calc_spectral_overlap_matrix(self):
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.Iuv[u,v] = self.calc_spectral_overlap_between_states(u, v)
        return self.Iuv

    def calc_kinetic_contribution_matrix(self):
        k0 = 1/(4*np.pi*np.pi*hbar*hbar)*self.Vel*self.Vel
        self.kuv = k0*np.matmul(np.diag(self.Pu), self.Suv*self.Suv*self.Iuv)
        return self.kuv

    def calculate(self, mass=massH, T=298):
        self.calc_proton_vibrational_states(mass)
        self.calc_reactant_state_distributions(T)
        self.calc_proton_overlap_matrix()
        self.calc_spectral_overlap_matrix()
        self.calc_kinetic_contribution_matrix()

        self.k_tot = np.sum(self.kuv)
        return self.k_tot

    def calc_total_spectra(self, mass=massH, T=298):
        self.calc_proton_vibrational_states(mass)
        self.calc_ground_state_distributions(T)
        self.calc_reactant_state_distributions(T)
        self.calc_proton_overlap_matrix_ReacGS()
        self.calc_proton_overlap_matrix_GSProd()

        def TotalDonorEmLineshape(hbaromega):
            total_Dem = 0
            for u in range(self.NStates):
                for w in range(self.NStates):
                    total_Dem += self.Pu[u]*self.Suw[u,w]*self.Suw[u,w]*self.DonorEmLineshape(hbaromega+self.GSProtonEnergyLevels[w]-self.ReacProtonEnergyLevels[u])
            return total_Dem

        def TotalAcceptorAbsLineshape(hbaromega):
            total_Aabs = 0
            for w in range(self.NStates):
                for v in range(self.NStates):
                    total_Aabs += self.Pw[w]*self.Swv[w,v]*self.Swv[w,v]*self.AcceptorAbsLineshape(hbaromega+self.GSProtonEnergyLevels[w]-self.ProdProtonEnergyLevels[v])
            return total_Aabs
        
        return TotalDonorEmLineshape, TotalAcceptorAbsLineshape
    
    def get_ground_proton_states(self):
        """
        returns proton vibrational energy levels and wave functions of the ground state
        """
        return self.GSProtonEnergyLevels, self.GSProtonWaveFunctions

    def get_reactant_proton_states(self):
        """
        returns proton vibrational energy levels and wave functions of the reactant state
        """
        return self.ReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self):
        """
        returns proton vibrational energy levels and wave functions of the product state
        """
        return self.ProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_ground_state_distributions(self):
        return self.Pw

    def get_reactant_state_distributions(self):
        return self.Pu

    def get_proton_overlap_matrix(self):
        return self.Suv

    def get_proton_overlap_matrix_ReacGS(self):
        return self.Suw

    def get_proton_overlap_matrix_GSProd(self):
        return self.Swv

    def get_spectral_overlap_matrix(self):
        return self.Iuv

    def get_kinetic_contribution_matrix(self):
        return self.kuv

    def get_total_rate_constant(self):
        return self.k_tot

