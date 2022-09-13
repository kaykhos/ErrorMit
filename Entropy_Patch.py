# Here we define basic building blocks to define a general Pauli spin chain Hamiltonian and initialise a
# quantum state. This file is to be called upon in other code to make the calculation of non-equilibrium
# dynamics of spin chains simple.


#############################################################################################################
############################################### Packages ####################################################
#############################################################################################################

import math

import numpy as np
from numpy import linalg as LA

import scipy.linalg
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import itertools

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#############################################################################################################
########################################## Hamiltonian terms ################################################
#############################################################################################################

# Here we define the terms that are used in our Hamiltonian, namely, Pauli matricies and Ising interaction.
# Other Pauli interactions can be defined in a similar manner but are omitted for simplicity of the code.

# Define basic Pauli matrix
X = csr_matrix([[0,1],[1,0]])
Y = csr_matrix([[0,-1j],[1j,0]])
Z = csr_matrix([[1,0],[0,-1]])

# Define Pauli Matrix on given site in a spin chain system - `N' is the spin chain length and `site' dictates what site the interaction will act on.

def PauliX(N,site):
    tempMat = sp.kron(sp.identity(2**site),X)
    tempMat = sp.kron(tempMat,sp.identity(2**(N-site-1)))
    return tempMat

def PauliY(N,site):
    tempMat = sp.kron(sp.identity(2**site),Y)
    tempMat = sp.kron(tempMat,sp.identity(2**(N-site-1)))
    return tempMat

def PauliZ(N,site):
    tempMat = sp.kron(sp.identity(2**site),Z)
    tempMat = sp.kron(tempMat,sp.identity(2**(N-site-1)))
    return tempMat

# Define Ising interactions - `N' is the spin chain length and `site' dictates what site the interaction will act on,
# i.e. the interaction is between site and site + 1

# We can define interactions such as XY, XZ etc or long-range interactions in a very similar manner but the code
# becomes repetitive.

def IsingX(N,site):
    tempMat = sp.kron(sp.identity(2**site),X)
    tempMat = sp.kron(tempMat,X)
    tempMat = sp.kron(tempMat,sp.identity(2**(N-site-2)))
    return tempMat

def IsingY(N,site):
    tempMat = sp.kron(sp.identity(2**site),Y)
    tempMat = sp.kron(tempMat,Y)
    tempMat = sp.kron(tempMat,sp.identity(2**(N-site-2)))
    return tempMat

def IsingZ(N,site):
    tempMat = sp.kron(sp.identity(2**site),Z)
    tempMat = sp.kron(tempMat,Z)
    tempMat = sp.kron(tempMat,sp.identity(2**(N-site-2)))
    return tempMat

#############################################################################################################
############################################### Hamiltonian #################################################
#############################################################################################################

# Here we define the spin chain Hamiltonian class, `N' is the spin chain length`.
# We will use functions to make it easy to add terms, calculate energies and calculate the evolution opterator.

class SpinChainHamilonian:
    def __init__(self, N):
        # Define the length of the spin chain system and create a base Hamiltonian of zeros.
        self.length = N
        self.hamiltonian = csr_matrix((2**N,2**N))

    def add_term(self,term,strength):
        # This function makes adding a term to te Hamiltonian simple.
        # `term' is one of the above defined Pauli functionsmand `strength' is the coefficient of this term.
        self.hamiltonian = self.hamiltonian + strength*term

    def calculate_energies(self):
        # This function will calculate the eigensytem of the Hamiltonian.
        E, V = LA.eigh(self.hamiltonian.todense())
        self.energies = E
        self.vectors = V

    def evolution_operator(self,t):
        # This function will calculate the evolution operator of the Hamiltonian for a given time, `t'.`.
        return scipy.linalg.expm(-1j*self.hamiltonian.todense()*t)

#############################################################################################################
############################################# Subspace Basis ################################################
#############################################################################################################

def subspace_basis(N,patch):

    if patch[-1]-patch[0] != len(patch)-1:
        print('patch error')
        quit()

    # This function will calculate the basis of a subsystem in the spin chain, this is used to calculate Renyi entropy.
    Na = len(patch)
    tempBasisList = [np.reshape(np.array(i), Na) for i in itertools.product([0, 1], repeat = Na)]
    BasisList = []
    temp=[]

    for i in range(len(tempBasisList)):
        state = 1
        for ii in range(Na):
            if tempBasisList[i][ii] == 0:
                state = sp.kron(state,[[1],[0]])
            if tempBasisList[i][ii] == 1:
                state = sp.kron(state,[[0],[1]])
        if patch[0] > 0 and patch[-1] < N-1:
            state = sp.kron(sp.identity(2**(patch[0])),state)
            BasisList.append(sp.kron(state,sp.identity(2**(N-patch[-1]-1))))
        elif patch[0] == 0 and patch[-1] < N-1:
            BasisList.append(sp.kron(sp.identity(2**(N-patch[-1]-1)),state))
        elif patch[0] > 0 and patch[-1] == N-1:
            BasisList.append(sp.kron(state,sp.identity(2**(patch[0]))))
        elif patch[0] == 0 and patch[-1] == N-1:
            BasisList.append(state)
    return BasisList

#############################################################################################################
############################################### State vector ################################################
#############################################################################################################

# Here we create the state vector class, for simplicity of the code, we have just included a class that
# initialises a state polarized in the Z direction, this can easily be genralised.

# `N' is the spin chain length, `state' corresponds to a list of 1's and 0's that correspond to the state
# of each spin - e.g. [up, up, down, down, up, down] == [0, 0, 1, 1, 0, 1].`

class SpinChainInitailState_Z:
    def __init__(self, N, state):
        if len(state) != N:
            print('Initial state size error!')
            exit()
        # Define a state vector of length `N' with polarisation given by `state'.`
        self.length = N

        full_initial_state = 1
        for site in range(N):
            if state[site] == 0:
                full_initial_state = np.kron(full_initial_state,[[1],[0]])
            if state[site] == 1:
                full_initial_state = np.kron(full_initial_state,[[0],[1]])
        self.vector = full_initial_state

        # Calculate the density matrix.
        self.density_matrix = np.outer(self.vector,self.vector)

    def time_evolve(self,U):
        # This function time evolves the state vector and density matrix for a given evolution operator `U'.
        self.vector = U.dot(self.vector)
        self.density_matrix = U.conj().T.dot(self.density_matrix.dot(U))

    def measure_localmagnetisation(self, polarisation, site):
        # This function measures the local magnetisation in the axis of `polarisation' of site `site'.
        if polarisation == 'X':
            return self.vector.conj().T.dot(PauliX(self.length,site).dot(self.vector))[0,0]
        elif polarisation == 'Y':
            return self.vector.conj().T.dot(PauliY(self.length,site).dot(self.vector))[0,0]
        elif polarisation == 'Z':
            return self.vector.conj().T.dot(PauliZ(self.length,site).dot(self.vector))[0,0]

    def measure_correlation(self, polarisation, site1, site2):
        # This function measures the two site correlation function in the axis of `polarisation' of sites `site1' and `site2'.
        if polarisation == 'X':
            return self.vector.conj().T.dot(PauliX(self.length,site1).dot(PauliX(self.length,site2).dot(self.vector)))[0,0] - self.vector.conj().T.dot(PauliX(self.length,site1).dot(self.vector))[0,0]*self.vector.conj().T.dot(PauliX(self.length,site2).dot(self.vector))[0,0]
        elif polarisation == 'Y':
            return self.vector.conj().T.dot(PauliY(self.length,site1).dot(PauliY(self.length,site2).dot(self.vector)))[0,0] - self.vector.conj().T.dot(PauliY(self.length,site1).dot(self.vector))[0,0]*self.vector.conj().T.dot(PauliY(self.length,site2).dot(self.vector))[0,0]
        elif polarisation == 'Z':
            return self.vector.conj().T.dot(PauliZ(self.length,site1).dot(PauliZ(self.length,site2).dot(self.vector)))[0,0] - self.vector.conj().T.dot(PauliZ(self.length,site1).dot(self.vector))[0,0]*self.vector.conj().T.dot(PauliZ(self.length,site2).dot(self.vector))[0,0]

    def measure_entropy(self, Na, BasisList):
        # This function measures the Renyi entropy of the subsystem in the system spanned by `BasisList'.

        partial_density_matrix = np.zeros((2**(self.length-Na),2**(self.length-Na)),dtype='complex')

        for i in range(len(BasisList)):
            partial_density_matrix += np.dot(BasisList[i].T.toarray(),np.dot(self.density_matrix,BasisList[i].toarray()))

        D,V = np.linalg.eigh(partial_density_matrix)
        return np.sum(np.square(np.abs(D)))



#############################################################################################################
############################################# Purities ######################################################
#############################################################################################################

# A function to calculate the purity dynamics given a chain length, N, an Ising coupling strngth, J, a
# transverese field strength, hx, a max time, t_max and a set of patches

def patch_entropies(N,J,hx,t_max,Patches):

    state = [0,0,0,0,0,0,0,0]

    # define the patches we are measuring.
    Patch_Basis = []
    for patch in Patches:
        Patch_Basis.append(subspace_basis(N,patch))

    # Times we will consider.
    dt = 0.1
    times = np.arange(0,t_max+dt,dt)

    # Define Hamiltonian.
    H = SpinChainHamilonian(N)

    # Add Hamiltonian terms
    for n in range(N-1):
        H.add_term(IsingZ(N,n),J)
    for n in range(N):
        H.add_term(PauliX(N,n),hx)

    # Calculate the evolution opterators for each Hamiltonian for a time step of `dt'.
    U = H.evolution_operator(dt)

    # Prepare the initial states.
    V = SpinChainInitailState_Z(N,state)

    # Create a numpy array of zeros to store results in.
    results = np.zeros((len(Patches),len(times)), dtype = 'complex')

    # For each time step, calculate the local magnetisation and then evolve the states.
    for i in range(len(times)):
        ii=0
        for patch_basis in Patch_Basis:
            results[ii,i] = V.measure_entropy(len(Patches[ii]),patch_basis)
            ii+=1

        V.time_evolve(U)

    return times, np.real(results).T
