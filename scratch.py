#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:40:43 2022

@author: kiran
"""

import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
import qiskit_entropy as qe
import scipy_entropy as se 



#%% Basic use of qiskit_entropy

nb_qubits = 7

# Create circuit
circ1 = qk.QuantumCircuit(qk.QuantumRegister(nb_qubits, 'regs_1'), name='circ1')
# Trotterised TFIM (defaults used)
circ1 = qe.TFIMandLF(circ1, steps=2, time=0)
  

# Append random unitaries and get results obj
circuits = qe.append_random_unitaries(circ1, 
                                      nb_random=20,
                                      seed=42)
# run circuit to get results (default qasm simulator)
results = qe.Simulator().execute(circuits=circuits)

# Find Tr[rho^2] via slicing sets
start_end_sets = [[0,3], 
                  [1,4],
                  [2,5],
                  [3,6],
                  [4,7]]
subsets_entropiesLST = qe.subsets_entropies(results,
                                            start_end_sets=start_end_sets)

# Find Tr[rho^2] via explicit indexes (should be same as above - just testing)
qubit_index_sets = [list(range(tt[0], tt[1])) for tt in start_end_sets]

subsets_entropiesIDX = qe.subsets_entropies(results,
                                            qubit_index_sets=qubit_index_sets)

# Plotting stuff
f, ax = plt.subplots(1,2)
ax[0].plot(np.atleast_2d(subsets_entropiesLST), 'd')
ax[0].legend(start_end_sets)
ax[0].set_title('Tr[rho^2] of sub-patches')

circ1.draw(output = 'mpl', ax = ax[1])
ax[1].set_title('TFIM circ (time = 0)')

#%% basic use of scipy_entropy



# Super simple code to calculate the purity dynamics of the transverese field Ising model.

N = 7 #Chain length
J = -1 #Ising coupling strength
hx = -0.5 #transverese field strength
t_max = 5 #max time
Patches = [list(range(tt[0], tt[1])) for tt in start_end_sets]




#%% Plot as above but using entropy sampling....

tVec, entropies = qe.patch_entropies(N,
                                     Patches=Patches,
                                     J=J, 
                                     hx=hx,
                                     t_max=t_max,
                                     steps=15,
                                     trotter_steps=20,
                                     nb_random=400,
                                     seed=42)
plt.plot(tVec, entropies)
plt.legend(Patches)
plt.title('Measuered purity on patches')
plt.show()


#%% Reduced DM from circuit

tVec, entropies5 = qe.circuit_patch_entropies(N,
                                             Patches,
                                             J=J,
                                             hx=hx,
                                             t_max=t_max,
                                             steps=15,
                                             trotter_steps=7)
tVec, entropies15 = qe.circuit_patch_entropies(N,
                                             Patches,
                                             J=J,
                                             hx=hx,
                                             t_max=t_max,
                                             steps=15,
                                             trotter_steps=20)
times, results = se.patch_entropies(N=N,
                                    Patches=Patches,
                                    t_max=t_max,
                                    steps=30,
                                    J=J,
                                    hx=hx)   # returns the times and the purities for the patches.

plt.plot(times, results, alpha = 0.75)
plt.plot(tVec, entropies5, 'd', alpha = 0.5)
plt.plot(tVec, entropies15, '.', alpha = 0.5)
plt.plot(tVec, entropies, '*', alpha = 1)

plt.legend(Patches)
plt.title('Measuered purity on patches (7-20 Trotters)')
plt.show()
