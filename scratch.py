#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:40:43 2022

@author: kiran
"""

import qiskit as qk
import numpy as np
import scipy as sp
import copy
import pdb
import cross_fidelity as cf


# Set up backend executor
backend = qk.Aer.get_backend('qasm_simulator')
instance = qk.utils.QuantumInstance(backend, 
                                    shots=2**13, 
                                    optimization_level=0)

# Create some circuit (Replace with Trotterised H-model)
nb_qubits = 7
circ1 = qk.QuantumCircuit(qk.QuantumRegister(nb_qubits, 'regs_1'), name='circ1')
for qq in range(nb_qubits):
    circ1.rx(0,qq)
circ1.barrier()
  

# Append random unitaries from the cf function
circuits = cf.append_random_unitaries(circ1, 
                            nb_random=400,
                            seed=42)



results = instance.execute(circuits=circuits)

# Find Tr[rho^2] via slicing sets
subsets_entropiesLST = cf.subsets_entropies(results,
                                         start_end_sets=[[0,2], [2,6]])

# Find Tr[rho^2] via explicit indexes (should be same as above)
subsets_entropiesIDX = cf.subsets_entropies(results,
                                         qubit_index_sets=[[0,1],
                                                           [2,3,4,5]])


