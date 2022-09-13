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
import qiskit_based as qb


nb_qubits = 7

# Create circuit
circ1 = qk.QuantumCircuit(qk.QuantumRegister(nb_qubits, 'regs_1'), name='circ1')
# Trotterised TFIM (defaults used)
circ1 = qb.TFIMandLF(circ1, steps=2)
  

# Append random unitaries and get results obj
circuits = qb.append_random_unitaries(circ1, 
                                      nb_random=10,
                                      seed=42)
# run circuit to get results (default qasm simulator)
results = qb.Simulator().execute(circuits=circuits)

# Find Tr[rho^2] via slicing sets
subsets_entropiesLST = qb.subsets_entropies(results,
                                            start_end_sets=[[0,2], [2,6]])

# Find Tr[rho^2] via explicit indexes (should be same as above - just testing)
subsets_entropiesIDX = qb.subsets_entropies(results,
                                            qubit_index_sets=[[0,1],
                                                              [2,3,4,5]])


