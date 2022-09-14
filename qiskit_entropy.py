#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/04/2020

@author: kiran

Functions work on a single anzatz circuit and append single qubit Haar
    random unitaries at the end. 
    NEEDED: Have to add functionality to compare only subset of measurements!
    + Can deal with two of the same size circuits running on singe device
    + append random seeded unitaries to list of circs
    + prefix1 and prefix2 now spesify circ names bewteen results objects
    
    - Might rename prefix1/2 if it makes sense
"""


import qiskit as qk
import numpy as np
import scipy as sp
import copy
import pdb
pi = np.pi


#%% testing

# REFACTOR NOTES



# Annoying to set this, but I don't want to hard code anything
NB_QUBITS_HAVE_DIM_2 = 2
NB_DEFAULT_RANDOM_CIRCUITS = 5
NB_DEFAULT_SEED = 10
STR_DEFAULT_PREFIX = 'Haar_Random'



#%% Appending random unitaries to circuit
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def append_random_unitaries(circs, 
                            nb_random=NB_DEFAULT_RANDOM_CIRCUITS,
                            seed=NB_DEFAULT_SEED,
                            qregs_blocks=0, # to be replaced by names
                            prefix=STR_DEFAULT_PREFIX):
    """ 
    Appends Haar-random unitaries to each circuit given. 
    
    Parameters
    ---------------
    circs: 
        Single qiskit circuit, or list of quiskit circuits to append the random 
        unitaries to
        
    nb_random: default 5
        The number of random unitaries to append per circuit. 
        
    seed: default 10
        Random number seed (every circuit in the list uses this seed)
        
    qregs_blocks: default 0 (currently not functioning)
        Name (string) or quantum register block (int) to append the random 
        measurement to
        
    prefix: default 'Haar_Random'
        A string to identify which circuits in the results object correspond 
        to the random unitaries. (Only nessassary to provide if sending several 
        several cross_fidelity circuits in the same quantum job). 
    """
    if type(circs) != list:
        circs = [circs]
    circs = _unique_circ_name(circs)
    circ_list = []
    for circ in circs:
        circ_list += _append_random_unitaries_single_circ(circ,
                                                          nb_random=nb_random,
                                                          seed=seed,
                                                          qregs_blocks=qregs_blocks,
                                                          prefix=prefix)
    return circ_list


def _unique_circ_name(circs):
    """ Returns a new list with the same circuits, but with uniquie names.
        Input list names are unchanged. 
        
    Parameters
    -----------
    circs:
        List of qiskit circuits
    """
    circ_names = []
    circ_list = []
    ct = 0
    for circ in circs:
        this_circ = copy.deepcopy(circ)
        if this_circ.name in circ_names:
            this_circ.name = this_circ.name + '_' + str(ct)
            ct+=1
        else:
            circ_names.append(this_circ.name)
        circ_list.append(this_circ)
    return circ_list
    





def _append_random_unitaries_single_circ(circ, 
                       nb_random=NB_DEFAULT_RANDOM_CIRCUITS, 
                       seed=NB_DEFAULT_SEED,
                       qregs_blocks=0, # will replace this with name
                       prefix=STR_DEFAULT_PREFIX):
    """ Creates a list of n=nb_random circuits with Haar random unitaries. If
        mutiple qregs_blocks are passed, each block has the same unitaries
        - DOES NOT modify input circuit
        TO DO: Allow passing of qregs names in the list"""
    # input formatting
    if type(qregs_blocks) == int:
        qregs_blocks = [qregs_blocks]
    np.random.seed(seed=seed)
    
    circ_list = []
    # Run over different numbers of circuits
    for ii in range(nb_random):
        # Fix circuit and set random seed (so each block has SAME unitaries)
        this_circ = copy.deepcopy(circ)
        this_circ.name = prefix + '_' +  str(ii) + '_' + this_circ.name
        nb_qubits = this_circ.qregs[qregs_blocks[0]].size
        seeds = np.random.randint(0, 2**32, nb_qubits)
        for qregs_block in qregs_blocks:
            this_circ = _random_measurement_helper(this_circ, 
                                                   qregs_block=qregs_block, 
                                                    seeds=seeds)
        circ_list.append(this_circ)
    return circ_list


def _random_measurement_helper(circ, 
                               seeds, 
                               qregs_block=0):
    """Appends (seeded) random unitaries for a circuit
        - Will modify input circ if the input circ is passes as object
        - Not made to be interacted with directly 
        - Adds measurement registers with same name as qregs_block
        - Assumes circuit construction is created with single /mutiple blocks"""
    # get the registers to measure and add classical bits
    qregs = circ.qregs[qregs_block]
    nb_qubits = qregs.size
    cbits = qk.ClassicalRegister(nb_qubits, 'cl_'+qregs.name)
    circ.add_register(cbits)

    
    # generate Haar random (with known seeds)
   # if type(seeds) is not list:
   #     seeds = np.random.randint(0, 2**32, nb_qubits)
    u_random = [qk.quantum_info.random_unitary(2, seed=seeds[ii]) 
                for ii in range(nb_qubits)]
    
    # append unitaries and measure to right classical registers
    for ii in range(nb_qubits):
        circ.append(u_random[ii], [qregs[ii]])
    circ.measure(qregs, cbits)
    return circ



#%% Implimenting cross fidelity between measurement results
    # Handels results.to_dict() inputs OR direct result objctes
    # In Single obj is passed, user MUSH spesify which jobs to look at via the 
    #   unitary_blocks lists
    # Unfortinuately nb_qubits is nessassary (for now) to deal with the bullshit hex keys
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def cross_fidelity(results_in,
                   prefix1 = STR_DEFAULT_PREFIX, 
                   prefix2 = STR_DEFAULT_PREFIX,
                   unitary_block1=None,
                   unitary_block2=None):
    """ Accepts as input results_in is a results object, a results.to_dict()
        or a list of EXACTLY two results objs in either format
        
        TO DO: add ability to pass miltiple blocks in 
        TO DO: add ability to pass list of measurement outcomes for spesific unitary"""
    results1, results2 = _gen_results_to_compare(results_in,                   
                                                 prefix1 = prefix1, 
                                                 prefix2 = prefix2,
                                                 unitary_block1=unitary_block1,
                                                 unitary_block2=unitary_block2)    
    cross_overlap = density_matrix_overlap(results1, results2)
    first_overlap = density_matrix_overlap(results1, results1)
    secon_overlap = density_matrix_overlap(results2, results2)
    F = cross_overlap / max(first_overlap, secon_overlap)
    return F


def density_matrix_overlap(results1, results2): # d=2 for qubit, 3 for qutrit
    """Impliments Eq. 2 from Cross-Platform Verification of Intermediate Scale 
        Quantum Devices
        Basic checks at the beginning to ensure inputs are valid (not exhaustive)
        Assumes inputs are LISTS OF DICTIONARIES
        """
    # very basic checks on inputs
    nb_qubits1 = len(list(results1[0].keys())[0])
    nb_qubits2 = len(list(results2[0].keys())[0])
    nb_u1 = len(results1)
    nb_u2 = len(results2)
    if nb_qubits1 == nb_qubits2 and nb_u1 == nb_u2:
        nb_qubits = nb_qubits1
    else:
        assert False, 'Error - results dont match dims for this method'
    
    # Generae all possible keys may, or may not have measurement realisation
    keys1 = _gen_keys_from_lists(results1)
    keys2 = _gen_keys_from_lists(results2)
    keys = list(set.intersection(keys1, keys2))
    nb_hilbert_dims = NB_QUBITS_HAVE_DIM_2**nb_qubits
    
    # Double sum in Eq. 2, coefficient times cross corrslation paper
    Trace = 0
    for k1 in keys:
        for k2 in keys: 
            hamming_distance = int(nb_qubits*sp.spatial.distance.hamming(list(k1), list(k2)))
            coeff = nb_hilbert_dims * (-NB_QUBITS_HAVE_DIM_2)**(-hamming_distance)
            Trace += coeff * _correlation(results1, results2, k1, k2)
            
    return Trace


def _correlation(results1, results2, key1, key2):
    """Computs the corelation between at two measurement outcomes, assumes all inputs
        all contribute to correlation (really Prob(key1)*Prob(key2) over nb_unitaries"""
    # get number of random U matricies
    if type(results1) != list: results1 = _gen_measurement_list(results1)
    if type(results2) != list: results2 = _gen_measurement_list(results2)

    nb_u = len(results1)
    correlation = 0
    norm = sum(results1[0].values()) *  sum(results2[0].values())
    for ii in range(nb_u): # for each unitary
        correlation += results1[ii].get(key1, 0) * results2[ii].get(key2, 0)
        
        # Get counts and keys, and number of shots (robust to key errors)
        #keys1 = list(results1[ii].keys())
        #keys2 = list(results2[ii].keys())
        # basic error handeling if not all measurements are realized
        #if key1 in keys1 and key2 in keys2: 
        #    correlation += results1[ii][key1] *results2[ii][key2]
    
    # Normalize ensemble mean to number of input shots, and number of unitariess
    correlation = correlation / nb_u / norm
    return correlation


# -----------------------------------------------------------------------------
# --------- B/S functions that deal with results.to_dict() keys----------------
# -----------------------------------------------------------------------------
    
def _gen_results_to_compare(results_in,
                            prefix1=STR_DEFAULT_PREFIX,
                            prefix2=STR_DEFAULT_PREFIX,
                            unitary_block1=None, 
                            unitary_block2=None):
    """ Helper function inputs results object OR dictionary, and returns a subset
        of measurement results corresponding to the unitary blocks
        - results_in can be two different objects or one single object
        - if results in is a single object, unitary blocks must be spesified
        """
    if type(results_in) == list: # if comparing two objects
        results1 = results_in[0]
        results2 = results_in[1]
        
        results1_list = _gen_measurement_list(results1)
        results2_list = _gen_measurement_list(results2)
        
        # Either return unitary block list IF GIVEN otherwise by name
        if unitary_block1 != None:
            results1_list = [results1_list[ii] for ii in unitary_block1]
        else:
            results1_names = _gen_measurement_names(results1)
            results1_list = _match_names_and_prefix(prefix1, results1_names, results1_list)
                    
                    
        if unitary_block2 != None:
            results2_list = [results2_list[ii] for ii in unitary_block2]
        else:
            results2_names = _gen_measurement_names(results2)
            results2_list = _match_names_and_prefix(prefix2, results2_names, results2_list)
    else:
        # assert unitary_block1 != None, " Unitary blocks MUST be spesified for a single input"
        # assert unitary_block2 != None, " Unitary blocks MUST be spesified for a single input"
        temp_res = _gen_measurement_list(results_in)

        if unitary_block1 != None and unitary_block2 != None:
            results1_list = [temp_res[ii] for ii in unitary_block1]
            results2_list = [temp_res[ii] for ii in unitary_block2]
        else:
            assert prefix1 != prefix2, "For a single obj input, prefixes must be different"
            temp_res_names = _gen_measurement_names(results_in)
            results1_list = _match_names_and_prefix(prefix1, temp_res_names, temp_res)
            results2_list = _match_names_and_prefix(prefix2, temp_res_names, temp_res)
    return results1_list, results2_list
    

def _gen_measurement_list(results_in):
    """ Returns a LIST OF DICTIONARIES from results object/dict (all resutls)"""
    if type(results_in) == dict:
        results_in = qk.result.result.Result.from_dict(results_in)
    ls = []
    for ii in range(len(results_in.results)):
        ls.append(results_in.get_counts(ii))
    return ls


def _gen_measurement_names(results_in):
    """ Returns a list of Circuit NAMES for each circ in result """
    if type(results_in) == qk.result.result.Result:
        results_in = results_in.to_dict()
    ls = []
    for ii in range(len(results_in['results'])):
        ls.append(results_in['results'][ii]['header']['name'])
    return ls


def _match_names_and_prefix(prefix, circ_names, results_list):
    """ Matches circuit names to prefixes """
    ls = []
    for ii in range(len(results_list)):
        if prefix in circ_names[ii]:
            ls.append(results_list[ii])
    return ls



def _gen_keys_from_lists(list_in):
    keys = set({})
    for rr in list_in:
        keys = keys.union(set(rr.keys()))
    return keys



#%% added to find entanglement entropy on subsets

def _p_trace_results(results_in,
                     start_end=[0,1],
                     qubit_index=None):
    """
    Performs a partial trace over the measurement results

    Parameters
    ----------
    results_in : qiskit results object (unitary trace sections)
    start_end : Start and end point for linear chain of qubits (used if qubit index is None)
        The default is [0,1].
    qubit_index : Qubit indexes to use (for more general use)

    Returns
    -------
    reduced_counts : 
        List of count dicts for reduced string.

    """
    reduced_counts = []
    for di in results_in.get_counts():
        exp_di = {}
        for full_meas in di.keys():
            if qubit_index:
                part_meas = np.array(list(full_meas))[qubit_index]
                part_meas = ''.join(part_meas)
            else:
                part_meas = full_meas[start_end[0]:start_end[1]]
            exp_di[part_meas] = di[full_meas] + exp_di.get(part_meas,0)
        reduced_counts.append(exp_di)
    return reduced_counts
            
            
            
def subsets_entropies(results1,
                    start_end_sets=None,
                    qubit_index_sets=None):
    """
    Computes the entanglement+noise entropy on each subset. 
    Spesify one of start_end_sets (for 1D-like indexing), 
    OR
    qubit_index_sets for more general lists of indexes/subsets to use

    Parameters
    ----------
    results1 : Results object (from quantum device).
    start_end_sets : List of pairs of [[start0,end0], [start1,end1]...]
    qubit_index_sets : List of qubit subsets [[q0-0,q0-1...q0-n],
                                              [q1-0,q1-1...q1-m]...]
        over which to measure the entropies

    Returns
    -------
    tr_rho2 : tr[rho^2] for each subset given

    """
    if not qubit_index_sets and not start_end_sets:
        raise ValueError('spesify ONE of qubit_index_sets OR start_end_sets')
    if qubit_index_sets:
        start_end_sets = [None]*len(qubit_index_sets)
    else:
        qubit_index_sets = [None]*len(start_end_sets)
    
    tr_rho2 = []
    for start_end, qubit_index in zip(start_end_sets, qubit_index_sets):
        reduced_counts = _p_trace_results(results1, 
                                          start_end=start_end,
                                          qubit_index=qubit_index)
        tr_rho2.append(density_matrix_overlap(reduced_counts, reduced_counts))
    return tr_rho2
    




#%% From joe (TFIM qiskit circuit)



def TFIMandLF(circuit,
              hx = 0,
              hz = 0,
              J = 1,
              steps = 5,
              time = 1,
              dt = None,
              L = None,
              qr = None,
              bitflip = None):
    """
    Parameters
    ----------
    circuit : Quantum circuit to operate on
        DESCRIPTION.
    hx : X-field
        DESCRIPTION. The default is 0.5.
    hz : Z-field
        DESCRIPTION. The default is 0.5.
    J : ZZ rate
        DESCRIPTION. The default is 1.
    steps : total steps (trotter)
        DESCRIPTION. The default is 5.
    time : total time
        DESCRIPTION. The default is 1.
    dt : can spesift, else dt = time/steps
        DESCRIPTION. The default is None.
    L : can spesify number of qubits (else taken from circuit obj)
        DESCRIPTION. The default is None.
    qr : Can spesify named set of quantum registers (else taken from circuit)
        DESCRIPTION. The default is None.

    Returns
    -------
    circuit update (acts in place anyway)

    """
    if time and dt:
        raise ValueError('cannot have both time and dt set')
    
    if bitflip:
        bitflip = np.atleast_1d(bitflip)
        for qubit in bitflip:
            circuit.x(qubit)
    if not L:
        L = circuit.num_qubits
    if not qr:
        qr = circuit.qregs[0]
    if not dt:
        dt = time / steps


    sites = range(L)
    order = []
    for i in sites:
        if i % 2 == 0 and i < L-1:
            order.append(i)
    for i in sites:
        if i % 2 != 0 and i and i < L-1:
            order.append(i)

    for j in range(steps):
        for i in sites:
            circuit.rx(hx*dt,qr[i])
            circuit.rz(2*hz*dt,qr[i])

        for k in order:
            circuit.rzz(2*J*dt,qr[k],qr[k+1])

        for i in sites:
            circuit.rx(hx*dt,qr[i])
    return circuit


def Simulator(name = 'qasm_simulator',
              shots = 2**13,
              optimization_level = 0,):
    """ Returns a quantum instance simulator"""
    if 'qasm' in name:
        backend = qk.Aer.get_backend(name=name)
    elif 'state' in name:
        from qiskit.providers.aer import StatevectorSimulator
        backend = StatevectorSimulator(precision='single')
    instance = qk.utils.QuantumInstance(backend=backend, 
                                        shots=shots, 
                                        optimization_level=optimization_level)
    return instance



def patch_entropies(N,
                    Patches,
                    J = 1,
                    hx = 0.5,
                    t_max = 1,
                    steps = 10,
                    trotter_steps = 5,
                    nb_random = 20,
                    seed = 42,
                    shots=2**10,
                    bitflip=None):
    """
    Wrapper to be similar to Joe's scipy_entropy.patch_entropies code

    Parameters
    ----------
    N : qubit count
    Patches : patch indexes
    J : ZZ for ising chain
        DESCRIPTION. The default is 1.
    hx : X_field
        DESCRIPTION. The default is 0.5.
    t_max : max time (for output plots)
        DESCRIPTION. The default is 1.
    steps : steps for the final plots
        DESCRIPTION. The default is 10.
    trotter_steps : for the qiskit circuit
        DESCRIPTION. The default is 5.
    nb_random : Harr random circuit samples
        DESCRIPTION. The default is 20.
    seed : Harr random see
        DESCRIPTION. The default is 42.

    tVec, entropies (for each patch)
    -------
    """
    tVec = [0]
    entropy_array = [[1]*len(Patches)]
    for ct in range(steps):
        time = t_max / steps * (ct + 1)
        tVec.append(time)
        
        # Make circuit
        qc = qk.QuantumCircuit(qk.QuantumRegister(N, 'regs_1'), name='circ1')
        qc = TFIMandLF(qc, 
                       steps=trotter_steps,
                       J=J,
                       hx=hx,
                       hz=0,
                       time=time,
                       bitflip=bitflip)
        # Add Harr random circs + simulate results
        circuits = append_random_unitaries(qc,
                                           nb_random=nb_random,
                                           seed=seed)
        results = Simulator(shots=shots).execute(circuits=circuits)
        # Calc entropy patches
        subsets_entropiesIDX = subsets_entropies(results,
                                                 qubit_index_sets=Patches)
        entropy_array.append(subsets_entropiesIDX)
    return np.array(tVec), np.array(entropy_array)
        

#%% using state vector to purity inc. trotter error

def _exact_circuit_purity_helper(circuit, Patches,
                       include = True):
    """
    Uses statevector to calculate reduced DM for a given circuit

    Parameters
    ----------
    circuit : TYPE
    Patches : TYPE
    include : (include or exclude patches)

    Returns
    -------
    entropy_array : tr[rho^2] for each patch

    """
    if include:
        nb_qubits = circuit.num_qubits
        complete = set(range(nb_qubits))
        trace_over = []
        for ct in range(len(Patches)):
            trace_over.append(list(complete.difference(set(Patches[ct]))))
    else:
        trace_over = Patches
    
    
    sv = qk.quantum_info.DensityMatrix(circuit)
    entropy_array = []
    for pp in trace_over:
        rho = np.array(qk.quantum_info.partial_trace(sv, pp))
        entropy_array.append(np.trace(rho@ rho).real)
    return entropy_array


def circuit_patch_entropies(N,
                           Patches,
                           J = 1,
                           hx = 0,
                           t_max = 1,
                           steps = 10,
                           trotter_steps = 5,
                           bitflip = None):
    """
    Trotterised DM purities for given patches

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    Patches : TYPE
        DESCRIPTION.
    J : TYPE, optional
        DESCRIPTION. The default is 1.
    hx : TYPE, optional
        DESCRIPTION. The default is 0.5.
    t_max : TYPE, optional
        DESCRIPTION. The default is 1.
    steps : TYPE, optional
        DESCRIPTION. The default is 10.
    trotter_steps : TYPE, optional
        DESCRIPTION. The default is 5.
    nb_random : TYPE, optional
        DESCRIPTION. The default is 20.
    seed : TYPE, optional
        DESCRIPTION. The default is 42.

    Returns
    -------
    Array of patches
    """
    tVec = [0]
    er = [[1]*len(Patches)]
    for ct in range(steps):
        time = t_max / steps * (ct + 1)
        tVec.append(time)
        
        # Make circuit
        qc = qk.QuantumCircuit(qk.QuantumRegister(N, 'regs_1'), name='circ1')
        qc = TFIMandLF(qc, 
                       steps=trotter_steps,
                       J=J,
                       hx=hx,
                       hz=0,
                       time=time,
                       bitflip=bitflip)
        tmp = _exact_circuit_purity_helper(qc, 
                                           Patches=Patches,
                                           include=True)
        er.append(tmp)
    
    return np.array(tVec), np.array(er)
