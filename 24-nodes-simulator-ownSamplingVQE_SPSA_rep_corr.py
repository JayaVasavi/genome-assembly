#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:30:57 2025

@author: kalyan
"""


from openfermion.ops import QubitOperator
import numpy as np
from tqdm import tqdm
import pandas as pd
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import networkx as nx
from samplingVqeCvar import getExpectationValue
from repetition_correction import sanity_check, sequence_list, back_to_bitstring, correct_nodes
import pickle
from qiskit.circuit.library import TwoLocal, RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
# from qiskit_algorithms import NumPyMinimumEigensolver
# from qiskit import execute, Aer
# from qiskit.primitives import BackendSampler
# from qiskit.primitives import Sampler
# from qiskit.providers.options import Options
# from qiskit.providers.aer import AerSimulator
# from qiskit_algorithms.optimizers import SLSQP,COBYLA,ADAM, SPSA, QNSPSA,NFT
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from samplingVqeCvar import getExpectationValue

# from qiskit_algorithms import SamplingVQE
from qiskit_aer import AerSimulator

import matplotlib as mpl
import os

# Set TEX environment variable
# os.environ['TEX'] = '/home/himashu/.TinyTeX'

# Add TinyTeX bin directory to PATH
# os.environ['PATH'] = '/home/himashu/.TinyTeX/bin/x86_64-linux:' + os.environ['PATH']

# Configure matplotlib to use LaTeX
# mpl.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'comfortaa'
plt.rcParams['text.color'] = 'k'
plt.rcParams['axes.labelcolor'] = 'k'
# plt.style.use('seaborn')
# plt.style.use('default')

adjacency_matrix = np.array([[	0,	14,	16,	14,	11,	14,	15,	16,	13,	14,	11,	11,	18,	13,	10,	15,	14,	13,	12,	16,	13,	10,	13,	18	],
[	0,	0,	0,	0,	0,	0,	0,	0,	0,	14,	12,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],
[	0,	13,	0,	10,	10,	12,	13,	15,	16,	14,	11,	11,	14,	19,	9,	13,	15,	10,	10,	13,	19,	8,	14,	17	],
[	0,	14,	0,	0,	12,	14,	0,	0,	0,	11,	11,	0,	14,	15,	0,	14,	0,	0,	12,	0,	0,	0,	0,	0	],
[	0,	15,	0,	0,	0,	14,	0,	0,	0,	13,	9,	0,	14,	14,	0,	11,	0,	0,	10,	0,	0,	0,	0,	0	],
[	0,	15,	0,	0,	0,	0,	0,	0,	0,	13,	12,	0,	20,	18,	0,	13,	0,	0,	11,	0,	0,	0,	0,	0	],
[	0,	14,	0,	11,	11,	12,	0,	15,	0,	17,	9,	0,	12,	19,	0,	15,	17,	10,	9,	16,	20,	0,	13,	14	],
[	0,	18,	0,	10,	11,	15,	0,	0,	0,	14,	9,	0,	17,	14,	0,	10,	14,	0,	13,	0,	19,	0,	0,	0	],
[	0,	12,	0,	11,	13,	13,	11,	15,	0,	13,	12,	12,	20,	17,	11,	12,	14,	11,	11,	13,	16,	15,	14,	17	],
[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],
[	0,	0,	0,	0,	0,	0,	0,	0,	0,	12,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],
[	0,	13,	0,	11,	13,	13,	12,	19,	0,	16,	11,	0,	16,	13,	14,	12,	15,	12,	11,	12,	17,	13,	13,	16	],
[	0,	14,	0,	0,	0,	0,	0,	0,	0,	17,	13,	0,	0,	13,	0,	14,	0,	0,	12,	0,	0,	0,	0,	0	],
[	0,	12,	0,	0,	0,	0,	0,	0,	0,	10,	12,	0,	0,	0,	0,	0,	0,	0,	13,	0,	0,	0,	0,	0	],
[	0,	12,	0,	10,	11,	11,	15,	15,	0,	13,	11,	0,	15,	15,	0,	12,	14,	11,	11,	14,	18,	13,	14,	18	],
[	0,	13,	0,	0,	0,	0,	0,	0,	0,	14,	15,	0,	0,	19,	0,	0,	0,	0,	12,	0,	0,	0,	0,	0	],
[	0,	13,	0,	14,	12,	11,	0,	0,	0,	10,	10,	0,	15,	11,	0,	12,	0,	0,	9,	0,	18,	0,	0,	0	],
[	0,	13,	0,	13,	12,	11,	0,	14,	0,	11,	9,	0,	14,	14,	0,	11,	12,	0,	8,	0,	17,	0,	0,	17	],
[	0,	12,	0,	0,	0,	0,	0,	0,	0,	17,	12,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],
[	0,	12,	0,	11,	14,	13,	0,	17,	0,	13,	16,	0,	11,	10,	0,	11,	14,	13,	12,	0,	21,	0,	0,	16	],
[	0,	11,	0,	12,	12,	12,	0,	0,	0,	14,	12,	0,	14,	13,	0,	15,	0,	0,	10,	0,	0,	0,	0,	0	],
[	0,	12,	0,	10,	15,	13,	14,	17,	0,	12,	12,	0,	17,	13,	0,	12,	14,	14,	11,	16,	19,	0,	14,	17	],
[	0,	19,	0,	15,	16,	16,	0,	18,	0,	16,	10,	0,	14,	17,	0,	13,	13,	6,	15,	19,	21,	0,	0,	18	],
[	0,	18,	0,	11,	13,	14,	0,	16,	0,	13,	9,	0,	13,	17,	0,	11,	16,	0,	10,	0,	18,	0,	0,	0	]],dtype = float)

# Create a directed graph from the adjacency matrix
G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# Draw the graph
plt.figure(figsize=(6, 6))
pos = nx.circular_layout(G)  # positions for all nodes
nx.draw_networkx(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', arrowsize=20)
plt.title('Directed Graph from Adjacency Matrix')
# plt.savefig('graph.jpg')
plt.show()

W = -10*np.array(adjacency_matrix,dtype = float)
num_pos = W.shape[0]
α = 1*np.max(abs(W))/10

for i in range(len(W)):
    for j in range(len(W)):
        if W[i,j] == 0:
            W[i,j] = α # + np.random.uniform()


node = W.shape[0]
node_qubit = int(np.ceil(np.log2(node)))
num_qubits = node_qubit*num_pos
print('Number of qubits for nodes :', node_qubit)
print('Number of positions : ',num_pos)
print('Number of qubits required:', num_qubits)

print(pd.DataFrame(W))

def P0(i):
    iden = QubitOperator('')
    Z = QubitOperator('Z' + str(i))
    return 0.5*(iden + Z)


def P1(i):
    iden = QubitOperator('')
    Z = QubitOperator('Z' + str(i))
    return 0.5*(iden - Z)


def number_to_fixed_length_binary_array(number, length):
    binary_string = bin(number)[2:]  # Get binary representation as a string, removing the '0b' prefix
    padded_binary_string = binary_string.zfill(length)  # Pad with leading zeros to ensure fixed length
    binary_array = np.array([int(bit) for bit in padded_binary_string])  # Convert binary string to numpy array
    return binary_array

def occupation_op(num,pos,N):
    
    # num : node number 
    # N : total qubit for node
    # pos : position runs from 0 to edge - 1

    Ait = QubitOperator('')
    
    binary_array = number_to_fixed_length_binary_array(num, N)
    Nbit = len(binary_array)
    for i in range(Nbit):
        if binary_array[i] == 1:
            Ait = Ait*P1(i + pos*Nbit)
        else:
            Ait = Ait*P0(i + pos*Nbit)
   
    return Ait 


bit = []

for ti in (range(num_pos)):
    bi = []
    for i in range(node):
        bi.append(occupation_op(i,ti,node_qubit))
    bit.append(bi)
    
# Visit Once 

identity = QubitOperator('')

H1 = 0 

A2 = 0.6*np.max(abs(W))

for i in range(node):
    term = 0
    for t in range(num_pos):
        term += bit[t][i]
    term_sq = (identity-term)*(identity-term)
    H1 += term_sq

# Visit once version 02

# H1 = 0 

# for t in range(num_pos):
#     for tdash in range(t+1,num_pos):
#         for i in range(node):
#             H1 += 2*bit[t][i]*bit[tdash][i] - (1/(node-1))*(bit[t][i] + bit[tdash][i] + 2)


# including weights 

H2 = 0

for t in tqdm(range(num_pos - 1)):
    for i in range(node):
        for j in range(node):
            H2 += W[i,j]*(bit[t][i]*bit[t+1][j])

B = 1

H = A2*H1 + B*H2    

H_op = H.terms

print('Constructing Hamiltonian...Completed.')

print('Initiating VQE..')

op = []

for key, value in H_op.items():
    strlist = ['I' for i in range(num_qubits)]  
    for item in key:
        strlist[item[0]] = 'Z'
    op_string = ''.join(strlist)
    op.append((op_string,value))

# op.pop(0)

print(pd.DataFrame(op))

Hop = SparsePauliOp.from_list(op)

## search for source node : 

for i in range(adjacency_matrix.shape[0]):
    x = np.sum(adjacency_matrix[:,i])
    if x == 0:
        print('source point -- ', i)
        source_node = i
        break

## end node :

for i in range(adjacency_matrix.shape[0]):
    x = np.sum(adjacency_matrix[i,:])
    if x == 0:
        print('end point -- ', i)
        end_node = i
        break
    
print('source node : ', source_node,' & end node : ', end_node)

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, RealAmplitudes, EfficientSU2

import time
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
# from qiskit_algorithms.optimizers import COBYLA
# from qiskit_algorithms import NumPyMinimumEigensolver
# from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
#from qiskit_aer.primitives import Sampler, Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Session
from scipy.optimize import minimize
from qiskit.result import sampled_expectation_value
from qiskit_algorithms.optimizers import SPSA, ESCH
from qiskit_aer import AerSimulator

ref = QuantumCircuit(num_qubits)

source_bit = number_to_fixed_length_binary_array(source_node,node_qubit)
end_bit = number_to_fixed_length_binary_array(end_node,node_qubit)
print(source_bit, end_bit)

# for i in range(node_qubit):
#     if source_bit[i] == 1:
#         ref.x(num_qubits-1-i) 
#     if end_bit[i] == 1:
#         ref.x(node_qubit-i-1)

# ref.draw()

ansatz = QuantumCircuit(num_qubits)
## With start and end nodes
for i in range(num_pos-2):
    ansatz.append(EfficientSU2(node_qubit,su2_gates=["rx", "ry"], reps=1, parameter_prefix='φ' + str(i) , insert_barriers=True).decompose(), range(node_qubit + i*node_qubit,node_qubit + (i+1)*node_qubit))

# Efficient SU2 throughout
# ansatz.append(EfficientSU2(num_qubits-2*node_qubit,su2_gates=["rx", "ry"], reps=1, insert_barriers=True).decompose(), range(node_qubit, num_qubits - node_qubit))
    
## No start and end nodes
# for i in range(num_pos):
#     ansatz.append(EfficientSU2(node_qubit,su2_gates=["rx", "ry"], reps=1, parameter_prefix='φ' + str(i) , insert_barriers=True).decompose(), range(i*node_qubit,(i+1)*node_qubit))

# ansatz = ref.compose(ansatz)
ansatz.measure_all()
# ansatz.decompose().draw('mpl')

Hop = SparsePauliOp.from_list(op)

cost_history_dict = {
    # "prev_vector": None,
    "iters": 0,
    "cost_history": [],
    "job_counts": [],
    # "job_id": [],
}

def plot_energy(counts, values):
    fig, ax = plt.subplots(figsize = (6,4))
    ax.plot(counts, values, marker='o', linestyle='-',color = '#C06C84',markerfacecolor='#355C7D',markersize = 5)
    plt.tick_params(top=True, left=True, right=True, bottom=True,direction="in",axis='both', which='both', labelsize=15)
    font2 = {'family':'comfortaa','color':'black','size':18}
    # ax.set_xlabel('Iteration',fontdict = font2)
    ax.set_xlabel('Iteration')
    # ax.set_ylabel('Energy',fontdict = font2)
    ax.set_ylabel('Energy')
    # plt.title('Energy vs Iteration')
    ax.grid(zorder=0)
    plt.show()

counts = []
values = []
params = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=(ansatz.num_parameters))
ans = ansatz.decompose()
# correct_iters = 5
shots = 1000

def cost_func(params):#, ansatz, hamiltonian, simul):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    job = simul.run(circuits=ans.assign_parameters(params),shots=shots)
    result = job.result()
    samp_counts = result.get_counts() 
    # energy = getExpectationValue([samp_counts], [Hop], shots=4096, alpha=0.5)

    corrected_samp_counts = {}
    samp_counts_copy = samp_counts.copy()
    # sanity check for each bitstring
    for bitstring in samp_counts.keys():
        bitstring_count = samp_counts_copy.pop(bitstring)
        seq = sequence_list(node,bitstring)
    # retain bitstring if things are okay
        sumseq, summ2 = sanity_check(seq)
        if sumseq == summ2:
            corrected_samp_counts[bitstring]= bitstring_count
    # correct bitstring if the check is off
        elif sumseq != summ2:
            corrected_seq = correct_nodes(seq)
            corrected_bitstring = back_to_bitstring(corrected_seq)
            corrected_samp_counts[corrected_bitstring]= bitstring_count
    energy = getExpectationValue([corrected_samp_counts], [Hop],shots=shots,alpha=0.5)
    cost_history_dict["job_counts"].append(list(corrected_samp_counts.keys()))
    cost_history_dict["iters"] += 1
    # cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    # cost_history_dict["job_counts"].append(list(samp_counts.keys()))
    #print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")
    clear_output(wait=True)
    if cost_history_dict["iters"]%100 == 0:
        # plot_energy(np.arange(0,cost_history_dict["iters"],1), cost_history_dict["cost_history"])
        print(cost_history_dict["iters"])
        
    if cost_history_dict["iters"]%1000 == 0:
        # with open('costHist24nodeSim.pkl', 'wb') as f:
        #     pickle.dump(cost_history_dict['job_counts'], f)
        plot_energy(np.arange(0,cost_history_dict["iters"],1), cost_history_dict["cost_history"])
        

    return energy

simul = AerSimulator(method="matrix_product_state")
# simul.matrix_product_state_max_bond_dimension = 4

optimizer = SPSA(maxiter=25000)
res = optimizer.minimize(fun=cost_func, x0=params, bounds=[-2*np.pi, 2*np.pi])

# import copy
## Local and global
# spsa = SPSA(maxiter = 15000)
# gl_opt = ESCH(max_evals = 1000)

# for ii in range(5):
### 1
# res1 = spsa.minimize(fun=cost_func, x0 = params1, bounds=[-2*np.pi, 2*np.pi])
# params2 = copy.deepcopy(res1.x)
# res2 = gl_opt.minimize(fun=cost_func, x0 = params2)
# params1 = res2.x

# res = spsa.minimize(fun=cost_func, x0 = params1, bounds=[-2*np.pi, 2*np.pi])

allbitS = []
for ii in range(len(cost_history_dict['job_counts'])):
    for jj in range(len(cost_history_dict['job_counts'][ii])):
        allbitS.append(cost_history_dict['job_counts'][ii][jj])
        
def violations(val):

    vio = 0
    num_seq = []

    for i in range(num_pos):
        num_seq.append(int(val[i*node_qubit:node_qubit*i+node_qubit],2))

    for i in range(num_pos-1):
        if ((num_seq[i] <= node-1) and (num_seq[i+1] <= node-1)):
            if adjacency_matrix[num_seq[i],num_seq[i+1]] == 0:
                vio +=1
                #print(num_seq[i],num_seq[i+1])
        else: vio += 100


    return vio,num_seq

#vio,num_seq = violations(val)

viols_ = []
#numseq_ = []
possibleSol = []
for ii in range(len(allbitS)):
    
    vio_,numseq = violations(allbitS[ii])
    if len(np.unique(numseq)) == len(numseq):
        viols_.append(vio_)
        possibleSol.append(allbitS[ii])

vio,num_seq = violations(possibleSol[np.argmin(np.array(viols_))])
print('No of violations ',vio)
print('Sequence ',num_seq)


# with open('costHist24nodeSim.pkl', 'wb') as f:
#     pickle.dump(cost_history_dict, f)

def list_of_all_connected_nodes(adjacency_mat:np.ndarray) -> list:
   
    """
    Takes adjacency matrix as input and 
    returns list of all connected nodes corresponding to every edge
    Useful to assess solutions
    """
    
    connected_nodes = [[i,j] for i in range(0,num_pos) for j in range (0,num_pos)
                        if adjacency_matrix[i][j]!=0]
    return connected_nodes

def connection_nodes(node:int, adjacency_mat:np.ndarray) -> list:

    """
    Takes a node and adjacency matrix as the input
    and returns list of nodes that have edges connecting them
    """

    connection_nodess = [j for j in range(0,num_pos) if adjacency_matrix[node][j]!=0]
    return connection_nodess

def violations(sol,adjacency_mat):

    """
    Takes solution and the adjacency matrix as inputs
    Returns list of nodes where violations are found
    """

    violation_sites = []
    for i in range(0,len(sol)-1):
        if sol[i+1] not in connection_nodes(sol[i],adjacency_mat):
            violation_sites.append(sol[i])
            print('Violation at Node',sol[i],'(Position',i,
                  '). Edge',[sol[i],sol[i+1]], 
                  'is not among', [[sol[i],j] for j in connection_nodes(sol[i],adjacency_mat)])
    return violation_sites

def repetitions(sol):

    """
    Takes solution candidate as the input
    Returns a list of the nodes in the solution that are repeated along the path 
    and their frequency
    """

    sol_set = list(set(sol))
    counts = [[i,sol.count(i)] for i in sol_set if sol.count(i)>1]
    [print('Node',counts[i][0],'repeated',counts[i][1],'times.') for i in range(0,len(counts))]
    return counts

def check_missing_nodes(sol,adjacency_mat):

    """
    Checks if there are any nodes missing in the path
    Takes the solution candidate and adjacency matrix as the input
    Returns a list of the missing nodes if they exist, else None 
    """

    missing_nodes = [i for i in range(0,len(adjacency_mat)) if i not in sol]
    if missing_nodes == []:
        print('No missing nodes in Solution')
        return None
    else:
        print('Solution missing node(s)',missing_nodes)
        return missing_nodes

def solution_quality_check(sol,adjacency_mat):

    """
    Checks the quality of solution by evaluating possible violations, repetitions,
    and missing nodes
    Takes solution candidate and adjacency matrix as the input
    Returns a dictionary with stats of violations, repetitions, and missing nodes
    """

    violation_counts = violations(sol,adjacency_mat)
    repetition_counts = repetitions(sol)
    missing_nodes = check_missing_nodes(sol,adjacency_mat)
    answer = {
            'violation_counts':[len(violation_counts),violation_counts],
            'repetition_counts':repetition_counts,
            'missing_nodes': missing_nodes
            }
    return answer

solution_quality_check(num_seq,adjacency_matrix)
# list(raw_result.optimal_parameters.values())

#ideal solution
# 0 -> 2 -> 8 -> 11 -> 14 -> 21 -> 6 -> 22 -> 19 -> 17 -> 23 -> 7 -> 16 -> 20 -> 3 -> 4 -> 5 -> 12 -> 15 -> 13 -> 18 -> 1 -> 10 -> 9

bits_min2 = []
for ii in range(len(viols_)):
    if viols_[ii] == 2:
        bits_min2.append(ii)
shot_2, _ = np.histogram(bits_min2, bins = 100)


# bits_min3 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 3:
#         bits_min3.append(ii)
# shot_3, _ = np.histogram(bits_min3, bins = 100)


# bits_min4 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 4:
#         bits_min4.append(ii)
# shot_4, _ = np.histogram(bits_min4, bins = 100)  

# bits_min5 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 5:
#         bits_min5.append(ii)
# shot_5, _ = np.histogram(bits_min5, bins = 100) 

# bits_min6 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 6:
#         bits_min6.append(ii)
# shot_6, _ = np.histogram(bits_min6, bins = 100) 

# bits_min7 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 7:
#         bits_min7.append(ii)
# shot_7, _ = np.histogram(bits_min7, bins = 100) 

# bits_min8 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 8:
#         bits_min8.append(ii)
# shot_8, _ = np.histogram(bits_min8, bins = 100) 

# bits_min9 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 9:
#         bits_min9.append(ii)
# shot_9, _ = np.histogram(bits_min9, bins = 100) 

# bits_min10 = []
# for ii in range(len(viols_)):
#     if viols_[ii] == 10:
#         bits_min10.append(ii)
# shot_10, _ = np.histogram(bits_min10, bins = 100) 


# shot_2n = shot_2/np.max(shot_2)  
# shot_3n = shot_3/np.max(shot_3)  
# shot_4n = shot_4/np.max(shot_4)  
# shot_5n = shot_5/np.max(shot_5)
# shot_6n = shot_6/np.max(shot_6)
# shot_7n = shot_7/np.max(shot_7)
# shot_8n = shot_8/np.max(shot_8)
# shot_9n = shot_9/np.max(shot_9)
# shot_10n = shot_10/np.max(shot_10)  

# plt.scatter(np.arange(100), shot_2n, color = 'r', marker = '*', label = '2 vio')
# plt.scatter(np.arange(100), shot_3n, color = 'b', label = '3 vio')
# plt.scatter(np.arange(100), shot_4n, color = 'g', label = '4 vio')
# plt.scatter(np.arange(100), shot_5n, color = 'r', label = '5 vio')
# plt.scatter(np.arange(100), shot_6n, color = 'c', label = '6 vio')
# plt.scatter(np.arange(100), shot_7n, color = 'm', label = '7 vio')
# plt.scatter(np.arange(100), shot_8n, color = 'y', label = '8 vio')
# plt.scatter(np.arange(100), shot_9n, color = 'k', label = '9 vio')
# plt.scatter(np.arange(100), shot_10n, color = 'b', marker = '*', label = '10 vio')
# plt.grid(True)
# plt.legend()
# plt.show()


