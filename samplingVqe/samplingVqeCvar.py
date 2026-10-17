from qiskit.quantum_info import SparsePauliOp
import numpy as np

'''
Sampling VQE
 
This code computes the expectation value of a diagonal hamiltonian with respect
to the sampled states. 

Input:

counts		: Sampled states of an ansatz
observables	: list of SparsePauli operators
shots 		: number of shots used for sampling
alpha		: aggregation percentage for CVAR

Output:

Expectation value of the observable.

Usage:
from samplingVqeCvar import getExpectationValue

expectationValue = getExpectationValue([sampledCounts], [op], shots=4096, alpha=1.0)

'''


_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)


def _evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> complex:
    
    """Get the expectation value """
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    
    return np.sum(observable.coeffs * _PARITY[reduced])

def aggregation(alpha, measurements):
    
    """Get the aggregation function for CVaR with confidence level ``alpha``."""
    if alpha is None:
        alpha = 1
    elif not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1] but was {alpha}")

    # if alpha is close to 1 we can avoid the sorting
    if np.isclose(alpha, 1):
        
            #print(measurements)
        return sum(probability * value for probability, value in measurements)

    else:

            # sort by values
        sorted_measurements = sorted(measurements, key=lambda x: x[1])


        accumulated_percent = 0  # once alpha is reached, stop
        cvar = 0
        for probability, value in sorted_measurements:
            cvar += value * min(probability, alpha - accumulated_percent)
            accumulated_percent += probability
            if accumulated_percent >= alpha:
                break

        return cvar / alpha
        

def getExpectationValue(counts, observables, shots, alpha):

    evaluations = [{}]
    for i in range(len(observables)):    
        for state, counts_ in counts[i].items():
            evaluations[i][state]=[counts_/shots, _evaluate_sparsepauli(int(state,2), observables[i])]

    results = np.array([aggregation(alpha, evaluated.values()) for evaluated in evaluations])

    return np.real(results[0])
    
