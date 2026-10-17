# genome-assembly
Repo for storing the quantum algorithm solutions for the genome assembly problem

The problem is posed as a Hamiltonian path search on an overlap graph of sequencing
reads, and solved with CVaR SamplingVQE on Qiskit simulators and IBM Quantum hardware.

On this branch:
15-nodes-hamiltonian_genome__.ipynb - main worked example, 15-node graph.
samplingVqeCvar.py - CVaR expectation value of a diagonal Hamiltonian from sampled counts.
repetition_correction/ - detects and repairs repeated nodes in a candidate sequence.

Branches:
main - core 15-node example, CVaR SamplingVQE, repetition correction.
classical_pre_postprocessing - classical pipeline around the solver: read de-duplication,
    overlap graph construction, contig generation, adjacency matrices, circuit analytics,
    and a small example/ input set.
data - adjacency matrices for the published cases, plus circuit analytics (2-qubit gate
    counts and depths on FakeTorino and Fake
