#!/usr/bin/env python3
"""
Circuit Analytics for Genome Assembly Problem
Analyzes circuits (with/without start-end nodes, custom ansatz)
and reports 2-qubit gate counts and depths on FakeTorino and FakeFez
with optimization levels 1, 2, and 3.
"""

import numpy as np
import networkx as nx
import json
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeFez
from qiskit import generate_preset_pass_manager
from collections import defaultdict


# Adjacency matrices from few_genome_adj_matrices.ipynb
CASES = {
    "case1_bacillus": {
        "nodes": 4,
        "matrix": np.array([
            [0., 19., 0., 0.],
            [18., 0., 19., 0.],
            [0., 19., 0., 18.],
            [18., 18., 0., 0.]
        ])
    },
    "case2_genome": {
        "nodes": 11,
        "matrix": np.array([
            [0., 5., 3., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 5., 4., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 6., 4., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 3., 0., 3., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 6., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 6., 4., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 3.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.],
            [0., 0., 0., 3., 0., 3., 0., 0., 0., 0., 0.]
        ])
    },
    "case3_bacteriodes": {
        "nodes": 7,
        "matrix": np.array([
            [0., 63., 67., 65., 67., 64., 0.],
            [0., 0., 60., 61., 60., 62., 0.],
            [0., 0., 0., 62., 65., 65., 61.],
            [0., 0., 0., 0., 65., 60., 0.],
            [0., 0., 0., 0., 0., 61., 0.],
            [0., 0., 0., 0., 0., 0., 63.],
            [0., 0., 0., 0., 0., 0., 0.]
        ])
    },
    "case4_avian_flu": {
        "nodes": 18,
        "matrix": np.array([
            [0., 24., 0., 0., 0., 0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 22., 25., 21., 0., 0., 0., 0., 22., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 24., 0., 0., 0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 22., 0., 0., 0., 0., 0., 23., 0., 0., 0., 0., 0., 21., 0.],
            [0., 0., 0., 0., 0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 21., 0., 0., 0., 25., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 25., 25., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 26., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 21., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 21., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 21., 21., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 21., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 22., 22., 22.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 24., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 22.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])
    },
    "case5_denv2": {
        "nodes": 6,
        "matrix": np.array([
            [0., 23., 18., 18., 24., 0.],
            [0., 0., 22., 16., 24., 22.],
            [0., 0., 20., 18., 20., 22.],
            [0., 0., 0., 20., 25., 0.],
            [0., 0., 0., 16., 22., 0.],
            [0., 17., 24., 21., 0., 0.]
        ])
    },
    "case6_monkeypox": {
        "nodes": 9,
        "matrix": np.array([
            [0., 44., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 26., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 44., 0., 0., 16., 0., 0.],
            [0., 0., 0., 0., 21., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 22., 15., 16., 0.],
            [0., 0., 0., 0., 16., 0., 21., 0., 0.],
            [0., 0., 0., 19., 0., 18., 20., 21., 0.],
            [0., 0., 0., 0., 15., 16., 0., 0., 34.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])
    },
    "case7_african_swine_21": {
        "nodes": 21,
        "matrix": np.array([
            [0., 16., 12., 13., 14., 14., 16., 18., 14., 16., 14., 13., 14., 18., 15., 13., 15., 13., 0., 0., 0.],
            [0., 0., 12., 13., 0., 12., 0., 16., 13., 17., 13., 0., 14., 0., 0., 21., 0., 0., 16., 0., 14.],
            [0., 0., 0., 0., 0., 12., 0., 0., 0., 0., 17., 0., 0., 0., 0., 0., 0., 0., 12., 0., 0.],
            [0., 0., 0., 0., 13., 13., 0., 17., 0., 14., 0., 0., 12., 14., 0., 17., 0., 0., 0., 0., 12.],
            [0., 0., 12., 0., 0., 14., 0., 0., 14., 0., 0., 0., 0., 14., 14., 0., 0., 0., 0., 0., 12.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 14., 0., 0., 0., 0., 0., 0., 0., 12., 0., 0.],
            [0., 13., 0., 0., 0., 13., 0., 17., 12., 15., 14., 16., 15., 14., 13., 19., 13., 14., 0., 0., 0.],
            [0., 0., 0., 0., 0., 18., 0., 0., 14., 16., 13., 0., 16., 13., 0., 18., 0., 0., 0., 0., 13.],
            [0., 0., 0., 0., 0., 15., 0., 0., 0., 0., 13., 0., 0., 20., 13., 0., 0., 0., 12., 0., 0.],
            [0., 0., 13., 0., 0., 18., 0., 0., 15., 0., 14., 0., 14., 17., 0., 19., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 13., 0., 0., 0., 12., 0., 17., 13., 15., 13., 0., 14., 20., 12., 16., 0., 14., 12., 15., 13.],
            [0., 0., 0., 0., 14., 13., 0., 0., 0., 0., 0., 0., 0., 15., 12., 18., 0., 0., 0., 0., 12.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 12., 0., 0., 13., 0., 0., 0., 0., 14., 0., 0., 0., 0., 0., 0., 0., 15., 0., 0.],
            [0., 0., 0., 0., 12., 0., 0., 0., 12., 0., 14., 0., 0., 14., 15., 0., 0., 0., 12., 0., 12.],
            [0., 16., 0., 0., 0., 14., 0., 14., 12., 15., 17., 0., 17., 12., 15., 20., 0., 13., 0., 0., 0.],
            [0., 19., 15., 0., 15., 19., 0., 18., 16., 18., 16., 0., 13., 14., 13., 21., 0., 0., 0., 0., 16.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 16., 0., 14., 0., 12., 0., 17., 13., 17., 12., 0., 14., 17., 12., 19., 14., 14., 12., 0., 15.],
            [0., 0., 0., 0., 0., 15., 0., 0., 14., 0., 13., 0., 0., 14., 0., 0., 0., 0., 0., 0., 0.]
        ])
    },
    "case8_african_swine_24": {
        "nodes": 24,
        "matrix": np.array([
            [0., 14., 16., 14., 11., 14., 15., 16., 13., 14., 11., 11., 18., 13., 10., 15., 14., 13., 12., 16., 13., 10., 13., 18.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 14., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 13., 0., 10., 10., 12., 13., 15., 16., 14., 11., 11., 14., 19., 9., 13., 15., 10., 10., 13., 19., 8., 14., 17.],
            [0., 14., 0., 0., 12., 14., 0., 0., 0., 11., 11., 0., 14., 15., 0., 14., 0., 0., 12., 0., 0., 0., 0., 0.],
            [0., 15., 0., 0., 0., 14., 0., 0., 0., 13., 9., 0., 14., 14., 0., 11., 0., 0., 10., 0., 0., 0., 0., 0.],
            [0., 15., 0., 0., 0., 0., 0., 0., 0., 13., 12., 0., 20., 18., 0., 13., 0., 0., 11., 0., 0., 0., 0., 0.],
            [0., 14., 0., 11., 11., 12., 0., 15., 0., 17., 9., 0., 12., 19., 0., 15., 17., 10., 9., 16., 20., 0., 13., 14.],
            [0., 18., 0., 10., 11., 15., 0., 0., 0., 14., 9., 0., 17., 14., 0., 10., 14., 0., 13., 0., 19., 0., 0., 0.],
            [0., 12., 0., 11., 13., 13., 11., 15., 0., 13., 12., 12., 20., 17., 11., 12., 14., 11., 11., 13., 16., 15., 14., 17.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 13., 0., 11., 13., 13., 12., 19., 0., 16., 11., 0., 16., 13., 14., 12., 15., 12., 11., 12., 17., 13., 13., 16.],
            [0., 14., 0., 0., 0., 0., 0., 0., 0., 17., 13., 0., 0., 13., 0., 14., 0., 0., 12., 0., 0., 0., 0., 0.],
            [0., 12., 0., 0., 0., 0., 0., 0., 0., 10., 12., 0., 0., 0., 0., 0., 0., 0., 13., 0., 0., 0., 0., 0.],
            [0., 12., 0., 10., 11., 11., 15., 15., 0., 13., 11., 0., 15., 15., 0., 12., 14., 11., 11., 14., 18., 13., 14., 18.],
            [0., 13., 0., 0., 0., 0., 0., 0., 0., 14., 15., 0., 0., 19., 0., 0., 0., 0., 12., 0., 0., 0., 0., 0.],
            [0., 13., 0., 14., 12., 11., 0., 0., 0., 10., 10., 0., 15., 11., 0., 12., 0., 0., 9., 0., 18., 0., 0., 0.],
            [0., 13., 0., 13., 12., 11., 0., 14., 0., 11., 9., 0., 14., 14., 0., 11., 12., 0., 8., 0., 17., 0., 0., 17.],
            [0., 12., 0., 0., 0., 0., 0., 0., 0., 17., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 12., 0., 11., 14., 13., 0., 17., 0., 13., 16., 0., 11., 10., 0., 11., 14., 13., 12., 0., 21., 0., 0., 16.],
            [0., 11., 0., 12., 12., 12., 0., 0., 0., 14., 12., 0., 14., 13., 0., 15., 0., 0., 10., 0., 0., 0., 0., 0.],
            [0., 12., 0., 10., 15., 13., 14., 17., 0., 12., 12., 0., 17., 13., 0., 12., 14., 14., 11., 16., 19., 0., 14., 17.],
            [0., 19., 0., 15., 16., 16., 0., 18., 0., 16., 10., 0., 14., 17., 0., 13., 13., 6., 15., 19., 21., 0., 0., 18.],
            [0., 18., 0., 11., 13., 14., 0., 16., 0., 13., 9., 0., 13., 17., 0., 11., 16., 0., 10., 0., 18., 0., 0., 0.]
        ])
    },
    "case9_hantavirus_31": {
        "nodes": 31,
        "matrix": np.array([
            [0., 19., 20., 19., 19., 22., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 19., 20., 0., 21., 0., 21., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 25., 0., 0., 0., 0., 24., 22., 23., 25., 25., 24., 25., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 20., 0., 0., 19., 0., 0.],
            [0., 0., 0., 0., 23., 24., 0., 0., 19., 19., 20., 19., 21., 20., 20., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 20., 0., 22., 19., 20., 19., 0., 19., 0., 0., 0., 0., 0., 0., 23., 0., 0., 0., 0., 0., 0., 0., 22., 19., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 20., 20., 19., 22., 21., 22., 21., 21., 0., 0., 0., 0., 19., 0., 0., 0., 0., 0., 19., 0., 21., 21., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 19., 20., 21., 20., 20., 21., 21., 21., 20., 21., 28., 21., 21., 19., 21., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 19., 0., 0., 0., 0., 0., 0., 0., 0., 22., 24., 25., 25., 21., 24., 25., 26., 24., 21., 20., 25., 0., 0., 20., 22.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 21., 20., 22., 21., 0., 0., 0., 22., 22., 23., 24., 23., 22., 20., 21., 0., 0., 0., 19., 21.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 19., 19., 19., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 20., 0., 19., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 0., 0., 0., 21., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 22., 23., 23., 0., 0., 0., 0., 0., 0., 0., 0., 0., 22., 19., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 22., 22., 0., 0., 0., 0., 0., 0., 0., 0., 0., 23., 20., 0., 0., 20., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 22., 0., 0., 0., 0., 0., 0., 0., 0., 0., 23., 20., 0., 0., 20., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 24., 21., 0., 0., 23., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 19., 22., 23., 22., 23., 22., 22., 0., 0., 0., 0., 21., 0., 23., 0., 0., 24., 27., 20., 21., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 21., 20., 22., 21., 0., 0., 23., 22., 22., 23., 24., 23., 22., 20., 21., 0., 0., 0., 19., 21.],
            [0., 0., 0., 0., 0., 0., 0., 0., 22., 21., 20., 22., 22., 24., 22., 0., 0., 0., 19., 22., 19., 20., 19., 19., 20., 21., 0., 0., 0., 21., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 20., 20., 22., 21., 22., 22., 0., 0., 0., 0., 21., 22., 23., 22., 21., 20., 21., 0., 0., 0., 19., 20.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 21., 20., 22., 21., 0., 0., 0., 0., 0., 23., 24., 23., 22., 20., 21., 0., 0., 0., 0., 21.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 19., 19., 0., 0., 0., 0., 0., 0., 26., 25., 24., 0., 20., 0., 0., 0., 0., 23.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 21., 20., 21., 21., 0., 0., 0., 0., 0., 0., 0., 23., 22., 20., 21., 0., 0., 0., 0., 21.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 20., 21., 20., 22., 21., 0., 0., 0., 0., 0., 0., 0., 0., 22., 20., 21., 0., 0., 0., 0., 21.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 0., 20., 20., 20., 20., 0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 20., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 24., 0., 0., 65., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 65., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 24., 25., 24., 24., 23., 21., 23., 0., 0., 0., 0., 0., 0., 0., 0., 0., 23., 24., 0., 23., 65., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 21., 21., 21., 0., 21., 0., 19., 21., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 21., 20., 22., 21., 0., 0., 0., 0., 22., 23., 24., 23., 22., 20., 21., 0., 0., 0., 0., 21.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 19., 19., 21., 20., 22., 21., 0., 0., 0., 0., 0., 0., 0., 0., 22., 20., 21., 0., 0., 0., 0., 0.]
        ])
    }
}


def get_source_and_end_nodes(adjacency_matrix):
    """Find source and end nodes from adjacency matrix"""
    source_node = None
    end_node = None

    for i in range(adjacency_matrix.shape[0]):
        if np.sum(adjacency_matrix[:, i]) == 0:
            source_node = i
            break

    for i in range(adjacency_matrix.shape[0]):
        if np.sum(adjacency_matrix[i, :]) == 0:
            end_node = i
            break

    if source_node is None:
        source_node = 0
    if end_node is None:
        end_node = adjacency_matrix.shape[0] - 1

    return source_node, end_node


def number_to_fixed_length_binary_array(n, length):
    """Convert number to fixed-length binary array"""
    return np.array([int(b) for b in format(n, f'0{length}b')], dtype=int)


def build_circuits(num_nodes, source_node, end_node):
    """Build three circuit variants: with start/end, without start/end, custom ansatz"""
    node_qubit = int(np.ceil(np.log2(num_nodes)))
    num_qubits = num_nodes * node_qubit
    num_pos = num_nodes

    circuits = {}

    # Reference circuit (source and end node initialization)
    ref = QuantumCircuit(num_qubits)
    source_bit = number_to_fixed_length_binary_array(source_node, node_qubit)
    end_bit = number_to_fixed_length_binary_array(end_node, node_qubit)

    for i in range(node_qubit):
        if source_bit[i] == 1:
            ref.x(num_qubits - 1 - i)
        if end_bit[i] == 1:
            ref.x(node_qubit - i - 1)

    # Circuit 1: With start and end nodes
    ansatz1 = QuantumCircuit(num_qubits)
    for i in range(num_pos - 2):
        ansatz1.append(
            EfficientSU2(node_qubit, su2_gates=["rx", "ry"], reps=1,
                        parameter_prefix='φ' + str(i), insert_barriers=True).decompose(),
            range(node_qubit + i * node_qubit, node_qubit + (i + 1) * node_qubit)
        )
    circ1 = ref.compose(ansatz1)
    circ1.measure_all()
    circuits['with_start_end'] = circ1

    # Circuit 2: Without start and end nodes
    ansatz2 = QuantumCircuit(num_qubits)
    for i in range(num_pos):
        ansatz2.append(
            EfficientSU2(node_qubit, su2_gates=["rx", "ry"], reps=1,
                        parameter_prefix='φ' + str(i), insert_barriers=True).decompose(),
            range(i * node_qubit, (i + 1) * node_qubit)
        )
    circ2 = ansatz2.copy()
    circ2.measure_all()
    circuits['without_start_end'] = circ2

    # Circuit 3: Custom ansatz (Efficient SU2 throughout - middle qubits)
    ansatz3 = QuantumCircuit(num_qubits)
    ansatz3.append(
        EfficientSU2(num_qubits - 2 * node_qubit, su2_gates=["rx", "ry"], reps=1,
                    parameter_prefix='φ', insert_barriers=True).decompose(),
        range(node_qubit, num_qubits - node_qubit)
    )
    circ3 = ref.compose(ansatz3)
    circ3.measure_all()
    circuits['custom_ansatz'] = circ3

    return circuits


def analyze_circuit(circuit, backend, opt_level):
    """Transpile circuit and extract analytics"""
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    transpiled = pm.run(circuit)

    # Count 2-qubit gates
    gate_counts = defaultdict(int)
    for inst in transpiled.data:
        if len(inst.qubits) == 2:
            gate_counts[inst.operation.name] += 1

    total_2q_gates = sum(gate_counts.values())
    depth = transpiled.depth()

    # Compute 2-qubit gate depth
    two_qubit_depth = transpiled.depth(lambda instr: len(instr.qubits) == 2)

    return {
        'total_2q_gates': total_2q_gates,
        'depth': depth,
        'two_qubit_gate_depth': two_qubit_depth,
        'gate_breakdown': dict(gate_counts)
    }


def main():
    """Main analysis function"""
    backends = {
        'FakeTorino': FakeTorino(),
        'FakeFez': FakeFez()
    }

    opt_levels = [1, 2, 3]
    circuit_types = ['with_start_end', 'without_start_end', 'custom_ansatz']

    results = {}

    for case_name, case_data in CASES.items():
        print(f"\n{'='*80}")
        print(f"Analyzing: {case_name} ({case_data['nodes']} nodes)")
        print(f"{'='*80}")

        num_nodes = case_data['nodes']
        adjacency_matrix = case_data['matrix']

        source_node, end_node = get_source_and_end_nodes(adjacency_matrix)
        print(f"Source node: {source_node}, End node: {end_node}")

        circuits = build_circuits(num_nodes, source_node, end_node)
        results[case_name] = {}

        for backend_name, backend in backends.items():
            # Check if circuit fits on backend
            backend_qubits = backend.num_qubits
            node_qubit = int(np.ceil(np.log2(num_nodes)))
            num_qubits = num_nodes * node_qubit

            if num_qubits > backend_qubits:
                print(f"\n  Backend: {backend_name} - SKIPPED (requires {num_qubits} qubits, backend has {backend_qubits})")
                results[case_name][backend_name] = {"error": f"Exceeds backend qubit count: requires {num_qubits}, backend has {backend_qubits}"}
                continue

            print(f"\n  Backend: {backend_name}")
            results[case_name][backend_name] = {}

            for circ_type in circuit_types:
                print(f"    Circuit Type: {circ_type}")
                results[case_name][backend_name][circ_type] = {}

                circuit = circuits[circ_type]

                for opt_level in opt_levels:
                    try:
                        analytics = analyze_circuit(circuit, backend, opt_level)
                        results[case_name][backend_name][circ_type][f'opt_level_{opt_level}'] = analytics

                        print(f"      Opt Level {opt_level}:")
                        print(f"        2-qubit gates: {analytics['total_2q_gates']}")
                        print(f"        2-qubit gate depth: {analytics['two_qubit_gate_depth']}")
                        print(f"        Total depth: {analytics['depth']}")
                        if analytics['gate_breakdown']:
                            print(f"        Gate breakdown: {analytics['gate_breakdown']}")
                    except Exception as e:
                        print(f"      Opt Level {opt_level}: ERROR - {str(e)}")
                        results[case_name][backend_name][circ_type][f'opt_level_{opt_level}'] = {"error": str(e)}

    # Save results to JSON
    with open('circuit_analytics_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Results saved to circuit_analytics_results.json")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    main()
