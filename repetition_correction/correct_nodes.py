import  numpy as np
import math
from copy import deepcopy


def binary_list(n,binarystring):
    m = len(bin(n))-2
    b_list = [(binarystring[i*m:(i+1)*m]) for i in range(n)]
    return b_list



def sequence_list(n,binarystring):
    m = len(bin(n))-2
    seq_list = [int(binarystring[i*m:(i+1)*m],2) for i in range(n)]
    return seq_list



def sequence_to_binary(m,sequence_list_):
    seq2bin = [(bin(i)[2:]).zfill(m) for i in sequence_list_]
    return seq2bin



def back_to_binstring(m,seq):
    bin_list = sequence_to_binary(m,seq)
    binstr = '' 
    for i in bin_list:
        binstr += i
    binstr = [binstr[:]] 
    return binstr



def sanity_check(sequence:list):

    """Evaluates input sequence """

    m = len(sequence)
    
    #sum m_2 is sum of the first m numbers
    sum_m2 = int(1/6*m*(m+1)*(2*m+1))

    sum_seq = []

    for i in sequence:
       sum_seq += [(i+1)**2]
   
   #square of every (element+1)^2 and the sum of the entire array
    sum_seq = sum(sum_seq)

   # if no repetitions or missing nodes present, the array should be the sum of the square of first m numbers

    if sum_seq == sum_m2:
        print("Sanity check positive: Thetas correspond to valid nodes. Execute next 50 iterations with the same final thetas ")

    elif sum_seq != sum_m2:
        print("Sanity check negative: Repetitions or non-existant nodes present; Correct thetas before next 50 iterations ")
        
    return (sum_seq,sum_m2)



def check_sequence_anamolies(sequence):

    m = len(sequence)
    seq_A = [i for i in range(0,m)]
    sequence_ = deepcopy(sequence)
    missing_nodes = []
    anamolous_nodes = []

    for i in range(0,m):
        try:
            seq_A.remove(sequence[i])
            sequence_.remove(sequence[i])
        except:
            anamolous_nodes.append(sequence[i])
    
    missing_nodes = seq_A
    
    return missing_nodes, anamolous_nodes


def hamming_weight(bit1,bit2):
    ham_weight = sum([abs(int(bit1[i])-int(bit2[i])) for i in range(len(bit1))])
    return ham_weight


def hamming_weight_matrix(n,missing_nodes,anamolous_nodes):
    m = len(bin(n))-2
    hamming_weight_matrix = np.array([[hamming_weight(i,j) for i in sequence_to_binary(m,missing_nodes)] for j in sequence_to_binary(m,anamolous_nodes)])
    return hamming_weight_matrix


def pick_min_weight_mapping(n,missing_nodes,anamolous_nodes):
    mappings = []
    matrix = hamming_weight_matrix(n,missing_nodes,anamolous_nodes)
    replace_num = len(bin(n))-1 ## this is 1+ maximum possible hamming weight difference for two numbers in missing/anamalous nodes
    for p in range(len(matrix)):
        i,j = np.unravel_index(np.argmin(matrix.flatten()), matrix.shape)
        mappings.append([missing_nodes[i],anamolous_nodes[j]]) 
        for q in range(0,len(matrix)):
            matrix[i][q]=replace_num
            matrix[q][j]=replace_num
        print(matrix)
    return mappings


def correct_nodes(seq):
    #sequence is given as list. if it is binary use sequence_list to convert bitstring to sequence
    n = math.ceil(math.log2(len(seq)))
    missing, anamolies = check_sequence_anamolies(seq)
    mapping = pick_min_weight_mapping(len(seq),missing,anamolies)
    for i in mapping:
        seq[seq.index(i[1])]= i[0]
        # print(seq)
        seq_bin_list = sequence_to_binary(n,seq)
    return seq, seq_bin_list

