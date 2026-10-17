

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
