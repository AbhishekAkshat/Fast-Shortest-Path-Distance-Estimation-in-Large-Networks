# Import Required Libraries:
import networkx as nx
import random
import math
import metis
import numpy as np
import pandas as pd
import time
#---------------------------- BASIC STRATEGIES --------------------------------#

# Random: Select a set of n nodes uniformly at random from the graph.
def get_random_nodes(G, n=None):
    if n==None:
        nodes = list(nx.nodes(G))
        random.shuffle(nodes)
        return nodes
    else:
        return random.sample(nx.nodes(G), n)

# Degree: Sort the nodes of the graph by decreasing degree and Choose the top n nodes.
def get_high_degree_nodes(G, n=None):
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    degrees = [node[0] for node in degrees]
    if n is None:
        return degrees
    else:
        return degrees[:n]

# Centrality: Select as landmarks the n nodes with the highest betweenness centrality.
def get_high_betweenness_nodes(G, n=None, samples=None):
    if not samples:
        samples = int(len(nx.nodes(G)) ** (1/3))
        print(samples)
    betweenness = nx.betweenness_centrality(G, samples)
    node_info = [(k, v) for  k, v in betweenness.items()]
    node_info = sorted(node_info, key=lambda x:x[1], reverse=True)
    node_info = [node[0] for node in node_info]
    if n is None:
        return node_info
    else:
        return node_info[:n]

# Centrality: Select as landmarks the n nodes with the lowest closeness centrality.
def get_low_closeness_nodes(G, n=None):
    closeness_Centrality = nx.closeness_centrality(G)
    sorted_closeness_Centrality = sorted(closeness_Centrality.items(), key=lambda x:x[1])
    ordered_nodes = [node[0] for node in sorted_closeness_Centrality]
    if n is None:
        return ordered_nodes
    else:
        return ordered_nodes[:n]

#------------------------ CONSTRAINED STRATEGIES -----------------------------#

def constrained_landmark_selection(G, num_landmarks, strategy='degree', samples=50):
    if (strategy=='degree'):
        nodes = get_high_degree_nodes(G)
    elif (strategy=='random'):
        nodes = get_random_nodes(G)
    elif (strategy=='betweenness'):
        nodes = get_high_betweenness_nodes(G, samples=samples)
    elif (strategy=='closeness'):
        nodes = get_low_closeness_nodes(G)
    else:
        print('No valid selection strategy was provided')
        return None
    landmarks = []
    discard = set()
    while len(landmarks)<num_landmarks:
        node = nodes.pop()
        if node in discard:
            continue
        for n in nx.all_neighbors(G, node):
            discard.add(n)
        landmarks.append(node)
    return landmarks

#----------------------- PARTITION BASED STRATEGIES --------------------------#

# Partitioning the Graph using METIS, a graph-partitioning algorithm:

def partition(G, num_landmarks):

    # Convert networkX graph to Metis for Partitioning
    G_undir = G.to_undirected()
    G_metis = metis.networkx_to_metis(G_undir)
    (edgecuts, parts) = metis.part_graph(G_metis, num_landmarks)
    # Create a DataFrame with Nodes, Partition Index, Degree and Centrality
    partition_df = pd.DataFrame(list(zip(list(G.nodes()), parts)), columns = ['Nodes', 'Partition'])

    nodes_Degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    degree_df = pd.DataFrame(nodes_Degree, columns=['Nodes', 'Degree'])
    complete_df = pd.merge(partition_df, degree_df, on='Nodes')

    closeness_Centrality = nx.closeness_centrality(G)
    centrality_df = pd.DataFrame(list(closeness_Centrality.items()), columns = ['Nodes', 'Centrality'])
    complete_df = pd.merge(complete_df, centrality_df, on='Nodes')

    return complete_df

# Degree/P, Pick the node with the highest degree in each partition:

def degree_partitioned(complete_df, num_landmarks):

    landmarks = []
    # Iterate over partitions and Pick the node with the highest degree
    for i in range(num_landmarks):
        df_pt = complete_df.loc[complete_df['Partition'] == i]
        #df_pt = df_pt.sort_values(by=['Degree'], ascending=False)
        node_HighestDegree = df_pt.loc[df_pt['Degree'].idxmax()][0]
        landmarks.append(node_HighestDegree)

    return landmarks

# Centrality/P, Pick the node with the lowest centrality in each partition:

def closeness_partitioned(complete_df, num_landmarks):

    landmarks = []
    # Iterate over partitions and Pick the node with the lowest centrality
    for i in range(num_landmarks):
        df_pt = complete_df.loc[complete_df['Partition'] == i]
        #df_pt = df_pt.sort_values(by=['Centrality'], ascending=False)
        node_LowestCentrality = df_pt.loc[df_pt['Centrality'].idxmin()][0]
        landmarks.append(node_LowestCentrality)

    return landmarks

def random_partitioned(complete_df, num_landmarks):
    landmarks = []
    for i in range(num_landmarks):
        df_pt = complete_df.loc[complete_df['Partition'] == i]
        random_node = df_pt.sample().iloc[0, 0]
        landmarks.append(random_node)
    return(landmarks)
#------------------------------ PRECOMPUTATION STEP---------------------------#

# Compute Distance:

def distance_estimation(source, target, distances_to_landmarks, distances_from_landmarks=None):
    upper_bound = math.inf
    if distances_from_landmarks is None:
        distances_from_landmarks = distances_to_landmarks
    for incoming, outgoing in zip(distances_to_landmarks, distances_from_landmarks):
        try:
            upper_bound = min(upper_bound, (incoming[source] + outgoing[target]))
        except:
            continue
    return (upper_bound)


# Calculate Landmark Distances:

'''
SSSPL returns a dictionary with the path length to all reachable nodes from a given landmark.
Finding the distance from a node to a given landmark is then a simple dictionary lookup, with constant time.
'''

def calculate_landmark_distances(G, landmarks):
    distances_from_landmarks = []
    for l in landmarks:
        distances_from_landmarks.append(dict(nx.single_source_shortest_path_length(G, l)))
    if not nx.is_directed(G):
        return distances_from_landmarks, None
    else:
        distances_to_landmarks =[]
        for l in landmarks:
            distances_to_landmarks.append(dict(nx.single_target_shortest_path_length(G, l)))
        return distances_to_landmarks, distances_from_landmarks


def print_time(start, function):
    end = time.time()
    print(function, end-start)
    return end

def score_accuracy(G, distances_to_landmarks, distances_from_landmarks, iterations):
    score = 0
    num_evaluated = 0
    sources = random.sample(nx.nodes(G), iterations)
    for source in sources:
        print(source)
        actual_distances = nx.single_source_shortest_path_length(G, source)
        actual_distances.pop(source)
        num_evaluated+=len(actual_distances)
        for k, v in actual_distances.items():
            estimated = distance_estimation(source, k, distances_to_landmarks, distances_from_landmarks)
            if estimated is math.inf:
                estimated = -1
            error = abs(v-estimated)/v
            score+=error
    score = score/num_evaluated
    print("score: ", score)
    return score


def evaluate(edgelist_filename, directed=False):
    if directed:
        G = nx.read_edgelist(edgelist_filename, create_using=nx.DiGraph)
    else:
        G = nx.read_edgelist(edgelist_filename)
    times = []
    start = time.time()
    eval_set = random.sample(nx.nodes(G), 20)
    eval_distances = []
    results = pd.DataFrame(columns=['source', 'target', 'actual distance', 'random', 'degree', 'betweenness', 'closeness',\
                                    'random constrained', 'degree constrained', 'betweenness constrained', 'closeness constrained', \
                                    'random partition', 'degree partition', 'closeness partition'])
    sources = []
    targets = []
    distances = []
    for node in eval_set:
        temp = nx.single_source_shortest_path_length(G, node)
        eval_distances.append(temp)
    times.append(time.time()-start)
    for node, distance_dict in zip(eval_set, eval_distances):
        for target, distance in distance_dict.items():
            sources.append(node)
            targets.append(target)
            distances.append(distance)
    results.loc[:, 'source'] = sources
    results.loc[:, 'target'] = targets
    results.loc[:, 'actual distance'] = distances

    for function, name in zip((get_random_nodes, get_high_degree_nodes, get_high_betweenness_nodes, get_low_closeness_nodes), \
                              ('random', 'degree', 'betweenness', 'closeness')):
        start = time.time()
        landmarks = function(G, NUM_LANDMARKS)
        distance_to, distance_from = calculate_landmark_distances(G, landmarks)
        mid = time.time()
        temp = []
        for source, target in zip(sources, targets):
            estimation = distance_estimation(source, target, distance_to, distance_from)
            if estimation is math.inf:
                estimation = -1
            temp.append(estimation)
        end = time.time()
        times.append(mid - start)
        times.append(end - mid)
        results.loc[:, name] = temp

    for method in ('random', 'degree', 'betweenness', 'closeness'):
        start = time.time()
        landmarks = constrained_landmark_selection(G, NUM_LANDMARKS, method)
        distance_to, distance_from = calculate_landmark_distances(G, landmarks)
        mid = time.time()
        temp = []
        for source, target in zip(sources, targets):
            estimation = distance_estimation(source, target, distance_to, distance_from)
            if estimation is math.inf:
                estimation = -1
            temp.append(estimation)
        end = time.time()
        times.append(mid - start)
        times.append(end - mid)
        name = method +' constrained'
        results.loc[:, name] = temp
    dataset = edgelist_filename.split(sep="/")[-1]
    result_filename = results_location + dataset[:-5] + ".tsv"
    start = time.time()
    partition_df = partition(G, NUM_LANDMARKS)
    times.append(time.time()-start)
    for function, name in zip((random_partitioned, degree_partitioned, closeness_partitioned), \
                              ('random partition', 'degree partition', 'closeness partition')):
        start = time.time()
        landmarks = function(partition_df, NUM_LANDMARKS)
        distance_to, distance_from = calculate_landmark_distances(G, landmarks)
        mid = time.time()
        temp = []
        for source, target in zip(sources, targets):
            estimation = distance_estimation(source, target, distance_to, distance_from)
            if estimation is math.inf:
                estimation = -1
            temp.append(estimation)
        end = time.time()
        times.append(mid - start)
        times.append(end - mid)
        results.loc[:, name] = temp


    print(result_filename)
    results.to_csv(result_filename, sep='\t')
    times = [str(time) for time in times]
    times = "\t".join(times)
    times = dataset + "\t" + times+ "\n"
    with open(time_file, "a") as out:
        out.write(times)
#----------------------------------- TEST ------------------------------------#

# Read Data:
# Define Number of Landmarks:
NUM_LANDMARKS = 100
# Check Graph Information:

time_file = "/results/times.tsv"
results_location = "snacs/results/"

time_categories = ("dataset\treal distances\trandom offline\trandom online\tdegree offline\tdegree online\tbetweenness offline" \
                   "\tbetweenness online\tcloseness offline\tcloseness online\tconstr random offline\tconstr random online\tconstr degree offline\t" \
                   "constr degree online\tconstr betweenness offline\tconstr betweenness online\tconstr closeness offline" \
                   "\tconstr closeness online\tpartitioning\trandom part offline\trandom part online\tdegree part offline" \
                   "\tdegree part online\tcloseness part offline\tcloseness part online\n")
with open(time_file, 'w') as file:
    file.write(time_categories)

datasets = ("CA-GrQc.txt", "WormNet.v3.benchmark.txt")
dataset_directed = (False, True)

for dataset, directed in zip(datasets, dataset_directed):
    filename = "/snacs/datasets/" + dataset
    evaluate(filename, directed=directed)

# Test for Different Landmark Selection Strategies:
#---------------------------- End of Code ------------------------------------#
