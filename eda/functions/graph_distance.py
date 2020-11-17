import netcomp
from itertools import combinations
import matplotlib.pyplot as plt

def plot_graph_distance(networks, network_names):
    dc_distance_list = []
    ged_distance_list = []
    names = []
    network1 = networks[0]
    ## compare all the network starting from the second to the first network (reduce run time)
    i = 1
    for network in networks[1:]:
        dc_distance_list.append(netcomp.deltacon0(network1.values, network.values))
        ged_distance_list.append(netcomp.edit_distance(network1.values, network.values))
        names.append(f'{network_names[0]} vs {network_names[i]}')
        i += 1
    ## pairwise combination
    # for sub_network1, sub_network2 in combinations(zip(networks, network_names), 2):
    #     dc_distance_list.append(netcomp.deltacon0(sub_network1[0].values, sub_network2[0].values))
    #     ged_distance_list.append(netcomp.edit_distance(sub_network1[0].values, sub_network2[0].values))
    #     names.append(f'{sub_network1[1]} vs {sub_network2[1]}')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(names, dc_distance_list)
    plt.title('Deltacon distance')
    plt.xlabel('Number of edges')
    plt.subplot(1, 2, 2)
    plt.bar(names, ged_distance_list)
    plt.title('GEM distance')
    plt.xlabel('Number of edges')
    plt.subplots_adjust(wspace=0.5)
