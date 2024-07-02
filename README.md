# Fast Shortest Path Distance Estimation in Large Networks

Many real-world applications rely on computing distances between node pairs. 

In this project, we explored methods to estimate shortest path distance between two nodes in large networks fast using various strategies proposed. We used landmark-selection strategies to estimate distance estimation on real-world graphs.

We implemented approximate landmark-based methods for point-to-point distance estimation. The central idea was to select a subset of nodes as landmarks and compute the distances offline from each node in the graph to those landmarks. In the course of run-time, we can use these pre-computed distances from landmarks to estimate distance between two nodes. We tested the robustness and efficiency of these techniques and strategies with five large real-world network datasets. We extended our work by applying these methods on directed graphs as well. We also explored a new landmark selection strategy based on approximate betweenness centrality. 

Our experiments suggest that optimal landmark selection can yield more accurate results faster than the traditional approach of selecting landmarks at random. We evaluated the efficiency of these strategies using approximation error.
