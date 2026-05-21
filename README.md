# morphological_embedding

Morpological Embedding takes as input cell morphologies in the form of 3D point clouds and creates a 2-dimensional embedding in which cells with a similar morphology (shape) are close to each other.

## Motivation
Comparing cell morphologies is important for exploring the role of cell phenotypes in neurodevelopmental disorders and neurodegenerative diseases. However, this is not computationally trivial because unlike arrays, point clouds do not necessarily have the same number of points and these points are not aligned. This necessitates the development of a methodology for comparing the shapes of different cells. For this reason the following methodology has been implemented:
1. The Coherent Point Drift algorithm is used to create an alignment score between the shapes of two cells (Myrorenko et al., 2012).
2. The comparison of all possible cells of interest results in a 2D similarity matrix.
3. The similarity matrix is then given as input to the t-SNE dimensionality reduction algorithm, which outputs a 2D embedding of their alignment.
5. 2D scatter plots are then used to visualize the result.
