# Morphology Embedder

Morpological Embedder takes as input cell morphologies in the form of 3D point clouds and creates a 2-dimensional embedding in which cells with a similar morphology (shape) are close to each other.

![Schematic Overview](figures/Figure_6_updated.jpg)

## Motivation
Comparing cell morphologies is important for exploring the role of cell phenotypes in neurodevelopmental disorders and neurodegenerative diseases. However, this is not computationally trivial because unlike arrays, point clouds do not necessarily have the same number of points and these points are not aligned. This necessitates the development of a methodology for comparing the shapes of different cells. For this reason the following methodology has been implemented:
## Methodology
1. **Alignment:** The Coherent Point Drift (CPD) algorithm is used to create an alignment score between the shapes of two cells (*Myronenko et al., 2012*).
2. **Similarity Matrix:** Comparison of all cells of interest results in a pairwise distance matrix.
3. **Embedding:** The similarity matrix is processed using the **UMAP** dimensionality reduction algorithm to produce a 2D embedding that preserves both local and global morphological relationships.
4. **Gradient Analysis:** A morphological gradient is computed using a **two-anchor relative distance index** in the embedding space.
5. **Spatial Correlation:** The morphological gradient is correlated with the spatial distribution of neurons (using **geodesic distances** for brain space).
6. **Validation:** Statistical significance is verified via **Canonical Correlation Analysis (CCA)** and **Permutation Testing**.

## Installation
```bash
git clone https://github.com/ofcru/Morphology_Embedder.git
cd Morphology_Embedder
pip install .
```
Or via requirements:
```bash
pip install -r requirements.txt
```

## Usage
The main entry point for the pipeline is `main.py`. You can run the full analysis (Mesoscale, CPD, UMAP, Topography) via the command line:

```bash
python main.py --data_repository atlas_files --source_areas VPM --n_perms 1000
```

### Arguments
- `--main_path`: Base directory for data (default: `.`).
- `--data_repository`: Folder containing atlas files (default: `atlas_files`).
- `--source_areas`: Source brain regions (e.g., `VPM`).
- `--target_areas`: Target brain regions for connectivity.
- `--n_perms`: Number of permutations for CCA validation.

## References
This code has been used in the following published work:  
Timonidis, Nestor, et al. "Translating single-neuron axonal reconstructions into meso-scale connectivity statistics in the mouse somatosensory thalamus." Frontiers in neuroinformatics 17 (2023): 1272243. doi: https://doi.org/10.3389/fninf.2023.1272243  

*Myronenko, Andriy, and Xubo Song. "Point set registration: Coherent point drift." IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.


