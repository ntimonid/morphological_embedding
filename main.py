import os
import argparse
import numpy as np
import pandas as pd
from src.utils import load_useful_data, name_2_fd
from src.neuron_population import NeuronPopulation
from src.projection_utils import (
    get_tSNE, 
    get_motif_distribution, 
    flatmap_the_clusters,
    Port_Han_Motif_Code
)
from src.morpho_embedding_tools import (
    compare_source_to_targets,
    morphological_embedding,
    flatmap_the_gradients
)
from src.topography_tools import (
    get_source_and_target_arrays,
    get_topopgraphic_rotations,
    get_topographic_plots,
    align_topography_and_gradient,
    compute_cca_and_permutation
)

def run_pipeline(args):
    # 1. Initialization
    print("--- Initializing Data ---")
    useful_vars = load_useful_data(args.data_repository)
    [annotation, template, acr2id, id2acr, ancestorsById, neuriteLengthDistribution, acr_to_morpho_id] = useful_vars
    nld_list = neuriteLengthDistribution.keys()
    
    # Setup directories
    braintell_dir = os.path.join(args.main_path, 'Data Repositories/Braintell')
    mouselight_dir = os.path.join(args.main_path, 'Data Repositories/Mouselight/json')
    another_data_repository = os.path.join(args.main_path, 'Data Repositories/mouse_connectivity/')
    flatmap_dir = os.path.join(args.main_path, 'Data Repositories/')
    code_dir = os.path.join(args.main_path, 'Code Repositories')

    # 2. Mesoscale Analysis
    print("--- Running Mesoscale Analysis ---")
    neuropop_cls = NeuronPopulation(data_path=args.data_repository, res=args.res)
    mesoscale_stats_df = neuropop_cls.make_connectivity_matrix(
        args.source_areas, args.target_areas, 
        feature='counts', mode='full', extract='terminals'
    )
    
    # Reindex if specified (paper order)
    if 'VPM' in args.source_areas:
        paper_order = ['SSp-ul', 'SSp-m', 'SSp-n', 'SSp-ll', 'SSs', 'SSp-bfd']
        available_order = [c for c in paper_order if c in mesoscale_stats_df.columns]
        mesoscale_stats_df = mesoscale_stats_df.reindex(available_order, axis=1)

    # 3. Projection Motif Analysis
    print("--- Analyzing Projection Motifs ---")
    cmap = ['#00FF00','#FF0000','#DD8800','#0000FF','#00AAFF','#333333','#880000','#884400','#000088','#0088DD',"#00FF00"]
    proj_mat_embedded, max_proj_labels = get_tSNE(mesoscale_stats_df, cmap)
    
    unique_motifs, representation_list = get_motif_distribution(
        mesoscale_stats_df, proj_thr=5, source_area=args.source_areas[0]
    )

    # Motif overrepresentation analysis
    mesoscale_stats_array = np.nan_to_num(np.array(mesoscale_stats_df))
    Port_Han_Motif_Code(mesoscale_stats_array, 6, 0.05, list(mesoscale_stats_df.columns))

    # 4. Morphological Embedding
    print("--- Computing Morphological Embedding (CPD) ---")
    source_neuron_ids = sorted(list(mesoscale_stats_df.loc[args.source_areas[0]].index))
    source_neuron_ids = [name_2_fd(val, acr_to_morpho_id, args.source_areas[0], mouselight_dir, braintell_dir) 
                         for val in source_neuron_ids]
    
    # Run CPD alignment (this can be slow)
    print(f"Comparing {len(source_neuron_ids)} neurons...")
    Affinity_df_total, Affinity_df_trs_total, soma_dst_df, soma_pos = compare_source_to_targets(
        source_neuron_ids, source_neuron_ids
    )

    print("--- Generating UMAP Embedding and Gradient ---")
    keyRegions = list(mesoscale_stats_df.columns)
    
    # We need color_neuron_dict from flatmap_the_clusters or similar for plotting
    source_vol, target_flts, color_neuron_dict = flatmap_the_clusters(
        mesoscale_stats_df, neuropop_cls, max_proj_labels, 
        annotation, template, acr2id, another_data_repository,
        flatmap_dir, code_dir, source=args.source_areas[0]
    )

    morpho_res = morphological_embedding(
        Affinity_df_trs_total, soma_pos, annotation, color_neuron_dict, 
        max_proj_labels, nld_list, keyRegions
    )
    Affinity_mat_red, diff_grad_root1, diff_grad_nrm, neuronid_to_grad, mdist_clust_labels = morpho_res

    # 5. Topographic Analysis and Validation
    print("--- Running Topographic Alignment & Validation ---")
    soma_arr, target_arr, grad_position = get_source_and_target_arrays(
        neuronid_to_grad, neuropop_cls, nld_list, annotation, id2acr, args.topography_target
    )
    
    Affine_grand, X_trs_aff, r, Rotats_grand, mse_aff = get_topopgraphic_rotations(soma_arr, target_arr)
    
    # Alignment & Spearman Correlation
    topograd_res = align_topography_and_gradient(soma_arr, target_arr, grad_position)
    
    # CCA and Permutation Testing
    print("--- Running CCA & Permutation Testing ---")
    # Using 2D UMAP coordinates as features
    obs_corr, p_val = compute_cca_and_permutation(Affinity_mat_red, soma_arr, n_permutations=args.n_perms)
    print(f"CCA Correlation: {obs_corr:.4f}, p-value: {p_val:.4f}")

    # 6. Final Visualization
    print("--- Generating Gradient Flatmaps ---")
    flatmap_the_gradients(
        neuropop_cls, neuronid_to_grad, annotation, template, acr2id, nld_list, 
        another_data_repository, flatmap_dir, code_dir
    )
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Morphology Embedder Pipeline')
    parser.add_argument('--main_path', type=str, default='.', help='Main path for data repositories')
    parser.add_argument('--data_repository', type=str, default='atlas_files', help='Path to atlas files')
    parser.add_argument('--res', type=int, default=10, help='Resolution')
    parser.add_argument('--source_areas', nargs='+', default=['VPM'], help='Source brain areas')
    parser.add_argument('--target_areas', nargs='+', 
                        default=['SSs', 'SSp-bfd', 'SSp-m', 'SSp-ul', 'SSp-ll', 'SSp-n'], 
                        help='Target brain areas')
    parser.add_argument('--topography_target', type=str, default='SSp', help='Target area for topography analysis')
    parser.add_argument('--n_perms', type=int, default=1000, help='Number of permutations for CCA validation')
    
    args = parser.parse_args()
    run_pipeline(args)
