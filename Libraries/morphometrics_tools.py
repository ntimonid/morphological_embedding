from imports import *
from utils import *

from NeuronMorphology import NeuronMorphology, get_morphometrics

def get_morphometrics_per_cluster(Affinity_df_trs_total, neuropop_cls, mdist_clust_labels, nld_list, mouselight_dir, braintell_dir):

    morpho_stats_per_cluster = {cluster: [] for cluster in np.unique(mdist_clust_labels)}
    morpho_stats_per_cluster2 = {cluster: [] for cluster in np.unique(mdist_clust_labels)}
    out_orientation = ['um(10)','PIR','corner']

    keep_list = []
    list_of_neurons_per_cluster = {}

    for cluster_idx,cluster in enumerate(np.unique(mdist_clust_labels)):
        list_of_neurons_per_cluster[cluster] = []
        cluster_indices = np.where(mdist_clust_labels == cluster)[0]
        cluster_fnames = list(Affinity_df_trs_total.index[cluster_indices])
        cluster_fnames2 = [braintell_2_nld(nld_list, neuron_id) for neuron_id in cluster_fnames]
        for n_idx,cluster_fname,neuron_id in zip(cluster_indices,cluster_fnames2,cluster_fnames):
            if cluster_fname in neuropop_cls.targets_per_neuron:
                if 'AA' in neuron_id:
                    mode = 'mouselight'
                else:
                    mode = 'braintell'
                if 'AA' in neuron_id:
                    file_id = os.path.join(mouselight_dir, neuron_id + '.json.gz')
                else:
                    file_id = os.path.join(braintell_dir, neuron_id + '.json.gz')

                morpho_cls = NeuronMorphology(neuronFile = file_id)
                morpho_cls.transform(out_orientation = out_orientation)
                morpho_cls.points = morpho_cls.points[:,0:3]
                morpho_cls.points -= morpho_cls.soma[0:3]
                morpho_cls.points[0,:] = [0,0,0]
                A, D, morph_stats = get_morphometrics(morpho_cls.asDict(), neurite = 'axons')
                if A != -1:
                    keep_list.append(n_idx)
                    x1 = np.mean(list(A.get_radial_distance().values()))
                    x2 = np.max(list(A.get_path_length().values()))
                    x3 = np.mean(list(A.get_segment_length().values()))
                    x4 = np.max(list(A.get_branch_order().values()))
                    x5 = np.mean(list(A.get_branch_angles().values()))
                    x6 = np.mean(list(A.get_root_angles().values()))
                    list_of_neurons_per_cluster[cluster].append(neuron_id)
                    morpho_stats_per_cluster[cluster].append([x1,x2,x3,x4,x5,x6])
                    morpho_stats_per_cluster2[cluster].append(morph_stats)

    return morpho_stats_per_cluster, morpho_stats_per_cluster2, list_of_neurons_per_cluster, keep_list


def plot_morphometrics(morpho_stats_per_cluster, morpho_stats_per_cluster2, list_of_neurons_per_cluster, neuronid_to_grad, keep_list,
                       mdist_clust_labels, cut_off = 0.001, rho_val = 0.6, savefile = None):

    cmap_loc = plt.get_cmap('viridis')
    mdist_clust_labels_cpy2 = deepcopy(mdist_clust_labels)
    mdist_clust_labels_cpy2[mdist_clust_labels == 0] = 2
    mdist_clust_labels_cpy2[mdist_clust_labels == 1] = 0
    mdist_clust_labels_cpy2[mdist_clust_labels == 2] = 1
    mdist_clust_labels_cpy2 = mdist_clust_labels_cpy2[keep_list]

    mdist_clust_labels_cpy = deepcopy(mdist_clust_labels)
    mdist_clust_labels_cpy[mdist_clust_labels_cpy==2] = 4
    mdist_clust_labels_cpy[mdist_clust_labels_cpy==1] = 3

    grad_list = np.asarray([neuronid_to_grad[val] for cluster in [0,1,2] for val in list_of_neurons_per_cluster[cluster]])
    for idx,metric in enumerate(morpho_stats_per_cluster2[0][0]['statistic']):
        temp_list = np.asarray([np.array(neuron['value'][idx]) for cluster in [0,1,2] for neuron in morpho_stats_per_cluster2[cluster]])
        temp_list[np.isnan(temp_list)] = 0
        temp_corr_pear, p_pear = sci.stats.pearsonr(grad_list, temp_list)
        temp_corr_spear, p_spear = sci.stats.spearmanr(grad_list, temp_list)
        if p_spear < cut_off and np.abs(temp_corr_spear) > rho_val:
            fig = plt.figure()
            ax = plt.gca()
            plt.title('{} - r : {}'.format(metric,np.around(temp_corr_spear,2)))
            plt.xlabel('Gradient value')
            plt.ylabel('morphometric')
            plt.scatter(grad_list, temp_list, cmap = cmap_loc, c = mdist_clust_labels_cpy[keep_list])
            patches  = [mpatches.Patch(color=cmap_loc(c/4), label='cluster {}'.format(c2+1))
                       for c,c2 in zip([4,3,0], [0,1,2])]
            ax.legend(handles=patches, loc='lower right', prop={'size': 10})
            if savefile is not None:
                plt.savefig('{}_{}.svg'.format(savefile,metric), bbox_inches = 'tight')
            plt.show()
            print(metric, temp_corr_spear)

    morpho_metr = {0: 'radial_distance', 1: 'path_length', 2: 'segment_length', 3: 'branch_order',
                   4: 'branch_angles', 5: 'root_angles'}
    stats_cat_no_1 = np.asarray([val for cluster in [0,1,2] for val in morpho_stats_per_cluster[cluster]])
    for categ in range(5):
        temp_corr_pear, p_pear = sci.stats.pearsonr(grad_list, stats_cat_no_1[:,categ])
        temp_corr_spear, p_spear = sci.stats.spearmanr(grad_list, stats_cat_no_1[:,categ])
        if p_spear < cut_off and np.abs(temp_corr_spear) > rho_val:
            fig = plt.figure()
            ax = plt.gca()
            plt.title('{} - r : {}'.format(morpho_metr[categ],np.around(temp_corr_spear,2)))
            plt.xlabel('Gradient value')
            plt.ylabel('morphometric')
            plt.scatter(grad_list, stats_cat_no_1[:,categ], cmap = cmap_loc, c = mdist_clust_labels_cpy[keep_list])
            patches  = [mpatches.Patch(color=cmap_loc(c/4), label='cluster {}'.format(c2+1))
                       for c,c2 in zip([4,3,0], [0,1,2])]
            ax.legend(handles=patches, loc='lower right', prop={'size': 10})
            if savefile is not None:
                plt.savefig('{}_{}.svg'.format(savefile, morpho_metr[categ]), bbox_inches = 'tight')
            plt.show()
            print(morpho_metr[categ], temp_corr_spear)
