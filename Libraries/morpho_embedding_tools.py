
from imports import *
from utils import *

# Our libraries ...
import convertAllenSpace as CAS
import BrainPlotter_latest as BP
import cortical_map_extra as cm_new

from NeuronMorphology import NeuronMorphology

sys.path.append('../../2_Neuron_Comparison_Visualization/neuroncomparer/')
from cpd_registration import RigidRegistration

hex2rgb = lambda x : tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


def compare_source_to_targets(source_neuron_ids, target_neuron_ids, cpd_params = None, flip = False):

    if cpd_params is None:
        cpd_params = {'max_it': 30, 'flag_in' : [1,1,-1], 'tol' : 0.001, 'branch_constraint': False}

    Affinity_dict = OrderedDict(); Affinity_dict_trs = OrderedDict();
    soma_dst = OrderedDict(); soma_pos = OrderedDict()

    # Source pre-processing
    for source_neuron_id in source_neuron_ids:

        db = 'mouselight' if 'AA' in source_neuron_id else 'braintell'
        neuronPath = 'https://neuroinformatics.nl/HBP/neuronsreunited-viewer/{}_json_gz/{}.json.gz'.format(db,source_neuron_id)
        file_content = requests.get(neuronPath).content

        source_neuron = json.loads(zlib.decompress(file_content, 16+zlib.MAX_WBITS))
        source_morpho_cls = NeuronMorphology(neuronDict = source_neuron)
        source_morpho_cls.transform(out_orientation = ['um(10)','PIR','corner'])
        source_name = source_neuron_id

        source_minor_lines, source_minor_points = source_morpho_cls.subsampledCopy([1,2], minDistance = 1e10)
        source_soma_point = [line[1] for lineIdx,line in enumerate(source_minor_lines) if line[0] == 1][0]
        X = np.array(source_minor_points)
        X[0,:] = X[1,:]
        X_c = X - X[source_soma_point,:]
        soma_pos[source_neuron_id] = X[source_soma_point,:]

        Affinity_dict[source_name] = OrderedDict(); Affinity_dict_trs[source_name] = OrderedDict(); soma_dst[source_name] = OrderedDict()

        for target_neuron_id in target_neuron_ids:

            if source_neuron_id == target_neuron_id:
                Affinity_dict[source_name][target_neuron_id] = 0
                Affinity_dict_trs[source_name][target_neuron_id] = 0
                soma_dst[source_name][target_neuron_id] = 0
                continue

            # The following five lines are in case we want to not load stored neurons but download them on the fly instead ...
            db = 'mouselight' if 'AA' in target_neuron_id else 'braintell'
            neuronPath = 'https://neuroinformatics.nl/HBP/neuronsreunited-viewer/{}_json_gz/{}.json.gz'.format(db,target_neuron_id)
            file_content = requests.get(neuronPath).content

            target_neuron = json.loads(zlib.decompress(file_content, 16+zlib.MAX_WBITS))
            target_morpho_cls = NeuronMorphology(neuronDict = target_neuron)
            target_morpho_cls.transform(out_orientation = ['um(10)','PIR','corner'])
            target_name = target_neuron_id #target_morpho_cls.filename.split('.')[0]

            target_minor_lines, target_minor_points = target_morpho_cls.subsampledCopy([1,2], minDistance = 1e10)
            Y = np.array(target_minor_points)
            target_soma_point = [line[1] for lineIdx,line in enumerate(target_minor_lines) if line[0] == 1][0]
            Y[0,:] = Y[1,:]

            if Y[target_soma_point, 2] > 1140//2:  # needs flipping ...
                Y[:,2] = 1140 - Y[:,2]

            Y_c = Y - Y[target_soma_point, :]

            # Time for registration ...
            reg = RigidRegistration(**{'X': X_c, 'Y': Y_c},\
                                    tolerance = cpd_params['tol'], max_iterations = cpd_params['max_it'],
                                    flag_in = cpd_params['flag_in'])
            TY, (s_reg, R_reg, t_reg) = reg.register()

            match_1_to_2 = np.argmax(reg.P,axis = 1)
            X_c_remap = X_c[match_1_to_2,:]
            MSE_trs = np.linalg.norm(X_c_remap[:,0:3] - TY[:,0:3])
            X_remap = X[match_1_to_2,:]
            MSE = np.linalg.norm(X_remap[:,0:3] - Y[:,0:3])

            soma_dst[source_name][target_name] = np.linalg.norm(X[source_soma_point,:] -  Y[target_soma_point, :])
            Affinity_dict[source_name][target_name] = MSE
            Affinity_dict_trs[source_name][target_name] = MSE_trs

    Affinity_df_total = pd.DataFrame(Affinity_dict, columns = Affinity_dict.keys(), index = Affinity_dict.keys())
    Affinity_df_trs_total = pd.DataFrame(Affinity_dict_trs, columns = Affinity_dict_trs.keys(), index = Affinity_dict_trs.keys())
    soma_dst_df = pd.DataFrame(soma_dst, columns = soma_dst.keys(), index = soma_dst.keys())

    return Affinity_df_total, Affinity_df_trs_total, soma_dst_df, soma_pos


def morphological_embedding(Affinity_df_trs_total, soma_pos, annotation, color_neuron_dict, max_proj_labels,
                            nld_list, keyRegions, right_only = False, cmap = None, n_clust = 3, savefig = None):

    if right_only is False:
        z_mid = annotation.shape[2]
    else:
        z_mid = annotation.shape[2]/2

    # Here put the ipsilateral constraint ...
    temp_list = []; index_list = []
    for idx,neuron_id in enumerate(Affinity_df_trs_total.index):
        x,y,z = soma_pos[neuron_id][0:3]
        if z < z_mid:
            neuron_id2 = braintell_2_nld(nld_list, neuron_id)
            temp_list.append(color_neuron_dict[neuron_id2]) #color_neuron_subdict[neuron_id2] for projection-subtypes
            index_list.append(idx)
    index_list = np.asarray(index_list)

    Affinity_mat_trs = np.asarray(Affinity_df_trs_total)
    Affinity_mat_trs = Affinity_mat_trs[index_list,:][:,index_list]
    Affinity_mat_trs[np.isnan(Affinity_mat_trs)] = 0

    Affinity_mat_red = TSNE(2, metric = 'precomputed', init = 'random').fit_transform(Affinity_mat_trs)
    ac = AgglomerativeClustering(n_clusters = n_clust)
    mdist_clust_labels = ac.fit_predict(Affinity_mat_red)

    root_pt = 35 #np.argmax(np.sum(Affinity_mat_red, axis = 1))
    diff_grad = [geodesic(list(Affinity_mat_red[root_pt,:]),list(Affinity_mat_red[idx,:])).km for idx in range(len(Affinity_mat_red))]
    diff_grad_nrm = 1 - diff_grad/np.max(diff_grad)
    neuronid_to_grad = OrderedDict([(val, diff_grad_nrm[idx]) for idx,val in enumerate(list(Affinity_df_trs_total.index))])

    if cmap is None:
        cmap = ['#00FF00','#FF0000','#DD8800','#0000FF','#00AAFF','#333333','#880000','#884400','#000088','#0088DD',"#00FF00"]

    mdist_clust_labels_cpy = deepcopy(mdist_clust_labels)
    mdist_clust_labels_cpy[mdist_clust_labels_cpy==2] = 4
    mdist_clust_labels_cpy[mdist_clust_labels_cpy==1] = 3

    unique_vals = len(np.unique(diff_grad_nrm))
    in_color_nrm = np.array(deepcopy(hex2rgb('#21918c')))/255
    in_color_nrm2 = np.array(deepcopy(hex2rgb('#A020F0')))/255
    color_range = np.vstack((in_color_nrm + [0.1,0.1,0.1], in_color_nrm, in_color_nrm/2,
                             in_color_nrm2/2, in_color_nrm2,  in_color_nrm2 + [0.1,0.1,0.1]))
    color_range[color_range > 1] = 1
    cmap_fg = LinearSegmentedColormap.from_list('random', color_range, N = unique_vals)
    cmap_loc = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(12,4),dpi=96)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('t-SNE for morphological gradients - registered distance')
    x1,y1 = Affinity_mat_red[np.argmin(diff_grad_nrm)]
    x2,y2 = Affinity_mat_red[np.argmax(diff_grad_nrm)]
    plt.plot(x1,y1,'r*')
    plt.plot(x2,y2, 'r*')
    sc = ax.scatter(Affinity_mat_red[:,0], Affinity_mat_red[:,1], cmap = cmap_fg, c = diff_grad_nrm)
    if savefig is not None:
        plt.savefig(savefig, bbox_inches = 'tight')
    plt.show()

    for color_type, in_color in {'morphological clusters': mdist_clust_labels_cpy, 'projection clusters': temp_list}.items():
        fig = plt.figure(figsize=(12,4),dpi=96)
        ax = fig.add_subplot(1,2,1)
        ax.set_title('t-SNE for total {}'.format(color_type))
        plt.plot(x1,y1,'r*')
        plt.plot(x2,y2, 'r*')
        sc = ax.scatter(Affinity_mat_red[:,0], Affinity_mat_red[:,1], cmap = cmap_loc, c = in_color)
        if color_type == 'projection clusters':
            patches = [mpatches.Patch(color=cmap[c], label='{}'.format(keyRegions[c])) for c in np.unique(max_proj_labels)]
        else:
            patches  = [mpatches.Patch(color=cmap_loc(c/4), label='cluster {}'.format(c2+1))
                           for c,c2 in zip(np.unique(mdist_clust_labels_cpy), np.unique(mdist_clust_labels))]
        ax.legend(handles=patches, loc='lower right', prop={'size': 10})
        if savefig is not None:
            plt.savefig(savefig.split('.')[0] + color_type + '.eps', bbox_inches = 'tight')
        plt.show()

    return Affinity_mat_red, diff_grad, diff_grad_nrm, neuronid_to_grad, mdist_clust_labels


def flatmap_the_gradients(neuropop_cls, neuronid_to_grad, annotation, template, acr2id, nld_list,
                          data_repository, flatmap_dir, code_dir, savefile = None):

    cmap_loc = plt.get_cmap('viridis')
    annot_shape = annotation.shape

    cluster_targets_new = []; soma_cords_new = []

    templateMapper = cm_new.CorticalMap('dorsal_flatmap',template.shape, module_name = flatmap_dir)

    grad_vals_target = []
    for neuron_idx,neuron_grad in neuronid_to_grad.items():
        neuron_reidx = braintell_2_nld(nld_list, neuron_idx)
        cluster_targets_new.extend(neuropop_cls.targets_per_neuron[neuron_reidx])
        soma_idx = np.where(np.array(list(neuropop_cls.targets_per_neuron.keys())) == neuron_reidx)[0][0]
        soma_cords_new.append(neuropop_cls.somata[soma_idx])
    #     tmp_clr = hex2rgb(mpl.colors.to_hex(np.array(cmap_loc(neuron_grad))[0:3]))
        for n_idx in range(len(neuropop_cls.targets_per_neuron[neuron_reidx])):
            grad_vals_target.append(neuron_grad)

    grad_vals_soma = np.array(list(neuronid_to_grad.values()))
    grad_vals_target = np.array(grad_vals_target)

    target_color = hex2rgb('#21918c')
    target_pts_2d = CAS.fast_pts2volume(np.array(cluster_targets_new), annotation.shape, values = grad_vals_target)
    soma_vol   = CAS.fast_pts2volume(np.array(soma_cords_new), annotation.shape, values = grad_vals_soma)#*1000

    if savefile is not None:
        savefile1 = '{}_dorsal_flatmap'.format(savefile)
    else:
        savefile1 = None
        savefile2 = None

    flt_Mat = BP.plot_flatmap(target_pts_2d, proj_type = 'dorsal_flatmap',
                              data_dir = data_repository, code_dir = code_dir,
                              flatmap_dir = flatmap_dir, new_color = target_color,
                              savefile = savefile1)

    for plane in ['IR','RP','PI']:
        if savefile is not None:
            savefile2 = '{}_max_projection_gradient_{}'.format(savefile, plane)
        BP.plot_plane(soma_vol, annotation, template, acr2id,
                       'VPM', target_color, code_dir, plane, orient = 'left',
                       savefile = savefile2)
