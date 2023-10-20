from imports import *
from utils import *

import convertAllenSpace as CAS
import BrainPlotter_latest as BP
import cortical_map_extra as cm_new

nchoosek =  lambda X, b : [comb for comb in combinations(X, b)]
hex2rgb = lambda x : tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def get_tSNE(mesoscale_stats_df, cmap, savefile = None):

    keyRegions = list(mesoscale_stats_df.columns)
    mesoscale_stats_array = np.array(mesoscale_stats_df)
    mesoscale_stats_array[np.where(np.isnan(mesoscale_stats_array))] = 0

    Proj_mat_embedded = TSNE(2, init = 'random').fit_transform(mesoscale_stats_array)
    max_proj_labels = np.argmax(mesoscale_stats_array,axis = 1)

    patches = [mpatches.Patch(color=cmap[c], label='{}'.format(keyRegions[c])) for c in np.unique(max_proj_labels)]
    rgb_proj = [cmap[dom] for dom in max_proj_labels]
    upd_clrs = [mpl.colors.to_hex(np.array(mpl.colors.to_rgb(rgb_proj[idx]))) \
                for idx in range(len(rgb_proj))]

    fig = plt.figure(figsize=(12,4), dpi=96)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('t-SNE')
    sc = ax.scatter(Proj_mat_embedded[:,0], Proj_mat_embedded[:,1], c = upd_clrs)
    ax.legend(handles=patches, loc='upper left', prop={'size': 10})
    plt.show()
    if savefile is not None:
        fig.savefig('Figures/vpm_tsne_scatter_{}.png'.format(clust_style), bbox_inches = 'tight')

    return Proj_mat_embedded, max_proj_labels


def get_motif_counts(mesoscale_stats_df, savefile = None):

    mesoscale_stats_array = np.array(mesoscale_stats_df)
    mesoscale_stats_array[np.where(np.isnan(mesoscale_stats_array))] = 0
    x_shape = mesoscale_stats_array.shape[0]

    # categorize neurons based on their projection properties: monofocal, bifurcating, etc ..
    proj_sum = np.array([len(np.nonzero(mesoscale_stats_array[i,:])[0]) for i in range(x_shape)])

    monofocal = np.where(proj_sum == 1)[0]
    bifurcating = np.where(proj_sum == 2)[0]
    trifurcating = np.where(proj_sum == 3)[0]
    quadrifurcating = np.where(proj_sum == 4)[0]
    above_four = np.where(proj_sum > 4)[0]

    motif_count_list = [monofocal, bifurcating, trifurcating, quadrifurcating, above_four]

    fig = plt.figure(figsize = (16,6))
    motif_labels = ['monofocal','bifurcating','trifurcating','quadrifurcating']
    motif_counts = [len(monofocal),len(bifurcating),len(trifurcating),len(quadrifurcating)]
    plt.bar(motif_labels, motif_counts)
    plt.show()
    if savefile is not None:
        fig.savefig(savefile, bbox_inches = 'tight')

    return motif_count_list

def flatmap_the_clusters(mesoscale_stats_df, neuropop_cls, max_proj_labels, annotation, template, acr2id,
                         data_repository, flatmap_dir, code_dir, source = 'VPM', savefile = None):

    cmap = ['#00FF00','#FF0000','#DD8800','#0000FF','#00AAFF','#333333','#880000','#884400','#000088','#0088DD',"#00FF00"]
    cmap_loc = plt.get_cmap('viridis')

    annot_shape = annotation.shape
    templateMapper = cm_new.CorticalMap('dorsal_flatmap', annot_shape, module_name = flatmap_dir)

    color_neuron_dict = {}
    color_neuron_dict_morpho = {}
    target_color = []

    for cluster_idx,cluster in enumerate(np.unique(max_proj_labels)):
        cluster_indices = np.where(max_proj_labels == cluster)[0]
        cluster_fnames = list(mesoscale_stats_df.loc[source].index[cluster_indices])
        cluster_targets = []; soma_cords = []
        for n_idx,cluster_fname in zip(cluster_indices,cluster_fnames):
            if cluster_fname in neuropop_cls.targets_per_neuron:
                cluster_targets.extend(neuropop_cls.targets_per_neuron[cluster_fname])
                soma_idx = np.where(np.array(list(neuropop_cls.targets_per_neuron.keys())) == cluster_fname)[0][0]
                soma_cords.append(neuropop_cls.somata[soma_idx])
                color_neuron_dict[cluster_fname] = mpl.colors.to_hex(np.array(mpl.colors.to_rgb(cmap[cluster])))

        target_pts_2d   = CAS.fast_pts2volume(np.array(cluster_targets), annotation.shape)
        soma_vol   = CAS.fast_pts2volume(np.array(soma_cords), annotation.shape)
        if cluster_idx == 0:
            target_flts = deepcopy(target_pts_2d.reshape((annot_shape[0],annot_shape[1],annot_shape[2],1)))
            source_vol  = deepcopy(soma_vol.reshape((annot_shape[0],annot_shape[1],annot_shape[2],1)))
        else:
            target_flts = np.concatenate([target_flts,
                                          target_pts_2d.reshape((annot_shape[0],annot_shape[1],annot_shape[2],1))],
                                          axis = 3)
            source_vol = np.concatenate([source_vol,
                                         soma_vol.reshape((annot_shape[0],annot_shape[1],annot_shape[2],1))],
                                         axis = 3)
        tmp_clr = hex2rgb(cmap[cluster])
        if tmp_clr == (0,0,0):
            tmp_clr = [52,52,52]
        target_color.append(tmp_clr)

    if savefile is not None:
        savefile1 = '{}_dorsal_flatmap'.format(savefile)
    else:
        savefile1 = None
        savefile2 = None
    flt_Mat = BP.plot_flatmap(target_flts, proj_type = 'dorsal_flatmap',template = None, annot_file = None,
                              savefile = savefile1, data_dir = data_repository, annot = 'allen', code_dir = code_dir,
                              flatmap_dir = flatmap_dir, new_color = target_color)

    for plane in ['IR','RP','PI']:
        if savefile is not None:
            savefile2 = '{}_max_projection_{}'.format(savefile, plane)
        BP.plot_plane(source_vol, annotation, template, acr2id,
                      source, target_color, code_dir, plane, orient = 'left',
                      savefile = savefile2)

    return source_vol, target_flts, color_neuron_dict


def get_motif_distribution(mesoscale_stats_df, proj_thr = 5, source_area = 'VPM', savefile = None):

    mesoscale_stats_array = np.array(mesoscale_stats_df)
    mesoscale_stats_array[np.where(np.isnan(mesoscale_stats_array))] = 0
    mesoscale_stats_array_cpy = deepcopy(mesoscale_stats_array)

    x_shape = mesoscale_stats_array.shape[0]
    mesoscale_stats_array[mesoscale_stats_array_cpy >= proj_thr] = 1
    mesoscale_stats_array[mesoscale_stats_array_cpy < proj_thr] = 0


    representation_list = []
    non_unique = []

    dist_mat = np.reshape([np.linalg.norm(x-y) for x in mesoscale_stats_array \
                           for y in mesoscale_stats_array],(x_shape,x_shape))
    unique_motifs = np.arange(len(dist_mat))

    for i in range(len(dist_mat)):
        if i in non_unique: continue
        zero_pts = np.where(dist_mat[i,:] == 0)[0]
        representation_list.append(len(zero_pts))
        if len(zero_pts) > 1:
            non_unique.extend(zero_pts[1:len(zero_pts)])

    unique_motifs = np.delete(unique_motifs,non_unique)

    motif_dict = OrderedDict()
    sort_motifs = np.argsort(representation_list)[::-1]
    for idx,motif in enumerate(unique_motifs[sort_motifs]):
        neur_id = mesoscale_stats_df.index[motif]
        targets = mesoscale_stats_df.loc[neur_id]
        target_acros = list(targets[mesoscale_stats_array[motif].astype('bool')].index)
        source_indices = np.where(dist_mat[motif,:] == 0)[0]
        motif_dict[idx+1] = [target_acros, list(mesoscale_stats_df.loc[source_area].index[source_indices])]

    fig = plt.figure(figsize=(16,6))
    (x,y) = mesoscale_stats_array[unique_motifs].T.shape
    p = sns.heatmap(np.abs(mesoscale_stats_array[unique_motifs][sort_motifs,:].T-1))
    p.set_yticklabels(list(mesoscale_stats_df.columns), rotation = 0, fontsize = 12)
    plt.xticks([idx for idx in range(y)],[representation_list[val] for val in sort_motifs], fontsize = 12)
    ax = plt.gca()
    p.set_xlabel('Projection Motif Counts', fontsize = 16)
    plt.title('VPM patterns', fontsize = 20)
    if savefile is not None:
        fig.savefig(savefile, bbox_inches = 'tight')

    return unique_motifs, representation_list



def get_motif_barplots(mesoscale_stats_df, neuronid_to_grad, nld_list, participation_thr = 4, exclude_list = None, savefile = None):

    mesoscale_stats_array = np.array(mesoscale_stats_df)
    mesoscale_stats_array[np.where(np.isnan(mesoscale_stats_array))] = 0

    columns = np.array(list(mesoscale_stats_df.columns))
    neuron_ids = mesoscale_stats_df.index

    neuron_to_motif = OrderedDict()
    motif_to_neuron = OrderedDict()
    for idx,row in enumerate(mesoscale_stats_array):
        rsorted = np.argsort(row)[::-1]
        nzero = np.where(row[rsorted] > 0)[0]
        high_areas = columns[rsorted][nzero]
        orig_str = str(high_areas[0])
        if len([v2 for v2 in exclude_list if v2 in orig_str]) > 0: continue
        if len(high_areas) > 0:
            for i in range(1,len(high_areas)):
                orig_str += ' - {}'.format(high_areas[i])
        neuron_to_motif[neuron_ids[idx][1]] = orig_str
        if orig_str not in motif_to_neuron.keys():
            motif_to_neuron[orig_str] = []
        motif_to_neuron[orig_str].append(neuron_ids[idx][1])

    motif_to_neuron2 = {}
    for key,val in motif_to_neuron.items():
        if len(val) < participation_thr: continue
        if len([v2 for v2 in exclude_list if v2 in key]) > 0: continue
        motif_to_neuron2[key] = len(val)

    motif_to_grad = {}
    motif_to_grad2 = {}
    for key in neuronid_to_grad.keys():
        new_key = braintell_2_nld(nld_list, key)
        if new_key not in neuron_to_motif.keys(): continue
        if neuron_to_motif[new_key] not in motif_to_grad.keys():
            motif_to_grad[neuron_to_motif[new_key]] = []
        motif_to_grad[neuron_to_motif[new_key]].append(neuronid_to_grad[key])

    for key,val in motif_to_grad.items():
        if key not in motif_to_neuron2.keys() or len(key) < participation_thr: continue
        if len([v2 for v2 in exclude_list if v2 in key]) > 0: continue
        motif_to_grad2[key] = np.round(np.median(val),2)


    motif_to_grad2_df = pd.DataFrame(motif_to_grad2, index = [0]).sort_values(by = [0], axis = 1, ascending = False)
    motif_to_neuron2_df = pd.DataFrame(motif_to_neuron2, index = [0])
    new_df3 = motif_to_neuron2_df.sort_values(by = [0], axis = 1, ascending = False).loc[0][motif_to_neuron2_df.loc[0] > 0]

    in_color_nrm = np.array(deepcopy(hex2rgb('#21918c')))/255
    in_color_nrm2 = np.array(deepcopy(hex2rgb('#A020F0')))/255
    diff_grad_nrm = list(neuronid_to_grad.values())
    unique_vals = len(np.unique(diff_grad_nrm))
    color_range = np.vstack((in_color_nrm + [0.1,0.1,0.1], in_color_nrm, in_color_nrm/2,
                             in_color_nrm2/2, in_color_nrm2,  in_color_nrm2 + [0.1,0.1,0.1]))
    color_range[color_range > 1] = 1
    cmap_fg = LinearSegmentedColormap.from_list('random', color_range, N = unique_vals)


    fig, ax = plt.subplots(figsize = (12,6))
    plt.title('Motif distributions')
    new_df3.plot.bar(stacked = True)
    if savefile is not None:
        plt.savefig(savefile.split('.svg')[0] + 'stacked.svg')
    plt.show()

    fig, ax = plt.subplots(figsize = (12,6))
    plt.title('Motif distributions ordered by avg gradient')

    motif_to_grad2_df.loc[0][motif_to_grad2_df.loc[0] > 0].plot.bar(color = [cmap_fg(i) for i in motif_to_grad2_df.loc[0][motif_to_grad2_df.loc[0] > 0]])
    if savefile is not None:
        plt.savefig(savefile.split('.svg')[0] + '_ordered_by_gradient.svg')
    plt.show()

    return motif_to_grad2_df, new_df3


def get_laminar_motifs(mesoscale_stats_df, mesoscale_stats_df_layered, source = 'VPM', savefile = None,
                       layer_acr = ['L1','L2/3','L4','L5','L6a','L6b'], target_areas = ['SSs', 'SSp-bfd', 'SSp-m', 'SSp-n']):

    proj_trg_laminar_dict = {}
    layers = ['1','2/3','4','5','6a','6b']

    for major_area in target_areas:
        layer_trg_patterns = {'primary': OrderedDict([(layer, 0) for layer in layer_acr]), 'secondary': OrderedDict([(layer, 0) for layer in layer_acr])}
        targets = mesoscale_stats_df.columns

        for fid in mesoscale_stats_df.loc[source].index:
            unique_trg_vals = [val if np.isnan(val) == False else -1 for val in mesoscale_stats_df.loc[source,fid].values]

            dom_proj_trg_idx = np.argmax(mesoscale_stats_df.loc[source,fid])
            dom_proj_trg_name = targets[dom_proj_trg_idx]
            if dom_proj_trg_name not in major_area: continue
            dom_proj_trg_layered = [dom_proj_trg_name + layer for layer in layers]

            layered_dist = mesoscale_stats_df_layered[dom_proj_trg_layered].loc[source,fid]
            layer_idx = np.argsort([val if np.isnan(val) == False else - 1 for val in layered_dist])[::-1]
            top_layer_refine = [layer_acr[val] for val in layer_idx]
            layer_trg_patterns['primary'][top_layer_refine[0]] += 1

            if len(unique_trg_vals) > 1:
                sec_proj_trg_idx = np.argsort(unique_trg_vals)[::-1][1]
                sec_proj_trg_name = targets[sec_proj_trg_idx]
                if sec_proj_trg_name not in target_areas: continue
                sec_proj_trg_layered = [sec_proj_trg_name + layer for layer in layers]

                sec_layered_dist = mesoscale_stats_df_layered[sec_proj_trg_layered].loc['VPM',fid]
                sec_layer_idx = np.argsort([val if np.isnan(val) == False else - 1 for val in sec_layered_dist])[::-1]
                sec_layer_refine = [layer_acr[val] for val in sec_layer_idx]
                layer_trg_patterns['secondary'][sec_layer_refine[0]] += 1

        layer_trg_df = pd.DataFrame(layer_trg_patterns)

        fig, ax = plt.subplots(figsize = (12,6))
        plt.title('Motif distributions for {}'.format(major_area))
        layer_trg_df.plot(kind='bar', ax=ax,  rot=0, stacked = False)
        if savefile is not None:
            plt.savefig('{}_{}.svg'.format(savefile, major_area), bbox_inches = 'tight')
        plt.show()

        proj_trg_laminar_dict[major_area] = layer_trg_df

    return proj_trg_laminar_dict


def findntotal(matrix,cutoff):

    # binarize input
    matrix_binary = matrix >= cutoff
    matrix_binary = matrix_binary[np.sum(matrix_binary,1)!=0,:]

    # find counts
    N_a = np.sum(matrix_binary,axis = 0)
    N_obs = np.shape(matrix_binary)[0]

    #calculate terms
    term5 = N_obs - np.sum(N_a)
    term5 = N_obs - np.sum(N_a)
    term4 = np.sum(np.prod(nchoosek(N_a,2),axis = 1))      #nchoosek(N_a,2),1))
    term3 = np.sum(np.prod(nchoosek(N_a,3),axis = 1))      #nchoosek(N_a,3),1))
    term2 = np.sum(np.prod(nchoosek(N_a,4),axis = 1))      #nchoosek(N_a,4),1))
    term1 = np.sum(np.prod(nchoosek(N_a,5),axis = 1))      #nchoosek(N_a,5),1))
    term0 = np.prod(N_a)

    # solve the polynomial
    p = [term5, term4, -1*term3, term2, -1*term1, term0]
    r = np.roots(p)

    #return the largest real root.
    N_t = np.max(np.real(r))

    return N_t

def Port_Han_Motif_Code(data, cutoff, alpha, area_labels = False, savefile = None):

    ## consider the set of neurons that project to two areas only. in this set are there certain classes/motifs
    # that are under or overrepresented relative to what we would expect if the
    # neurons in this set were distributed according to the product of the
    # first order innervation probabilities?

    # Binarize input matrix
#     data_binary_tmp = np.double(data >= cutoff)
#     data_binary = data_binary_tmp[np.sum(data_binary_tmp,1) != 0,:]
    data_binary = deepcopy(data)
    data_binary[data >= cutoff] = 1
    data_binary[data < cutoff] = 0
    targetnum = np.shape(data_binary)[1]

    # Uniq
    print(data_binary.shape)
    data_binary_uniq, ia, ib = np.unique(data_binary, return_index = True, return_inverse = True, axis = 0)
    data_counts = np.bincount(ib) #accumarray(ib,0)   - not sure about that port yet ...

    # get first order probabilities
    neuronnum = findntotal(data,cutoff) # get first order probabilities
    p = np.sum(data_binary,0)/neuronnum

    # Construct motif matrix and associated expected probabilities from first order stats

    # bifrucations
    motif_tmp = np.zeros((targetnum**2,targetnum))
    p_expected_tmp = np.zeros((targetnum**2,1))
    counter = 0
    for target1 in range(targetnum):
        for target2 in range(targetnum):
            motif_tmp[counter,target1] = 1
            motif_tmp[counter,target2] = 1
            p_expected_tmp[counter] = np.prod(p[motif_tmp[counter,:] == 1])*np.prod(1-p[motif_tmp[counter,:] != 1])
            counter = counter + 1

    # Remove doublecounts
    [motif_bi,ia,ib] = np.unique(motif_tmp[np.sum(motif_tmp,1)==2,:], return_index = True, return_inverse = True, axis = 0)
    p_expected_tmp2 = p_expected_tmp[np.sum(motif_tmp,1) == 2]
    p_expected_tmp3_bi = p_expected_tmp2[ia]

    # Trifrucations
    motif_tmp = np.zeros((targetnum**3,targetnum))
    p_expected_tmp = np.zeros((targetnum**3,1))
    counter = 0
    for target1 in range(targetnum):
        for target2 in range(targetnum):
            for target3 in range(targetnum):
                motif_tmp[counter,target1] = 1
                motif_tmp[counter,target2] = 1
                motif_tmp[counter,target3] = 1
                p_expected_tmp[counter] = np.prod(p[motif_tmp[counter,:] == 1])*np.prod(1-p[motif_tmp[counter,:] != 1])
                counter = counter + 1

    # remove doublecounts
    [motif_tri,ia,ib] = np.unique(motif_tmp[np.sum(motif_tmp,1)==3,:], return_index = True, return_inverse = True, axis = 0)
    p_expected_tmp2 = p_expected_tmp[np.sum(motif_tmp,1)==3]
    p_expected_tmp3_tri = p_expected_tmp2[ia]

    # quadfrucations
    motif_tmp = np.zeros((targetnum**4,targetnum))
    p_expected_tmp = np.zeros((targetnum**4,1))
    counter = 0
    for target1 in range(targetnum):
        for target2 in range(targetnum):
            for target3 in range(targetnum):
                for target4 in range(targetnum):
                    motif_tmp[counter,target1] = 1
                    motif_tmp[counter,target2] = 1
                    motif_tmp[counter,target3] = 1
                    motif_tmp[counter,target4] = 1
                    p_expected_tmp[counter] = np.prod(p[motif_tmp[counter,:] == 1])*np.prod(1-p[motif_tmp[counter,:] != 1])
                    counter = counter+1

    # remove doublecounts
    [motif_quad,ia,ib] = np.unique(motif_tmp[np.sum(motif_tmp,1) == 4,:], return_index = True, return_inverse = True, axis = 0)
    p_expected_tmp2 = p_expected_tmp[np.sum(motif_tmp,1) == 4]
    p_expected_tmp3_quad = p_expected_tmp2[ia]

    # combine all motifs and probabilities
    motif = np.concatenate((motif_bi, motif_tri, motif_quad),axis = 0)
    p_all = np.concatenate((p_expected_tmp3_bi, p_expected_tmp3_tri, p_expected_tmp3_quad),axis = 0)

    # count up the number of observed
    observed = np.zeros(np.shape(p_all))

    d,loc = ismember(motif,data_binary_uniq, 'rows')
    observed[d,0] = data_counts[loc[loc != 0]]
    observed = np.array([val for vals in observed for val in vals])

    # calculate expected
    expected = np.round(neuronnum*p_all).astype(int)
    expected = np.array([val for vals in expected for val in vals])

    # calculate pvalues
    #  pvalue = binocdf(observed,ceil(neuronnum),p_all)
    pvalue = binom.cdf(observed, np.ceil(neuronnum), p_all[:,0])
    pval_min = np.nan*np.ones(np.shape(pvalue))
    pval_min[pvalue < 0.5] = pvalue[pvalue < 0.5]
    pval_min[pvalue > 0.5] = 1 - pvalue[pvalue > 0.5]
    total_mask = np.where(np.logical_and(pval_min < 0.05,observed > 3))[0] #np.where(observed > 3)[0]
    sort_motifs = np.argsort(observed[total_mask])[::-1]

    # decide on significance using bonferoni correction
    motifnum = np.shape(motif[total_mask])[0]
    alpha_adj = alpha/motifnum
    overorunder = np.zeros(np.shape(p_all))
    overorunder[pvalue < (alpha_adj/2)] = -1
    overorunder[pvalue > (1-alpha_adj/2)] = 1
    significance = overorunder != 0

    fig = plt.figure(figsize=(16,6))

    ax = fig.add_subplot(211)
    index = np.arange(len(observed))
    data_df = pd.DataFrame(np.array([observed[total_mask][sort_motifs],expected[total_mask][sort_motifs]]).T, columns = ['observed','expected'])
    data_df.plot.bar(stacked = False, ax = ax)
    plt.ylabel('Motif count')
    plt.title('Observed and expected counts of polyfrucations')

    ax = fig.add_subplot(212)
    (x,y) = motif.T.shape
    p = sns.heatmap(1-motif.T[:,total_mask][:,sort_motifs], cbar = False)
    p.set_yticklabels(list(area_labels), rotation = 0, fontsize = 12)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches = 'tight')

    pval_min = pval_min[total_mask][sort_motifs]
    overorunder = overorunder[total_mask][sort_motifs]
    significance = significance[total_mask][sort_motifs]
    p_all = p_all[total_mask][sort_motifs]
    pvalue = pvalue[total_mask][sort_motifs]

#     plot volcano plot
#     plt.figure()
#     mavolcanoplot(observed,expected,pval_min,'LogTrans',1,'PlotOnly',1,'PCutoff',alpha_adj/2)

    return motif,pval_min,significance,overorunder,observed,expected,p_all,neuronnum, pvalue
