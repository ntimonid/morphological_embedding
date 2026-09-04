import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.cross_decomposition import CCA
from scipy.spatial.transform import Rotation as Rot
from geopy.distance import geodesic

import convertAllenSpace as CAS
from utils import braintell_2_nld, TopographicMapping, decompose_affine_matrix, VisualizeAxonalPairs


def get_source_and_target_arrays(neuronid_to_grad, neuropop_cls, nld_list, annotation, id2acr, target = 'SSp'):

    soma_position = []
    # soma_cords_dict = {}
    target_position = []
    grad_position = []

    annot_shape = annotation.shape
    for neuron_id,grad in neuronid_to_grad.items():
        neuron_id2 = braintell_2_nld(nld_list, neuron_id)

        soma_idx = np.where(np.array(list(neuropop_cls.targets_per_neuron.keys())) == neuron_id2)[0][0]
        soma_cords = np.array(neuropop_cls.somata[soma_idx]).astype(int)
        # soma_cords_dict[neuron_id2] = soma_cords
        if soma_cords[2] >= annot_shape[2]//2: continue

        cortical_sub = np.array([val for val in neuropop_cls.targets_per_neuron[neuron_id2] if target in id2acr[annotation[val[0],val[1],val[2]]]])
        if len(cortical_sub) == 0: continue
        kmedoids = KMedoids(n_clusters=1, random_state=0).fit(cortical_sub)
        cortical_centroid = np.array(kmedoids.cluster_centers_[0])

        soma_position.append(soma_cords)
        target_position.append(cortical_centroid)
        grad_position.append(grad)

    soma_position = np.asarray(soma_position)
    target_position = np.asarray(target_position)
    grad_position = np.asarray(grad_position)

    return soma_position, target_position, grad_position


def get_topopgraphic_rotations(soma_arr, target_arr):

    Rotats = {}; scales = {}; translates = {}
    X_trs_rot = {}

    Affine_grand, X_trs_aff, mse_aff = TopographicMapping(soma_arr, target_arr)
    R_grand,t,s = decompose_affine_matrix(Affine_grand)
    Rotats_grand = pd.DataFrame(np.degrees(R_grand[0:3,0:3]) ,index = ['A-P', 'S-I', 'L-R'], columns = ['A-P', 'S-I', 'L-R'])
    r = Rot.from_matrix(R_grand[0:3,0:3])

    print(mse_aff, Rot.from_matrix(R_grand[0:3,0:3]).as_quat())
    print(r.as_rotvec(), np.linalg.norm(r.as_rotvec())*(180/np.pi))
    print(r.as_euler('xyz')*(180/np.pi))
    print(Rotats_grand)

    return Affine_grand, X_trs_aff, r, Rotats_grand, mse_aff

# points1,points2,points3 = deepcopy(X_grand), deepcopy(Y_grand), deepcopy(X_trs_aff)
def get_topographic_plots(points1,points2,points3):

    root_pt = np.max(points1[:,1:3],axis = 0)

    lateromed_grad_src = [np.linalg.norm(val - root_pt) for val in points1[:,1:3]]
    lateromed_grad_src = lateromed_grad_src/np.max(lateromed_grad_src)

    Q_trs = CAS.convertAllenSpace(['um(10)','PIR','corner'],['um(10)','RAS','corner'])
    points1_trs = np.matmul(Q_trs[0:3,0:3],points1[:,0:3].T).T +  Q_trs[0:3,3]
    points2_trs = np.matmul(Q_trs[0:3,0:3],points2[:,0:3].T).T +  Q_trs[0:3,3]
    points3_trs = np.matmul(Q_trs[0:3,0:3],points3[:,0:3].T).T +  Q_trs[0:3,3]

    VisualizeAxonalPairs(points1_trs,points2_trs, points3_trs, lateromed_grad_src)

def align_topography_and_gradient(soma_position, target_position, grad_position):

    root_pt = np.max(soma_position[:,1:3],axis = 0)
    # Using geodesic for spatial gradient (simulated with a reference point for this implementation)
    lateromed_grad_src = [geodesic(val[1:3], root_pt).km for val in soma_position]
    lateromed_grad_src = 1 - lateromed_grad_src / np.max(lateromed_grad_src)

    plt.scatter(lateromed_grad_src, grad_position)
    rho1,pval1 = stats.spearmanr(lateromed_grad_src, grad_position)
    print(rho1,pval1)

    root_pt = np.min(target_position[:,[0,2]],axis = 0) #[695., 185.] #
    anterolat_grad_trg = [geodesic(val[[0,2]], root_pt).km for val in target_position]
    anterolat_grad_trg = 1 - anterolat_grad_trg/np.max(anterolat_grad_trg)

    plt.scatter(anterolat_grad_trg, grad_position)
    rho2, pval2 = stats.spearmanr(lateromed_grad_src, grad_position)
    print(rho2,pval2)

    return lateromed_grad_src, anterolat_grad_trg, rho1, pval1, rho2, pval2


def compute_cca_and_permutation(morpho_features: np.ndarray, spatial_coords: np.ndarray, n_permutations: int = 1000) -> Tuple[float, float]:
    """
    Compute Canonical Correlation Analysis (CCA) and perform Permutation Testing.

    :param morpho_features: N x M array of morphological features (or embedding coordinates)
    :param spatial_coords: N x 3 array of spatial coordinates
    :param n_permutations: number of shuffles for permutation test
    :return: (observed_correlation, p_value)
    """
    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(morpho_features, spatial_coords)
    observed_corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

    null_corrs = []
    for _ in range(n_permutations):
        shuffled_spatial = spatial_coords.copy()
        np.random.shuffle(shuffled_spatial)
        X_p, Y_p = cca.fit_transform(morpho_features, shuffled_spatial)
        null_corrs.append(np.corrcoef(X_p[:, 0], Y_p[:, 0])[0, 1])

    p_value = np.sum(np.array(null_corrs) >= observed_corr) / n_permutations
    return observed_corr, p_value
