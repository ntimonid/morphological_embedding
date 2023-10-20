from imports import *
from utils import *

import convertAllenSpace as CAS


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
    lateromed_grad_src = [np.linalg.norm(val - root_pt) for val in soma_position[:,1:3]]
    lateromed_grad_src = 1- lateromed_grad_src/np.max(lateromed_grad_src)

    plt.scatter(lateromed_grad_src, grad_position)
    rho1,pval1 = sci.stats.spearmanr(lateromed_grad_src, grad_position)
    print(rho1,pval1)

    root_pt = np.min(target_position[:,[0,2]],axis = 0) #[695., 185.] #
    anterolat_grad_trg = [np.linalg.norm(val - root_pt) for val in target_position[:,[0,2]]]
    anterolat_grad_trg = 1 - anterolat_grad_trg/np.max(anterolat_grad_trg)

    plt.scatter(anterolat_grad_trg, grad_position)
    rho2, pval2 = sci.stats.spearmanr(lateromed_grad_src, grad_position)
    print(rho2,pval2)

    return lateromed_grad_src, anterolat_grad_trg, rho1, pval1, rho2, pval2
