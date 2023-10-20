from imports import *

def name_2_fd(input_fname, acr_to_morpho_id, source_name, mouselight_dir, braintell_dir):
    name_tok = input_fname.split('_')
    if len(name_tok) > 1:
        tmp = [source for source in acr_to_morpho_id[source_name] if name_tok[0] in source[0]
               and name_tok[1] in source[0]][0]
        orig_fname = tmp[0]
    else:
        tmp = [source for source in acr_to_morpho_id[source_name] if input_fname in source[0]][0]
        orig_fname = tmp[0]
    # if 'AA' in orig_fname:
    #     file_id = os.path.join(mouselight_dir, orig_fname + '.json.gz')
    # else:
    #     file_id = os.path.join(braintell_dir, orig_fname + '.json.gz')
    file_id = orig_fname

    return file_id


def swc2json(morpho_fullpath, outdir, main_path, to = 'json'):

    node_js_dir = os.path.join(main_path,'Code\ Repositories', 'nodejs','morphology_io','index.js')

    tok = morpho_fullpath.split(' ')
    if len(tok) > 1:
        print('mpeeee')
        morpho_fullpath = tok[0] + '\ ' + tok[1]
    if to == 'json':
        outfile = morpho_fullpath.split('.swc')[0] + '.json'
    else:
        outfile = morpho_fullpath.split('.json')[0] + '.swc'
    outpath = os.path.join(outdir,outfile)
    get_ipython().system('node $node_js_dir --infile=$morpho_fullpath --outfile=$outpath   ')
    outpath = outpath.replace('\\','')

    return outpath

# %%
def braintell_2_nld(nld_list, morpho_id):
    if 'AA' in morpho_id:
        return morpho_id
    else:
        try:
            tmp = morpho_id.split('_reg')[0]
            matches =  [key for key in nld_list if tmp.split('_')[2] in key]
            return matches[0]
        except:
            return -1

def TopographicMapping(somata,target_centroids):

    somata_ext = np.append(somata,np.ones((len(somata),1)),axis=1)
    target_centroids_ext = np.append(target_centroids,np.ones((len(target_centroids),1)),axis=1)

    my_pinv = lambda x: np.dot(np.linalg.inv(np.dot(x.T,x)),x.T)

    A,t,rank,s = np.linalg.lstsq(somata_ext,target_centroids_ext,rcond=-1)
    A = A.T

    target_centroids_approx = np.matmul(A[0:3,0:3],somata_ext[:,0:3].T).T +  A[0:3,3]
    target_centroids_approx = np.round(target_centroids_approx).astype(int)
    mse = np.linalg.norm(target_centroids - target_centroids_approx)

    return A, target_centroids_approx, mse

def decompose_affine_matrix(H):

    T = H[0:3,3]

    L = H.copy()
    L[:3,3] = 0

    R, K = polar(L)

    if np.linalg.det(R) < 0:
        R[:3,:3] = -R[:3,:3]
        K[:3,:3] = -K[:3,:3]

    f, X = np.linalg.eig(K)

    S = []
    for factor, axis in zip(f, X.T):
        if not np.isclose(factor, 1):
            scale = np.eye(4) + np.outer(axis, axis) * (factor-1)
            S.append(scale)

    return R, T, S

def load_useful_data(data_repository, resolution = 10):

    try:
        [annotation,allenMeta] = nrrd.read('{}/annotation_{}.nrrd'.format(data_repository,resolution))
        [avg_template,tempMeta] = nrrd.read('{}/average_template_{}.nrrd'.format(data_repository,resolution))
        with open(os.path.join(data_repository, 'ccf3_acr2id.json')) as fp:
            acr2id = json.load(fp)
        with open(os.path.join(data_repository,'ancestorsById.json')) as fp:
            ancestorsById = json.load(fp)
    except:
        from allensdk.core.reference_space_cache import ReferenceSpaceCache
        reference_space_key = 'annotation/ccf_2017'
        rspc = ReferenceSpaceCache(10, reference_space_key, manifest='manifest.json')
        tree = rspc.get_structure_tree(structure_graph_id=1)
        annotation, allenMeta = rspc.get_annotation_volume()
        avg_template, tempMeta = rspc.get_template_volume()
        acr2id = tree.get_id_acronym_map()
        ancestorsById = tree.get_ancestor_id_map()
        annotation, allenMeta = rspc.get_annotation_volume()

    id2acr = { id:acr for acr,id in acr2id.items() }
    if 0 not in id2acr:
      acr2id['[background]'] = 0
      id2acr[0] = '[background]'

    # !! Warning: If I want to load exactly the data I used for the paper, I will need to change this one with the old acr2morpho_id file found at
    # /cortexdisk/data2/NestorRembrandtCollab/Data Repositories/mouse_connectivity/acr_to_morpho_id.json
    with open(os.path.join(data_repository,'acr_to_morpho_id_new.pkl'), 'rb') as infile:
        acr_to_morpho_id = pk.load(infile)

    neuriteLengthDistribution = {}
    databases = ['braintell','mouselight']

    for dbName in databases:
          with open(os.path.join(data_repository, 'neuriteLengthDistribution({}).json'.format(dbName))) as fp:
                neuriteLengthDistribution.update(json.load(fp))

    return annotation, avg_template, acr2id, id2acr, ancestorsById, neuriteLengthDistribution, acr_to_morpho_id

def VisualizeAxonalPairs(points1,points2,points3, lateromed_grad_src):

    pointSize = 2
    in_data = []
    go_scatter1 = go.Scatter3d(x=points1[:,0], y=points1[:,1], z=points1[:,2],
                          name= 'soma location',\
                          mode="markers",marker={"size":pointSize, "color": lateromed_grad_src, 'colorscale':'Blues'})#"color":"#0000FF"})
    go_scatter2 = go.Scatter3d(x=points2[:,0], y=points2[:,1], z=points2[:,2],
                            name='terminal centroid location (measured)',\
                            mode="markers",marker={"size":pointSize,"color":lateromed_grad_src, 'colorscale':'Greens'})

    go_scatter3 = go.Scatter3d(x = points3[:,0], y = points3[:,1], z = points3[:,2],
                            name = 'terminal centroid location (predicted)',\
                            mode = "markers",marker={"size":pointSize,"color": lateromed_grad_src, 'colorscale':'Reds'})
    in_data.append(go_scatter1); in_data.append(go_scatter2)
    in_data.append(go_scatter3)

    fig = go.Figure(data=in_data,
        layout=go.Layout(autosize=True))#,width=1200,height=1200))
    fig.update_layout(legend_title_text = "Segment")
    fig.show()


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)




def DownloadAxons(experimentId = None, mode = 'mouselight', rgb = '#FFBB00'): # Needs an update

    if mode == 'streamlines':
        save_path = os.path.join(main_path, 'Backup_Code/streamlines_tmp')
        infile = 'streamlines_{}.json.gz'.format(experimentId)
        streamline_link = 'https://neuroinformatics.nl/HBP/allen-connectivity-viewer/json/{}'.format(infile)
    elif mode == 'mouselight':
        save_path = os.path.join(main_path, 'Data Repositories/Mouselight/json')
        if type(experimentId) == int: # id not given directly
            experimentId += 1
            infile = 'AA{:04d}'.format(experimentId)
            experimentId = infile
        if '.json' not in experimentId:
            infile = '{}.json.gz'.format(experimentId)
        else:
            infile = experimentId
        streamline_link = 'https://neuroinformatics.nl/HBP/neuronsreunited-viewer/{}_json_gz/{}'.format(mode,infile)
    elif mode == 'braintell':
        save_path = os.path.join(main_path, 'Data Repositories/Braintell')
        braintell_link = 'https://neuroinformatics.nl/HBP/braintell-viewer/fname2soma.bas(sba.ABA_v3(mm,RAS,ac)).json'
        response = urllib.request.urlopen(braintell_link)
        file_content = response.read()
        out = json.loads(file_content)
        if type(experimentId) == int: # id not given directly
            cnt = 0
            for key in out.keys():
                if cnt == experimentId:
                    experimentId = key
                    break
                cnt+=1
            infile = '{}.json.gz'.format(experimentId)
        if '.json.gz' not in experimentId:
            infile = '{}.json.gz'.format(experimentId)
        else:
            infile = experimentId
        streamline_link = 'https://neuroinformatics.nl/HBP/neuronsreunited-viewer/{}_json_gz/{}'.format(mode,infile)

    infile2 = os.path.join(save_path,infile)
    print(infile, infile2, streamline_link)
    try:
        if os.path.exists(infile2) is False:
            wget.download(streamline_link,infile2)
        if '.json.gz' in infile2:
            with gzip.open(infile2, 'rb') as fp:
                file_content = fp.read()
        elif '.json' in infile2:
            with open(infile2, 'r') as fp:
                file_content = fp.read()
        out = json.loads(file_content)
    except:
        out = -1

    return out
