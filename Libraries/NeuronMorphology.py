from imports import *
from utils import *

import convertAllenSpace as CAS

main_path = os.path.abspath('../../')

class NeuronMorphology:

  def __init__(self, neuronDict = None, *, neuronFile = None, orientation = None):

    if neuronFile is not None:
        neuronDict = self._load_file(neuronFile)
    if neuronDict is not None:
        try:
            self.metaData = neuronDict['metaData']
            if 'AA' in self.metaData['originalHeader']:
                self.mode = 'mouselight'
                self.in_color = '#FFFF00'
            else:
                self.mode = 'braintell'
                self.in_color = '#00FF00'
        except:
            self.metaData = None

        self.customTypes = neuronDict['customTypes']
        self.customProps = neuronDict['customProperties']
        self.lines = neuronDict['treeLines']['data']
        self.lineColumns = neuronDict['treeLines']['columns']
        self.points = numpy.array(neuronDict['treePoints']['data'])
        self.pointColumns = neuronDict['treePoints']['columns']
        self._partition_morpho()
        if 'filename' not in dir(self) and 'filename' in neuronDict.keys():
            self.filename = neuronDict['filename']
        elif 'filename' not in dir(self) and 'filename' not in neuronDict.keys():
            self.filename = None
        if 'orientation' in neuronDict.keys():
            self.orientation = neuronDict['orientation']
        elif orientation is not None:         # !!!New addition (08/02/2023)
            self.orientation = orientation
        else:
            if self.somaPointIdx is not None:      # !!!New addition (08/02/2023)
                self._find_orientation()
            else:
                self.orientation = None        # !!!New addition (08/02/2023)
    else:
        self.metaData = None
        self.customTypes = None
        self.customProps = None
        self.lines = None
        self.lineColumns = None
        self.points = None
        self.pointColumns = None

  def getType2id(self):
        type2id = { 'soma': 1, 'axon': 2,  'dendrite': 3, 'apical': 4 }
        # for geom,types in self.customTypes:
        #     for tp in types:
        #         type2id[tp.name] = tp.id
        return type2id

  def _load_file(self, neuronFile):

        # 03-08-2022: Update to store the type of morphology and its filename
        tokens = neuronFile.split('/')
        filename = tokens[len(tokens)-1]
        experimentId = filename.split('.')[0]
        neuronDict = None
        if 'AA' in experimentId:
            self.mode = 'mouselight'
            self.in_color = '#FFFF00'
        elif 'X' in experimentId and 'Y' in experimentId:
            self.mode = 'braintell'
            self.in_color = '#00FF00'
        else:
            self.mode = 'local'  # usually Clasca neurons
            self.in_color = '#0000FF'
        self.filename = tokens[len(tokens)-1]
        try:
            if '.swc' in neuronFile:
                neuronFile = swc2json(neuronFile, '.', main_path)  # convert to json first
            if '.json.gz' in neuronFile:
                with gzip.open(neuronFile, 'rb') as fp:
                    file_content = fp.read()
                    neuronDict = json.loads(file_content)
            elif '.json' in neuronFile:
                with open(neuronFile) as fp:
                    neuronDict = json.load(fp)
            self.fullpath = neuronFile
        except:
            if self.mode != 'local':
                neuronDict = DownloadAxons(experimentId, mode = self.mode)
                print('mpeee')
            else:
                neuronDict =  -1
            if neuronDict == - 1:  #The attempt to download the morphology has failed
                print('Please specify proper file path.')
                neuronDict = None

        return neuronDict

  def _micron_checker(self, somaCoord, orient = 'PIR'):
        if somaCoord[0] < 10 and somaCoord[1] < 10 and somaCoord[2] < 10:
            res = 'mm'
            unit = 1
        else:
            res = 'um'
            if orient == 'PIR':
                P,I,R = (528, 320, 456)
                unit = 1 if (somaCoord[0]>P or somaCoord[1]>I or somaCoord[2]>R) else 25
            elif orient == 'LIP':
                L,I,P = (450, 320, 528)
                unit = 1 if (somaCoord[0]>L or somaCoord[1]>I or somaCoord[2]>P) else 25

        return res,unit

  def _find_orientation(self):
        somaCoord = self.points[self.somaPointIdx]
        if self.mode == 'mouselight':
            orient = 'LIP'
            position = 'corner'
        elif self.mode == 'braintell':
            orient = 'PIR'
            position = 'corner'
        elif self.mode == 'local':
            orient = 'RAS'
            position = 'ac'
        scale,res = self._micron_checker(somaCoord, orient)
        self.orientation = ['{}({})'.format(scale,res),orient,position]

  def _partition_morpho(self):
        self.soma = []
        self.axons = []   # self.axon()
        self.dendrites = []  #self.dendrite()
        self.proximal = []   # self.proximal()
        self.distal = []    # self.distal()
        self.type_to_id = {'soma': 1, 'dendrites': 3, 'axons': 2, 'proximal': 4, 'distal': 5} # not sure about proximal and distal
        self.somaLineIdx = self.getSomaLineIdx()
        if self.somaLineIdx is not None:          # !!!New addition (08/02/2023)
            self.somaPointIdx = self.lines[self.somaLineIdx][1]
        else:
            self.somaPointIdx = None
        distances, branchingOrders = self.getPointStatistics()
        self.points = np.column_stack((np.array(self.points)[:,0:3],distances, branchingOrders))
        id_to_type = {int(value):key for key,value in self.type_to_id.items()} # reverse it for later

        for line in self.lines:
            firstPointIdx = line[1]
            # Warning !! The following lines are purely experimental
            for i in range(firstPointIdx,firstPointIdx+line[2]):  #-1
                if line[0] not in id_to_type.keys(): continue
                if id_to_type[line[0]] == 'soma':
                    self.soma = self.points[i]
                elif id_to_type[line[0]] == 'axons':
                    self.axons.append(self.points[i])
                # If I will switch to an 'axons' class, I shall do the folllowing ..
                # self.axons.points.append(points[i]); self.axons.lines.extend(line);
                # self.axons.branch_order.append(branchingOrders[i]); self.axons.distances.append(distances[i])
                elif id_to_type[line[0]] == 'dendrites':
                    self.dendrites.append(self.points[i])
                elif id_to_type[line[0]] == 'proximal':
                    self.proximal.append(self.points[i])
                elif id_to_type[line[0]] == 'distal':
                    self.distal.append(self.points[i])

  def transform(self, out_orientation):
    Q = CAS.convertAllenSpace(fromUnitOrientationOrigin = self.orientation, toUnitOrientationOrigin = out_orientation)
    self.points = (np.matmul(Q[0:3,0:3],self.points[:,0:3].T).T +  Q[0:3,3]).tolist()
    self.points = np.asarray(self.points)
    self.orientation = out_orientation
    self._partition_morpho()


  def get_anatomical_stats(self,annotation,id2acr, mode = 'terminals'):
    self.anatomical_stats = {}
    axonal_length = {}
    if mode == 'terminal branches':
        axonal_terminal_lines, axonal_terminal_points = self.get_axonal_terminals(annotation,id2acr, full_branch = True)
    if mode == 'terminals':
        axonal_terminal_lines, axonal_terminal_points = self.get_axonal_terminals(annotation,id2acr)
    elif mode == 'branches':
        axonal_terminal_lines, axonal_terminal_points = self.get_axonal_branches()
    elif mode == 'all':
        axonal_terminal_points = [idx for idx in range(len(self.points))]
    for idx,val in enumerate(axonal_terminal_points):
        pts = np.round(self.points[val][0:3]).astype(int)
        try:
            area = id2acr[annotation[pts[0],pts[1],pts[2]]]
        except:
            continue
        if area not in self.anatomical_stats:
            self.anatomical_stats[area] = [pts]
#             axonal_length[area] = [target_length[idx]]
        else:
            self.anatomical_stats[area].append(pts)
#             axonal_length[area].append(target_length[idx])

    for area in self.anatomical_stats:
        self.anatomical_stats[area] = np.array(self.anatomical_stats[area])
#         axonal_length[area] = np.array(axonal_length[area])

    return self.anatomical_stats

  def initObjectProperties(self):
    propertyAssignments = self.customProps['for']
    objectProperties = {}
    for index,assignment in enumerate(propertyAssignments):
      props = assignment['set']
      if 'objects' in assignment:
        for objId in assignment['objects']:
          if not objId in objectProperties:
            objectProperties[objId] = props.copy()
          else:
            objectProperties[objId].update(props)
    self._objectProps = objectProperties


  def getObjectProperties(self):
    if self._objectProps is None:
      self.initObjectProperties()
    return self._objectProps

  def getSelectedPointIndices(self, selectedTypes, minSampleDistance=0, minBranchOrder = 0):
    pointIndices = []
    id_to_type = {int(value):key for key,value in self.type_to_id.items()} # reverse it for later
    selectedTypes = [self.type_to_id[key] for key in selectedTypes]  # converting morpho type to an id encoded in swc
    distances,branchingOrders = self.getPointStatistics()
    for line in self.lines:
        if line[0] in selectedTypes:
            firstPointIdx = line[1]
            dst0 = distances[firstPointIdx]
            for i in range(firstPointIdx,firstPointIdx+line[2]-1):
                if branchingOrders[i] < minBranchOrder:
                    continue
                if distances[i]-dst0 > minSampleDistance:
                    pointIndices.append(i)
                    dst0 = distances[i]
            # always include last point of line (=terminal or near start of next line)
            pointIndices.append(firstPointIdx+line[2]-1)

    return pointIndices

  def find_soma_location(self, annotation, id2acr, res = 10):
    old_orient = self.orientation
    self.transform(['um({})'.format(res),'PIR','corner'])
    pos = np.round(self.soma[0:3]).astype(int)
    soma_area = id2acr[annotation[pos[0],pos[1],pos[2]]]
    self.transform(old_orient)
    return soma_area

  def subsampledCopy(self, selectedTypes, minDistance=0, skeleton = False):
        type2id = self.getType2id()
        if skeleton is True:
            term_lns,term_pts = self.get_axonal_terminals()
            line2children = self.getLine2children()
        selectedTypeIds = []
        for tp in enumerate(selectedTypes):
            if isinstance(tp,str):
                # convert to id
                selectedTypeIds.append(type2id[tp])
            else:
                # in case the selectedType is already specified as an integer, just use it
                selectedTypeIds.append(tp)

        distances,branchingOrders = self.getPointStatistics()
        newPoints = [[0,0,0,0,0]] # first point is ignored
        newLines = [[0,0,0,0,0]] # first line is ignored
        line2new = {0:0}
        for lineIdx,line in enumerate(self.lines):
            firstIncluded = 0
            numIncluded = 0
            if line[0] in selectedTypes:
                parentLineIdx = line[3]
                if parentLineIdx:
                    parentLine = self.lines[parentLineIdx]
                    # first point is the last point of the parent
                    prevPointIdx = parentLine[1]+parentLine[2]-1-parentLine[4]
                else:
                    # first point is the first point of the line if it has no parent
                    prevPointIdx = line[1]

                dst0 = distances[prevPointIdx]
                firstPointIdx = line[1]
                lastPointIdx = firstPointIdx+line[2]-1
                for i in range(firstPointIdx,lastPointIdx+1):
                    # always include last point of line (=terminal or branch point)
                    if distances[i]-dst0 > minDistance or i == lastPointIdx:
#                         if i != lastPointIdx:
#                             print("{}".format(distances[i]-dst0))
                        if not firstIncluded:
                            firstIncluded = len(newPoints)
                        numIncluded += 1
                        newPoints.append(self.points[i,:])
                        dst0 = distances[i]

                line2new[lineIdx] = len(newLines)
                newLines.append([line[0],firstIncluded,numIncluded,line[3],line[4]])

        # parent line field should use new line indices
        for line in newLines:
            if line[3] in line2new:
                line[3] = line2new[line[3]]
            else:
                line[3] = 0 # no parent

        return newLines, newPoints
        # return NeuronMorphology(neuronDict = dict(
        #   metaData = self.metaData,
        #   customTypes = self.customTypes,
        #   customProperties = {}, # these are no longer valid
        #   treeLines = dict(columns=self.lineColumns,data=newLines),
        #   treePoints = dict(columns=self.pointColumns,data=newPoints),
        #   orientation = self.orientation,
        #   filename = self.filename))

  def getLine2children(self, selectedTypes = None):
    line2children = {}
    if selectedTypes is not None:
        selectedTypes = [self.type_to_id[key] for key in selectedTypes]
    for lineIdx,line in enumerate(self.lines):
          if selectedTypes is not None and line[0] not in selectedTypes: continue
          (tp,firstPoint,numPoints,parentLineIdx,negOffset) = line
          if lineIdx != parentLineIdx:
                if not parentLineIdx in line2children:
                      line2children[parentLineIdx] = []
                line2children[parentLineIdx].append(lineIdx)
    return line2children


  def _propagateDistance(self,
                         distances,branchOrders,
                         parentLineIdx,line2children):
    if parentLineIdx not in line2children:
      return
    parentLine = self.lines[parentLineIdx]
    prevPointIdx = parentLine[1]+parentLine[2]-1-parentLine[4] # firstPoint+nPoints-1-negOffset
    lineIndices = line2children[parentLineIdx]
    branchOrder = branchOrders[prevPointIdx]
    if len(lineIndices)>1:
        branchOrder += 1
    for lineIdx in lineIndices:
      (tp,firstPointIdx,nPoints,parentLineIdx,negOffset) = self.lines[lineIdx]
      prevPoint = self.points[prevPointIdx][0:3]
      dst = distances[prevPointIdx]
      for p in range(firstPointIdx,firstPointIdx+nPoints):
        point = self.points[p][0:3]
        dst += numpy.linalg.norm(point-prevPoint)
        distances[p] = dst
        branchOrders[p] = branchOrder
        prevPoint = point

      self._propagateDistance(
        distances,branchOrders,
        lineIdx,line2children)

  def getSomaLineIdx(self):
    somaType = 1
    somaPointIdx = None
    somaLineIdx = None
    for lineIdx,line in enumerate(self.lines):
      if line[0] == somaType:
        somaPointIdx = line[1]
        somaLineIdx = lineIdx
        break
    return somaLineIdx

  def getPointStatistics(self):
    distances = numpy.zeros((len(self.points)))
    branchOrders = numpy.zeros((len(self.points)))
    somaLineIdx = self.getSomaLineIdx()

    line2children = self.getLine2children()
    self._propagateDistance(
      distances,branchOrders,
      somaLineIdx,line2children)

    self.distances = distances
    self.branchOrders = branchOrders
    return distances, branchOrders

  def get_axonal_terminals(self, annotation = None, id2acr = None, full_branch = False):
    line2children = self.getLine2children(['axons'])
    allkids = [value for val in line2children.values() for value in val]
    allparents = list(line2children.keys())
    terminal_lines = list(set(allkids).difference(set(allparents)))

    axonal_terminal_lines = [val for val in terminal_lines if self.lines[val][0] == 2]
    axonal_terminal_points = [self.lines[terminal_line][1] + self.lines[terminal_line][2]-1
                              for terminal_line in axonal_terminal_lines]

    if full_branch is True:
        axonal_terminal_points = [idx for terminal_line in axonal_terminal_lines
              for idx in range(self.lines[terminal_line][1]+1, self.lines[terminal_line][1] + self.lines[terminal_line][2])]

    axonal_terminal_length = []
    for terminal_line in axonal_terminal_lines:
        if annotation is None or id2acr is None: continue

        line = self.lines[terminal_line]
        lineType,firstPoint,numPoints,prevLineId,negOffset = line
        prevPoint = None
        if prevLineId:
            prevLine = self.lines[line[3]]
            prevPoint = self.points[prevLine[1]+prevLine[2]-1-line[4]]

        length_sum = 0
        x_term,y_term,z_term = np.round(self.points[line[1]+line[2]-1]).astype(int)[0:3]
        actualAcr = id2acr[annotation[x_term,y_term,z_term]]
        for pointId in range(line[1],line[1]+line[2]):
            point = self.points[pointId]
            if prevPoint is not None:
                length_sum += np.linalg.norm(point[0:3] - prevPoint[0:3])
                x,y,z = np.round(point).astype(int)[0:3]
#                 try:
                allenAcr = id2acr[annotation[x,y,z]]
                if allenAcr != actualAcr: continue
#                 except:
#                     continue
            prevPoint = point
        axonal_terminal_length.append(length_sum)

    return axonal_terminal_lines, axonal_terminal_points

  def get_axonal_branches(self):
    line2children = self.getLine2children(['axons'])
    allkids = [value for val in line2children.values() for value in val]
    allparents = list(line2children.keys())
    branch_lines = list(set(allkids).intersection(set(allparents)))

    axonal_branch_lines  = [val for val in branch_lines if self.lines[val][0] == 2]
    axonal_branch_points = [self.lines[self.lines[branch_line][3]][1] + self.lines[self.lines[branch_line][3]][2] -1\
                             for branch_line in axonal_branch_lines if self.lines[self.lines[branch_line][3]][0] == 2]


    return axonal_branch_lines, axonal_branch_points


  def compute_axonal_length(self,annotation, id2acr, terminals = False): # Or use a modified version of _propagateDistance where you don't add the distance over points
    axonal_length = {}
    if terminals is True:
        axonal_terminal_lines, axonal_terminal_points = self.get_axonal_terminals()
    Q_neuron2mm = CAS.convertAllenSpace(self.orientation, ['um','RAS','ac'])
    for lineId,line in enumerate(self.lines):
        if terminals is True:
            if lineId not in axonal_terminal_lines: continue
        lineType,firstPoint,numPoints,prevLineId,negOffset = line
        prevPoint_mm = None
        if prevLineId:
            prevLine = self.lines[line[3]]
            prevPoint = self.points[prevLine[1]+prevLine[2]-1-line[4]]
            prevPoint_mm = Q_neuron2mm[0:3,0:3] @ prevPoint[0:3] + Q_neuron2mm[0:3,3]
        for pointId in range(line[1],line[1]+line[2]): #-line[4]
            point = self.points[pointId]
            point_mm = Q_neuron2mm[0:3,0:3] @ point[0:3] + Q_neuron2mm[0:3,3]
            if prevPoint_mm is not None:
                s = np.linalg.norm(point_mm - prevPoint_mm)
                x,y,z = np.round(point).astype(int)[0:3]
                try:
                    allenAcr = id2acr[annotation[x,y,z]]
                    # cond no 2 would be: if 'SS' in allenAcr:
                except:
                    continue
                if allenAcr not in axonal_length.keys():
                    axonal_length[allenAcr] = 0
                axonal_length[allenAcr] += s
            prevPoint_mm = point_mm

    self.axonal_length = axonal_length
    return axonal_length

  def asDict(self):
        return dict(
          metaData = self.metaData,
          customTypes = self.customTypes,
          customProperties = self.customProps,
          treeLines = dict(columns=self.lineColumns,data=self.lines),
          treePoints = dict(columns=self.pointColumns,data=self.points.tolist())
  )


def get_morphometrics(input_morpho, neurite = 'all', plot = False):

    if isinstance(input_morpho, str):
        with gzip.open(input_morpho, 'rb') as fp:
            file_content = fp.read()
            neuronDict = json.loads(file_content)
    else:
        neuronDict = input_morpho
    with open('/tmp/temp.json', 'w') as fp:
          fp.write(json.dumps(neuronDict))
    fp.close()

    swc2json('/tmp/temp.json','./', main_path, to = 'swc')
    filename = 'temp.swc'
    N = fm.load_swc_file("/tmp/" + filename)

    if neurite == 'axons' or neurite == 'all':
        try:
            Axons = N.get_axonal_tree()
            A = Axons.get_topological_minor()
        except:
            return -1,-1,-1
        D = []
    if neurite == 'dendrites' or neurite == 'all':
        Dendrites = N.get_dendritic_tree()
        D = Dendrites.get_topological_minor()
        A = []

    if plot is True:
        if neurite == 'axons' or neurite == 'all':
            fig = plt.figure(figsize=(10,10))
            show_threeview(A, fig)
            plt.suptitle('Topological minor of the axons', weight='bold')
        if neurite == 'dendrites' or neurite == 'all':
            fig = plt.figure(figsize=(10,10))
            show_threeview(D, fig)
            plt.suptitle('Topological minor of the dendrites', weight='bold')

    morph_long = compute_morphometric_statistics(N, format='long')

    return A, D, morph_long


# #### List of possible morphometrics to try

# +
# A.get_strahler_order().values(),
# A.get_sholl_intersection_profile(),
# A.get_segment_length().values(),
# A.get_root_angles().values(),
# A.get_radial_distance().values(),
# A.get_path_length().values(),
# A.get_axon_nodes().values(),
# A.get_branch_order().values(),
# A.get_branch_angles().values(),
# A.get_cumulative_path_length().values()

# x1 = np.max(list(A.get_strahler_order().values()))
# x2 = np.mean(list(A.get_branch_angles().values()))
# x3 = np.max(list(A.get_path_length().values()))
# x4 = np.max(list(A.get_branch_order().values()))
