import sys
import os
import re
import io
import base64
import tempfile
import nibabel,json
import cortical_map_extra as cm_new
import numpy as np
import matplotlib.pyplot as plt
import atlasoverlay as ao
from PIL import Image
from lxml import etree
import convertAllenSpace as CAS
import matplotlib.pyplot as plt
from copy import deepcopy


# img is 2d array of ids
def multiLabelSmooth(img,filter,ignore=[]):
  r = len(filter)//2
  sz = img.shape
  tp = img.dtype

  maximg = np.zeros(sz,float)
  newimg = np.zeros(sz,tp)
  regions = np.unique(img)
  for g in regions:
    if g in ignore: continue
    # np.dot: For N dimensions it is a sum product over the last axis of a and the second-to-last of b
    filteredslice0 = (img==g).astype(float)
    # filter over i-dimension
    filteredslice = np.zeros_like(filteredslice0)
    for i,coeff in enumerate(filter):
      d = i-r
      if d<0:
        filteredslice[-d:,:] += coeff*filteredslice0[:d,:]
      elif d>0:
        filteredslice[:-d,:] += coeff*filteredslice0[d:,:]
      else:
        filteredslice += coeff*filteredslice0
    filteredslice0 = filteredslice
    # filter over k-dimension
    filteredslice = np.zeros_like(filteredslice0)
    for k,coeff in enumerate(filter):
      d = k-r
      if d<0:
        filteredslice[:,-d:] += coeff*filteredslice0[:,:d]
      elif d>0:
        filteredslice[:,:-d] += coeff*filteredslice0[:,d:]
      else:
        filteredslice += coeff*filteredslice0
    maximg = np.maximum(maximg,filteredslice)
    newimg[np.logical_and(maximg==filteredslice,maximg>0)] = g
  return newimg


# expects SVG in which areas are grouped, and the group-id is formatted as "#rrggbb,id,acr"
def bestLabelPlacement(svgString):
  from svgpath2mpl import parse_path
  from shapely.geometry import Polygon
  from polylabel_rb import polylabel

  # Use lxml to read the paths from svg
  tree = etree.fromstring(svgString)
  # place labels as far as possible on the left
  svgElems = tree.xpath('//*[name()="svg"]')
  xMid = int(svgElems[0].attrib['width'])//2

  rgb2id = {}
  largestLeftCentroid = {}
  largestRightCentroid = {}
  gElems = tree.xpath('//*[name()="g"]')
  for elem in gElems:
    attrs = elem.attrib
    if 'id' in attrs:
      m = re.search(r'(#[\da-fA-F]+),(\d*),(.*)',attrs['id'])
      if m:
        rgb = m.group(1).lower()
        id = int(m.group(2))
        rgb2id[rgb] = id
        for ch in elem.iterchildren():
          path = parse_path(ch.get('d'))
          vertices = path._vertices
          poly = Polygon(vertices).buffer(5)
          centroid,radius = polylabel(poly)
          if centroid.x<=xMid:
            if id not in largestLeftCentroid or radius>largestLeftCentroid[id][1]:
              largestLeftCentroid[id] = ((centroid.x,centroid.y),radius)
          else:
            if id not in largestRightCentroid or radius>largestRightCentroid[id][1]:
              largestRightCentroid[id] = ((centroid.x,centroid.y),radius)

  # Place labels as far as possible in the left hemisphere
  largestCentroid = {}
  for rgb,id in rgb2id.items():
    cL = largestLeftCentroid[id] if id in largestLeftCentroid else None
    cR = largestRightCentroid[id] if id in largestRightCentroid else None
    if cL and cR:
      largestCentroid[id] = cL if cL[1]>0.8*cR[1] else cR
    elif cL or cR:
      largestCentroid[id] = cL if cL else cR

  return largestCentroid

def corticalProjection(projectionType,dataVolume,aggregateFunction,savePng=None,saveNifti=None):
  mapper = cortical_map.CorticalMap(projectionType,dataVolume.shape)
  proj = mapper.transform(dataVolume, agg_func = aggregateFunction)
  if savePng:
    im = Image.fromarray(proj.astype(np.uint8))
    im.save(savePng)
  if saveNifti:
    # for inspection with ITK-snap
    nii = nibabel.Nifti1Image(proj,np.eye(4))
    nibabel.save(nii,saveNifti)
  return proj

def selectiveCorticalProjection(projectionType,dataVolume,aggregateFunction, labelVolume,labelSelection,savePng=None,saveNifti=None):
  mapper = cortical_map.CorticalMap(projectionType,dataVolume.shape)
  proj = mapper.selectiveTransform(dataVolume,labelVolume,labelSelection, agg_func = aggregateFunction)
  if savePng:
    im = Image.fromarray(proj.astype(np.uint8))
    im.save(savePng)
  if saveNifti:
    # for inspection with ITK-snap
    nii = nibabel.Nifti1Image(proj,np.eye(4))
    nibabel.save(nii,saveNifti)
  return proj

# AGGREGATE FUNCTIONS
def layerMapFunc(regionsByLayer):
  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      hasLayer = [0,0,0,0,0]
      for id in nonzero:
        if id in regionsByLayer['Layer1']:
          hasLayer[0] = 1
        elif id in regionsByLayer['Layer2_3']:
          hasLayer[1] = 1
        elif id in regionsByLayer['Layer4']:
          hasLayer[2] = 1
        elif id in regionsByLayer['Layer5']:
          hasLayer[3] = 1
        elif id in regionsByLayer['Layer6']:
          hasLayer[4] = 1
      return hasLayer[0]+2*hasLayer[1]+4*hasLayer[2]+8*hasLayer[3]+16*hasLayer[4]
    return 0
  return AF

def nonlayerMapFunc(regionsByLayer):
  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      hasLayer = [0,0,0,0,0]
      for id in nonzero:
        if not (id in regionsByLayer['Layer1'] or id in regionsByLayer['Layer2_3'] or id in regionsByLayer['Layer4'] or id in regionsByLayer['Layer5'] or id in regionsByLayer['Layer6']):
          return id
    return 0
  return AF

def firstNonzeroFunc():
  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      return nonzero[0]
    return 0
  return AF

def firstElemFunc():
  def AF(arr):
    if len(arr):
      return arr[0]
    return 0
  return AF

def lastElemFunc():
  def AF(arr):
    if len(arr):
      return arr[-1]
    return 0
  return AF

def selectAreaFunc(ancestorsById,allowedParentIds):
  # Find out which areas have children
  hasChildren = set()
  for id,ancestors in ancestorsById.items():
    if len(ancestors) > 1:
      parent = ancestors[1]
      hasChildren.add(parent)

  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      x = []
      for id in nonzero:
        ancestors = ancestorsById[str(id)]
        ok = False
        for pid in allowedParentIds:
          if pid in ancestors:
            ok = True
        if ok:
          if ancestors[0] in hasChildren:
            x.append(ancestors[0])
          else:
            x.append(ancestors[1])
      if len(x):
        unique, counts = np.unique(x, return_counts=True)
        return unique[counts.argmax()]

    return 0

  return AF


def projectAnnotation(projectionType, annotationVolume, ancestorsById, allowedParentIds, savePng=None,saveNifti=None):
  idImage  = corticalProjection(projectionType,annotationVolume,selectAreaFunc(ancestorsById,allowedParentIds),savePng=savePng,saveNifti=saveNifti)
  return idImage


def contrastingColors(numColors):
  base = int(np.ceil(numColors**(1/3)))
  if base>256:
    raise RuntimeError( 'base ({}) must be 256 or below'.format(base) )
  base2 = base*base
  scaleup = 256 // base
  rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
  index2rgb = np.zeros(numColors,rgb_dtype)
  for index in range(0,numColors):
    index2rgb[index] = (
      (index % base) * scaleup,
      ((index % base2) // base) * scaleup,
      (index // base2) * scaleup
    )
  return index2rgb


def createAnnotationSvg(projectedAnnotation, id2acr, id2rgb=None, acr2full=None, strokeWidth=None, smoothness=None, mlFilter=None, saveNifti=None):
  if strokeWidth is None:
    strokeWidth = projectedAnnotation.shape[0]/300
  if smoothness is None:
    smoothness = projectedAnnotation.shape[0]/150

  # Convert array of id-values to rgb array that can be saved as an image
  def imagify(idVolume,id2rgb=None):
    unique = np.unique(idVolume)
    unique.sort()
    numColors = len(unique)
    indexVolume = np.zeros(idVolume.shape,np.uint8 if numColors<=256 else np.uint16 if numColors<=65536 else np.uint32)
    for index,id in enumerate(unique):
      indexVolume[idVolume==id] = index

    if id2rgb is None:
      index2rgb = contrastingColors(numColors)
    else:
      rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
      index2rgb = np.zeros(numColors,rgb_dtype)
      for id,rgb in id2rgb.items():
        index = np.where(unique==id)
        if len(index):
          index2rgb[index[0]] = tuple(int(rgb[i:i+2],16) for i in (1, 3, 5))
    rgb2id = { "#{:02x}{:02x}{:02x}".format(*rgb):unique[index] for index,rgb in enumerate(index2rgb) }

    rgbVolume = index2rgb[indexVolume]
    return rgbVolume,rgb2id

  idImage  = projectedAnnotation
  if mlFilter is not None:
    idImage = multiLabelSmooth(idImage,mlFilter)
    if saveNifti:
      nii = nibabel.Nifti1Image(idImage,np.eye(4))
      nibabel.save(nii,saveNifti.replace('.nii','_smooth({}).nii'.format(','.join([str(v) for v in mlFilter]))))

  backgroundId = idImage[0,0]
  rgbImage,rgb2id = imagify(idImage,id2rgb)
  with tempfile.TemporaryDirectory() as tmpdir:
    im = Image.fromarray(rgbImage.view(np.uint8).reshape(rgbImage.shape+(3,)))
    imageFile = os.path.join(tmpdir,'image.png')
    im.save(imageFile)
    svgString = ao.getSvgContours(imageFile, strokeColor = 'auto', strokeWidth=strokeWidth, smoothness=smoothness, rgb2id=rgb2id,id2acr=id2acr,acr2full=acr2full)

  labelCoords = bestLabelPlacement(svgString)
  fontSize_px = idImage.shape[0]/60
  s = ['<g id="area_labels" style="fill:#000; text-anchor: middle; dominant-baseline: middle; font-size:{}px; font-family: sans-serif">'.format(fontSize_px)]
  for id,coord in labelCoords.items():
    if id != backgroundId:
      s.append('<g><text stroke-width="{}" stroke="#666" x="{}" y="{}">{}</text>'.format(fontSize_px/10,coord[0][0],coord[0][1],id2acr[str(id)]))
      s.append('<text x="{}" y="{}">{}</text></g>'.format(coord[0][0],coord[0][1],id2acr[str(id)]))
  s.append('</g>')
  svgString = svgString[:-6]+"\n".join(s)+svgString[-6::]

  return svgString


def getAnnotationOverlay(projectionType,lineColor='#000', lineWidth='3',labelColor='#000',labelGlow='#AAA',
                         data_dir = './', annot = 'allen'):
    if annot == 'allen':
         annotationOverlay = os.path.join(data_dir,'annotation({},10).svg'.format(projectionType))
    else:
         annotationOverlay = os.path.join(data_dir,'YSK_annotation_smooth({},10).svg'.format(projectionType))

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(annotationOverlay,parser)
    print(tree)
    groups = tree.xpath('//*[name()="g"]')
    for g in groups:
        id = g.get('id')
        if id and id[0]=='#':
            g.set('stroke',lineColor)
            g.set('stroke-width',lineWidth)

    return etree.tostring(tree).decode('utf-8')

def image2svg(pngImage,rgbHex):
  if rgbHex[0] == '#': rgbHex = rgbHex[1:]
  r = int(rgbHex[0:2],16)
  g = int(rgbHex[2:4],16)
  b = int(rgbHex[4:6],16)
  im = Image.open(pngImage)
  data = np.array(im)
  rgba_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
  rgbImage = np.zeros(data.shape,dtype=rgba_dtype)
  rgbImage['R'] = r
  rgbImage['G'] = g
  rgbImage['B'] = b
  rgbImage['A'] = 255.9999*np.sqrt(data/256)
  im = Image.fromarray(rgbImage.view(np.uint8).reshape(rgbImage.shape+(4,)))
  pngBytes = io.BytesIO()
  im.save(pngBytes, format='PNG')

  imageSvg = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}"><image width="{}" height="{}" xlink:href="data:image/png;base64,{}"/></svg>""".format(im.width,im.height,im.width,im.height,base64.b64encode(pngBytes.getvalue()).decode('utf-8'))
  return imageSvg

def array2svg(data):
  rgba_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
  rgbImage = np.zeros(data.shape[0:2],dtype=rgba_dtype)
  opacity = np.zeros(data.shape[0:2], np.uint8)
  opacity[data[:,:,0]>0] = 255
  opacity[data[:,:,1]>0] = 255
  opacity[data[:,:,2]>0] = 255
  rgbImage['R'] = data[:,:,0]
  rgbImage['G'] = data[:,:,1]
  rgbImage['B'] = data[:,:,2]
  rgbImage['A'] = opacity
  im = Image.fromarray(rgbImage.view(np.uint8).reshape(rgbImage.shape+(4,)))
  pngBytes = io.BytesIO()
  im.save(pngBytes, format='PNG')

  imageSvg = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}"><image width="{}" height="{}" xlink:href="data:image/png;base64,{}"/></svg>""".format(im.width,im.height,im.width,im.height,base64.b64encode(pngBytes.getvalue()).decode('utf-8'))
  return imageSvg

# +
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from IPython.core.debugger import set_trace
hex2rgb = lambda x : tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def set_color(input_mat, in_color, cmap_bg):
    # one idea is to overlay two colourmaps
    max_input = np.max(input_mat)
    unique_vals = len(np.unique(input_mat))
    mask = input_mat > 0
    input_mat_rgb = cmap_bg(input_mat, bytes = True)

    # Create a colormap based on the input color ..
    if unique_vals == 2:
        if max_input > 0:   # There are labeled points in the cortex
            input_mat_rgb[:,:,0][mask] = in_color[0]
            input_mat_rgb[:,:,1][mask] = in_color[1]
            input_mat_rgb[:,:,2][mask] = in_color[2]
    else:
        in_color_nrm = np.array(in_color)/255
        color_range =  np.array([in_color_nrm/4, in_color_nrm/3, \
                        in_color_nrm/2, in_color_nrm])
#         color_range =  np.array([in_color_nrm/4, in_color_nrm/3 + [0.1,0.1,0.1], \
#                         in_color_nrm/2 + [0.2,0.2,0.2], in_color_nrm + [0.3,0.3,0.3]])
#         in_color_nrm = np.array((33, 145, 140))/255
#         in_color_nrm2 = np.array((160, 32, 240))/255
#         color_range = np.vstack((in_color_nrm, in_color_nrm, in_color_nrm/2,
#                                  in_color_nrm2/2, in_color_nrm2,  in_color_nrm2))

        color_range[color_range > 1] = 1
        cmap_fg = LinearSegmentedColormap.from_list('random', color_range, N = unique_vals)
        input_mat_fg = cmap_fg(input_mat, bytes = True)

        if max_input > 0:   # There are labeled points in the cortex
            input_mat_rgb[:,:,0][mask] = input_mat_fg[:,:,0][mask]
            input_mat_rgb[:,:,1][mask] = input_mat_fg[:,:,1][mask]
            input_mat_rgb[:,:,2][mask] = input_mat_fg[:,:,2][mask]

    return input_mat_rgb

def mix_set_color(input_mat1, input_mat2, in_color, in_cmap):

    mask = input_mat1 == np.max(input_mat1)
    input_mat1_rgb = in_cmap(input_mat1, bytes = True)
    input_mat2_rgb = in_cmap(input_mat2, bytes = True)
    mix_mat  = np.maximum(input_mat1_rgb, input_mat2_rgb)

    mix_mat[:,:,0][mask] = in_color[0]
    mix_mat[:,:,1][mask] = in_color[1]
    mix_mat[:,:,2][mask] = in_color[2]

    return mix_mat

def format_func(value, tick_number):
    return str(np.around(value * 0.001, 3))

def plot_flatmap(pd, proj_type = 'dorsal_flatmap', template = None, annot_file = None,
                 savefile = None, data_dir = './', annot = 'allen',
                 code_dir = './', flatmap_dir = '.', new_color=[[0,0,255],[0,255,0]], exception = False):

    if template is None:
        template10 = np.array(Image.open(os.path.join(data_dir, 'template_{}_gc.png'.format(proj_type))))
    else:
        template10 = template
    if annot_file is None:
        try:
            annotationOverlay = getAnnotationOverlay(proj_type, lineWidth='2', labelColor='#000', lineColor='#000',
                                                     labelGlow='#444', data_dir = data_dir, annot = annot)
        except:
            print('Error! Annotation overlay file is either not present or cannot be parsed.')
            # In a future version I should just generate the overlay from scratch here ...
            return -1
    else:
        annot_fname = annot_file.split('/')[len(annot_file.split('/'))-1]
        if annot_fname in os.listdir(data_dir):
            annotationOverlay = annot_file
        else:
            print('Error! Annotation overlay file is either not present or cannot be parsed.')
            # In a future version I should just generate the overlay from scratch here ...
            return -1

    in_path = os.path.join(code_dir,'atlasoverlay')

    #%% initialize colormaps for rgb conversion
    cmap_mask     = plt.get_cmap('gray')
    if len(np.shape(new_color)) == 1:
        new_color = [new_color]
    #%% transform projection density and nissl template
    print('TEMPLATESHAPE',template10.shape)
    if pd.shape[0] == template10.shape[0] and pd.shape[1] == template10.shape[1]: # custom flatmap inserted len(np.shape(pd)) == 2: #
        if len(np.shape(pd)) == 2:
            pd = np.reshape(pd,(pd.shape[0],pd.shape[1],1))
        for x in range(np.shape(pd)[2]):
            trs_pd      = deepcopy(pd[:,:,x])
            trs_pd_nrm  = trs_pd/(np.max(trs_pd)*1.0)
            if x > len(new_color) - 1:
                new_color.append(np.random.randint(0,high = 255,size = 3))
            rgb_pd  = set_color(trs_pd_nrm, new_color[x], cmap_mask)
            if x == 0:
                rgb_export  = deepcopy(rgb_pd)
                trs_old     = deepcopy(trs_pd)
            else:
                rgb_export[trs_pd > trs_old] = rgb_pd[trs_pd > trs_old]
                if exception is True: # New (28/02/2023): Remove fast!!!
                    rgb_export[np.logical_and(trs_pd!=0, trs_old!=0)] = [new_color[0][0],new_color[0][1],new_color[0][2],255]
                trs_old     = np.maximum(trs_pd, trs_old)
    else: # 3D matrix inserted, need to create flatmap
        templateMapper = cm_new.CorticalMap(proj_type,template10.shape, flatmap_dir)
        if len(np.shape(pd)) == 3:
            pd = np.reshape(pd,(pd.shape[0],pd.shape[1],pd.shape[2],1))
        for x in range(np.shape(pd)[3]):
            trs_pd        = templateMapper.transform(pd[:,:,:,x], agg_func = np.max)
            trs_pd_nrm    = trs_pd/(np.max(trs_pd)*1.0)
            if x > len(new_color) - 1:
                new_color.append(np.random.randint(0,high = 255,size = 3))
            rgb_pd = set_color(trs_pd_nrm, new_color[x], cmap_mask)
            if x == 0:
                rgb_export  = deepcopy(rgb_pd)
                trs_old     = deepcopy(trs_pd)
            else:
                # 28/02/2023: New technique for mixing colours: directly taking the indices where the new matrix has
                # higher values than the previous one, instead of mixing them
                rgb_export[trs_pd > trs_old] = rgb_pd[trs_pd > trs_old]
                if exception is True: # New (28/02/2023): Remove fast!!!
                    rgb_export[np.logical_and(trs_pd!=0, trs_old!=0)] = [new_color[0][0],new_color[0][1],new_color[0][2],255]
                trs_old     = np.maximum(trs_pd, trs_old)

    fig = plt.figure(figsize=(template10.shape[1]/240,template10.shape[0]/240))
    # fig = plt.figure(figsize=(template10.shape[1]/120/2,template10.shape[0]/120/2))
    ax = plt.axes([0,0,1,1])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    bgImage = template10
    pos = ax.imshow(bgImage,plt.get_cmap('gray'))
    plt.axis('off')
    svgFig = ao.matplot2svg(fig)
    print(rgb_export.shape)
    print(template10.shape)
    svg = array2svg(rgb_export)
    ao.atlasOverlay(svgFig, ax, svg, in_path = in_path)
    ao.atlasOverlay(svgFig, ax, annotationOverlay, in_path = in_path)
    ao.displaySvgFigure(svgFig)
    svg = ao.stringifySvgFigure(svgFig)
    if savefile is not None:
        with open(savefile.split('.')[0]+'.svg','wt') as fp:
            fp.write(svg)
    return rgb_export

# +
def plot_plane(point_cloud, annotation, template, acr2id, source_area, in_color = [[0,0,255],[0,255,0]], code_dir = './',
                sel_axis = 'IR', orient = 'left', savefile = None, style = 'max', section = None):
    # Three potential flatmapping strategies:
    # 'max': taking the maximum intensity of everything including anatomical borders
    # 'median': taking maximum intensity only for the source border, and overlaying it with the middle anatomical slice for the rest
    # 'solo': taking only the source border into account, and not including any other border
    cmap_pd  = plt.get_cmap('hot')
    cmap_gr  = plt.get_cmap('gray')
    in_path = os.path.join(code_dir,'atlasoverlay')
    # import atlasoverlay.atlasoverlay as ao

    if len(np.shape(in_color)) == 1:
        in_color = [in_color]
    if len(np.shape(point_cloud)) == 3:
        point_cloud = np.reshape(point_cloud,(point_cloud.shape[0],point_cloud.shape[1],point_cloud.shape[2],1))
    cnt = 0
    for x in range(np.shape(point_cloud)[3]):
        if style == 'slice':
            input_plane, plane_annotation, plane_intensity = Slice_Maker(point_cloud[:,:,:,x],
                                                                          annotation, template, acr2id,
                                                                          sel_axis, section = section)
        else:
            input_plane, plane_annotation, plane_intensity, return_coos = Subcortical_Map(point_cloud[:,:,:,x], annotation,
                                                                                          template, acr2id, source_area,
                                                                                          sel_axis, orient, style,
                                                                                          section = section)
        input_plane_nrm = input_plane/(np.max(input_plane)*1.0)
        if x > len(in_color)-1:
            in_color.append(np.random.randint(0,high = 255,size = 3))
        try:
            if input_plane == -1:
                continue
        except:
            a = 1

#         plane_mix = mix_set_color(input_plane, plane_intensity, in_color[x], cmap_gr)
        plane_mix = set_color(input_plane_nrm, in_color[x], cmap_gr)
        if x == 0 or cnt == 0:
            rgb_export = plane_mix
            input_plane_old = deepcopy(input_plane)
            cnt +=1
        else:
#             rgb_export[input_plane > input_plane_old] = [  0,   0,   0, 0]
#             rgb_export = np.maximum(rgb_export, plane_mix)
            rgb_export[input_plane > input_plane_old] = plane_mix[input_plane > input_plane_old]
            input_plane_old = np.maximum(input_plane, input_plane_old)
    # input_plane = soma_volume #[plane_to_id[sel_axis]]
    # plane_annotation = area_volume #[plane_to_id[sel_axis]]
    # plane_intensity = area_intensity #[plane_to_id[sel_axis]]

    fig = plt.figure(figsize=(15, 6))
    ax1   = plt.axes()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    pos = ax1.imshow(plane_intensity, cmap_gr)
#     plt.axis('off')
    svgFig = ao.matplot2svg(fig)
    svg = array2svg(rgb_export)
    ao.atlasOverlay(svgFig, ax1, svg, in_path = in_path)
    ao.atlasOverlay(svgFig, ax1, plane_annotation, in_path = in_path, strokeWidth=1, contourSmoothness=1)
    ao.displaySvgFigure(svgFig)

    if savefile is not None:
        ao.saveSvgFigure(svgFig, savefile.split('.')[0] + '.svg')

    return input_plane, plane_annotation, plane_intensity #, return_coos


# + active=""
# def Subcortical_Map(point_cloud, annotation, template, acr2id, area_name = 'VPM', sel_axis = 'IR', orient = 'left',
#                     style = 'max'):
#
#     plane_to_id = {'PI': 2,'IR': 0,'RP': 1}
#
#     area_mask = annotation==acr2id[area_name] # give me VPM for instance
#     area_loci = np.where(area_mask)
#     x_min,x_max = np.min(area_loci[0]), np.max(area_loci[0])
#     y_min,y_max = np.min(area_loci[1]), np.max(area_loci[1])
#     z_min,z_max = np.min(area_loci[2]), np.max(area_loci[2])
#     if orient == 'left':
#         z_midpoint = z_min + int(np.round(1*(z_max-z_min)/3)) -1   # taking the midth along the left-right axis
#         z_max = z_midpoint
#     elif orient == 'right':
#         z_midpoint = z_min + int(np.round(3*(z_max-z_min)/4)) -1   # taking the midth along the left-right axis
#         z_min = z_midpoint
#         z_max = z_max - 1
#     return_coos = (x_min, x_max, y_min,y_max,z_min,z_max)
#     area_volume = deepcopy(annotation[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])
#     area_intensity = deepcopy(template[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])
#
#     # soma_volume_specific = deepcopy(soma_volume[x_min:x_max+1,y_min:y_max+1,z_min:z_min+z_max+1])
#     if len(np.shape(point_cloud)) == 2: # a 3D point-cloud is given
#         somata_reorient = point_cloud - [x_min,y_min,z_min]
#         soma_volume_specific = np.zeros(np.shape(area_volume), dtype = 'uint32')
#         soma_volume_specific[somata_reorient[:,0],somata_reorient[:,1],somata_reorient[:,2]] = 1
#     elif len(np.shape(point_cloud)) == 3: # a 3D array is given
# #         soma_volume_specific = deepcopy(point_cloud[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])
#         soma_volume_specific = deepcopy(point_cloud[return_coos[0]:return_coos[1]+1,return_coos[2]:return_coos[3]+1,\
#                                                     return_coos[4]:return_coos[5]+1])
#         soma_volume_specific[area_volume != acr2id[area_name]] = 0
#
#     soma_volume_cords = np.nonzero(soma_volume_specific)
#     if len([val for val in soma_volume_specific[soma_volume_cords] if val > 0 and val < 1]) > 0:
#           soma_volume_specific *= 100
#
#     x_midpoint = int(np.round((x_max-x_min)/2))-1   # taking the midth along the left-right axis
#     y_midpoint = int(np.round((y_max-y_min)/2))-1   # taking the midth along the left-right axis
#     z_midpoint = int(np.round((z_max-z_min)/2))-1   # taking the midth along the left-right axis
#
#     #************
#     max_val = np.max(area_volume)*2
#     area_volume[area_volume == acr2id[area_name]] = max_val
#
#     area_volume_clone = deepcopy(area_volume)
#     if style == 'solo' or style == 'median':
#         area_volume[area_volume != max_val] = 0
#     if style == 'median':           #{'PI': 2,'IR': 0,'RP': 1}
#         if plane_to_id[sel_axis] == 0:
#             area_volume_mid = area_volume_clone[x_midpoint,:,:]
#         elif plane_to_id[sel_axis] == 1:
#             area_volume_mid = area_volume_clone[:,y_midpoint,:]
#         elif plane_to_id[sel_axis] == 2:
#             area_volume_mid = area_volume_clone[:,:,z_midpoint]
#     #***********
#     area_volume_tmp = np.max(area_volume, axis = plane_to_id[sel_axis])
#     if style == 'median':
#         area_volume_mid[area_volume_tmp==max_val] = max_val
#         area_volume_tmp = area_volume_mid
#
#     unique_annots = np.unique(area_volume_tmp)
#     for idx,annot in enumerate(unique_annots):
#         area_volume_tmp[area_volume_tmp==annot] = (idx + 1)*4
#
#     area_volume = np.array(area_volume_tmp, dtype = 'uint8')
#     # area_volume %= 100 #[area_volume > 100] %= 100
#     area_intensity = np.array(np.max(area_intensity, axis = plane_to_id[sel_axis]), dtype = 'uint32') #area_intensity[:,:,z_midpoint]
#
#     if len(soma_volume_cords[0]) == 0:
#         print('Volume is empty. Please provide new data.')
#         soma_volume = np.zeros(area_intensity.shape, dtype = 'uint8')
#     else:
#         soma_volume = np.array(np.max(soma_volume_specific, axis = plane_to_id[sel_axis]), dtype = 'uint32')*355
#
#     # maybe return_coos too
#     return soma_volume, area_volume, area_intensity, return_coos
# -

def Subcortical_Map(point_cloud, annotation, template, acr2id, area_name = 'VPM', sel_axis = 'IR', orient = 'left',
                    style = 'max', section = None):

    plane_to_id = {'PI': 2,'IR': 0,'RP': 1}

    area_mask = annotation==acr2id[area_name] # give me VPM for instance
    area_loci = np.where(area_mask)
    x_min,x_max = np.min(area_loci[0]), np.max(area_loci[0])
    y_min,y_max = np.min(area_loci[1]), np.max(area_loci[1])
    z_min,z_max = np.min(area_loci[2]), np.max(area_loci[2])
    if orient == 'left':
        z_midpoint = z_min + int(np.round(1*(z_max-z_min)/3)) -1   # taking the midth along the left-right axis
        z_max = z_midpoint
    elif orient == 'right':
        z_midpoint = z_min + int(np.round(3*(z_max-z_min)/4)) -1   # taking the midth along the left-right axis
        z_min = z_midpoint
        z_max = z_max - 1
    return_coos = (x_min, x_max, y_min,y_max,z_min,z_max)
    area_volume = deepcopy(annotation[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])
    area_intensity = deepcopy(template[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])

    # soma_volume_specific = deepcopy(soma_volume[x_min:x_max+1,y_min:y_max+1,z_min:z_min+z_max+1])
    if len(np.shape(point_cloud)) == 2: # a 3D point-cloud is given
        somata_reorient = point_cloud - [x_min,y_min,z_min]
        soma_volume_specific = np.zeros(np.shape(area_volume), dtype = 'uint32')
        soma_volume_specific[somata_reorient[:,0],somata_reorient[:,1],somata_reorient[:,2]] = 1
    elif len(np.shape(point_cloud)) == 3: # a 3D array is given
        soma_volume_specific = deepcopy(point_cloud[return_coos[0]:return_coos[1]+1,return_coos[2]:return_coos[3]+1,\
                                                    return_coos[4]:return_coos[5]+1])
        # Warning!!! Temporarily remove ...
        # soma_volume_specific[area_volume != acr2id[area_name]] = 0

    soma_volume_cords = np.nonzero(soma_volume_specific)
    if len([val for val in soma_volume_specific[soma_volume_cords] if val > 0 and val < 1]) > 0:
          soma_volume_specific *= 100

    x_midpoint = int(np.round((x_max-x_min)/2))-1   # taking the midth along the left-right axis
    y_midpoint = int(np.round((y_max-y_min)/2))-1   # taking the midth along the left-right axis
    z_midpoint = int(np.round((z_max-z_min)/2))-1   # taking the midth along the left-right axis

    #************
    max_val = np.max(area_volume)*2
    area_volume[area_volume == acr2id[area_name]] = max_val

    area_volume_clone = deepcopy(area_volume)

    if style == 'solo' or style == 'median':
        area_volume[area_volume != max_val] = 0
    if style == 'median':
        if plane_to_id[sel_axis] == 0:
            area_volume_mid = area_volume_clone[x_midpoint,:,:]
        elif plane_to_id[sel_axis] == 1:
            area_volume_mid = area_volume_clone[:,y_midpoint,:]
        elif plane_to_id[sel_axis] == 2:
            area_volume_mid = area_volume_clone[:,:,z_midpoint]
    elif style == 'max' and section is not None:
        if plane_to_id[sel_axis] == 0:
            area_volume_tmp = area_volume_clone[section - x_min,:,:]
            soma_volume = soma_volume_specific[section - x_min,:,:]
            area_intensity = area_intensity[section - x_min,:,:]
        elif plane_to_id[sel_axis] == 1:
            area_volume_tmp = area_volume_clone[:,section - y_min,:]
            soma_volume = soma_volume_specific[:,section - y_min,:]
            area_intensity = area_intensity[:,section - y_min,:]
        elif plane_to_id[sel_axis] == 2:
            area_volume_tmp = area_volume_clone[:,:,section - z_min]
            soma_volume = soma_volume_specific[:,:,section - z_min]
            area_intensity = area_intensity[:,:,section - z_min]
    #***********
    if section is None:
        area_volume_tmp = np.max(area_volume, axis = plane_to_id[sel_axis])
        area_intensity = np.max(area_intensity, axis = plane_to_id[sel_axis])
    if style == 'median':
        area_volume_mid[area_volume_tmp==max_val] = max_val
        area_volume_tmp = area_volume_mid

    unique_annots = np.unique(area_volume_tmp)
    for idx,annot in enumerate(unique_annots):
        area_volume_tmp[area_volume_tmp==annot] = (idx + 1)*4

    area_volume = np.array(area_volume_tmp, dtype = 'uint8')
    area_intensity = np.array(area_intensity, dtype = 'uint8')

    if len(soma_volume_cords[0]) == 0:
        print('Volume is empty. Please provide new data.')
        soma_volume = np.zeros(area_intensity.shape, dtype = 'uint8')
    else:
        if section is None:
            soma_volume = np.array(np.max(soma_volume_specific, axis = plane_to_id[sel_axis]), dtype = 'uint32')*355

    # maybe return_coos too
    return soma_volume, area_volume, area_intensity, return_coos


def Slice_Maker(label_volume_specific, annotation, template, acr2id, sel_axis = 'IR', section = None):

    plane_to_id = {'PI': 2,'IR': 0,'RP': 1}

    return_coos = (0,annotation.shape[0]-1,0,annotation.shape[1]-1,0,annotation.shape[2]-1)
    x_min,x_max = return_coos[0],return_coos[1]
    y_min,y_max = return_coos[2],return_coos[3]
    z_min,z_max = return_coos[4],return_coos[5]

    area_volume_clone = deepcopy(annotation)
    area_intensity_clone = deepcopy(template)
    label_volume_clone = deepcopy(label_volume_specific)
    label_volume_cords = np.nonzero(label_volume_specific)

    if len(label_volume_cords[0]) == 0:
        print('Volume is empty. Please provide new data.')
#         return -1,-1,-1,-1

    x_midpoint = int(np.round((x_max-x_min)/2))-1   # taking the midth along the left-right axis
    y_midpoint = int(np.round((y_max-y_min)/2))-1   # taking the midth along the left-right axis
    z_midpoint = int(np.round((z_max-z_min)/2))-1   # taking the midth along the left-right axis

    if plane_to_id[sel_axis] == 0:
        if section is None:
            area_volume_mid = area_volume_clone[x_midpoint,:,:]
            area_intensity_mid = area_intensity_clone[x_midpoint,:,:]
            soma_volume_mid = label_volume_clone[x_midpoint,:,:]
        else:
            area_volume_mid = area_volume_clone[section,:,:]
            area_intensity_mid = area_intensity_clone[section,:,:]
            soma_volume_mid = label_volume_clone[section,:,:]
    elif plane_to_id[sel_axis] == 1:
        if section is None:
            area_volume_mid = area_volume_clone[:,y_midpoint,:]
            area_intensity_mid = area_intensity_clone[:,y_midpoint,:]
            soma_volume_mid = label_volume_clone[:,y_midpoint,:]
        else:
            area_volume_mid = area_volume_clone[:,section,:]
            area_intensity_mid = area_intensity_clone[:,section,:]
            soma_volume_mid = label_volume_clone[:,section,:]
    elif plane_to_id[sel_axis] == 2:
        if section is None:
            area_volume_mid = area_volume_clone[:,:,z_midpoint]
            area_intensity_mid = area_intensity_clone[:,:,z_midpoint]
            soma_volume_mid = label_volume_clone[:,:,z_midpoint]
        else:
            area_volume_mid = area_volume_clone[:,:,section]
            area_intensity_mid = area_intensity_clone[:,:,section]
            soma_volume_mid = label_volume_clone[:,:,section]

    unique_annots = np.unique(area_volume_mid)
    for idx,annot in enumerate(unique_annots):
        area_volume_mid[area_volume_mid==annot] = (idx + 1)#*5

    cmap = plt.get_cmap('gray')
    area_volume = cmap(area_volume_mid, bytes = True)
    area_volume = np.array(area_volume_mid, dtype = 'uint8')
    area_intensity = np.array(area_intensity_mid, dtype = 'uint32')
    if section is None:
        label_volume = np.array(np.max(label_volume_clone, axis = plane_to_id[sel_axis]), dtype = 'uint32')*355
    else:
        label_volume = np.array(soma_volume_mid, dtype = 'uint32')*355

    return label_volume, area_volume, area_intensity
