import numpy
from PIL import Image
from matplotlib import pyplot as plt
import svgutils_transform as svgu
from matplotlib import cm
from IPython.display import SVG
from lxml import etree
import os, tempfile, subprocess
import re
import urllib.parse

def vectorizeLabelImage(labelImage,smoothness=0,bgColor='auto', in_path = None):
  curveTolerance = 1.0*smoothness;
  lineTolerance = 0.5*smoothness;
  # print('SMOOTHNESS',curveTolerance,lineTolerance)
  with tempfile.TemporaryDirectory() as tmpdir:
    os.makedirs(tmpdir,exist_ok=True)
    svgFile = os.path.join(tmpdir,'labelimage.svg')
    if in_path is None:
        in_path = os.getcwd()
    prog = os.path.abspath(os.path.join(in_path,'mindthegap/bin/mindthegap'))
    cmd = [prog,"-i",labelImage,"-o",svgFile,"-t",str(curveTolerance),"-s",str(lineTolerance)]
    if bgColor != 'auto':
      cmd.extend(('-c',bgColor))
      # print('USING BGCOLOR',bgColor,cmd)
    ans = subprocess.check_output(cmd, shell=False, stderr=subprocess.STDOUT)
    with open(svgFile,'rt') as fp:
      return fp.read()

def getSvgContours(labelImage,strokeColor='auto',strokeWidth=0.5,smoothness=0, rgb2id=None,id2acr=None,acr2full=None, in_path = None):
  parser = etree.XMLParser(remove_blank_text=True)
  if re.fullmatch('.*\.svg',labelImage):
    tree = etree.parse(labelImage,parser)
  else:
    im = Image.open(labelImage)
    im = im.convert('RGB')
    bgColor = "#{:02x}{:02x}{:02x}".format( *im.getpixel((0,0)) )
    svgString = vectorizeLabelImage(labelImage,smoothness,bgColor, in_path)
    tree = etree.fromstring(svgString,parser)
    # remove rectangle elements
    paths = tree.xpath('//*[name()="rect"]')
    for path in paths:
      path.getparent().remove(path)
    # unfill paths, use stroke instead
    paths = tree.xpath('//*[name()="path"]')
    rgbGroups = {}
    for p in paths:
      rgb = p.get('fill')
      if rgb == bgColor:
        p.getparent().remove(p)
      else:
        if rgb in rgbGroups:
          rgbGroups[rgb].append(p)
        else:
          rgbGroups[rgb] = [p]

        p.attrib.pop('fill')
        p.attrib.pop('stroke')
        p.attrib.pop('stroke-width')
    for rgb,paths in rgbGroups.items():
      rgb = rgb.lower()
      id = rgb2id[rgb] if (rgb2id and rgb in rgb2id) else ''
      acr = id2acr[id] if (id2acr and id in id2acr) else ''
      acr = urllib.parse.quote(acr)
      full = acr2full[acr] if (acr2full and acr in acr2full) else ''
      full = urllib.parse.quote(full)
      stroke = rgb if (strokeColor=='auto') else strokeColor
      g = etree.SubElement(tree, "g",{"id":"{},{},{},{}".format(rgb,id,acr,full),"fill":"none","stroke":stroke,"stroke-width":str(strokeWidth)})
      for p in paths:
        p.getparent().remove(p)
        g.append(p)

  svg = etree.tostring(tree).decode('utf-8')
  return svg

def matplot2svg(fig):
  # Create svg container figure
  sz = fig.get_size_inches()
  svgFig = svgu.SVGFigure('{}in'.format(sz[0]), '{}in'.format(sz[1]))
  svgFig.set_size(['{}in'.format(sz[0]), '{}in'.format(sz[1])])
  svg_layer = svgu.from_mpl(fig)
  svgFig.append(svg_layer.getroot())
  plt.close(fig)
  return svgFig

def getSize(svgFig):
  (width,height) = svgFig.get_size()
  if width[-2::] == 'in':
    width = numpy.round(72*float(width[:-2:]))
    height = numpy.round(72*float(height[:-2:]))
  else:
    width = int(width)
    height = int(height)
  return (width,height)

def atlasOverlay(svgFig,mplAxis,warpedAtlasSlice,strokeColor="auto",strokeWidth=0.5,contourSmoothness=1, in_path = None):
  figBox = mplAxis.get_position()
  (width,height) = getSize(svgFig)

  # save warpedAtlasSlice if supplied as ndarray
  if isinstance(warpedAtlasSlice,numpy.ndarray):
    tmpdir = tempfile.gettempdir()
    im = Image.fromarray(warpedAtlasSlice)
    warpedAtlasSlice = os.path.join(tmpdir,'warpedAtlasSlice.png')
    im.save(warpedAtlasSlice)

  # Load warpedAtlasSlice and detect size
  try:
    with Image.open(warpedAtlasSlice) as im:
      svgSize = [im.width,im.height]
    # Use the mind-the-gap algorithm to convert the image to svg contours
    svgString = getSvgContours(warpedAtlasSlice,strokeColor=strokeColor,strokeWidth=strokeWidth,smoothness=contourSmoothness, in_path = in_path)
  except:
    try:
      tree = etree.parse(warpedAtlasSlice)
    except:
      tree = etree.fromstring(warpedAtlasSlice)
    svgElems = tree.xpath('//*[name()="svg"]')
    attrs = svgElems[0].attrib
    svgSize = (int(attrs['width']),int(attrs['height']))
    svgString = etree.tostring(tree).decode('utf-8')

  svg_layer = svgu.fromstring( svgString )

  # Fit the svg contours to the specified figBox
  svg_layer.set_size([str(width),str(height)])
  root_layer = svg_layer.getroot()
  root_layer.moveto(width*figBox.x0+0.25,height*(1.0-figBox.y1)-0.25)
  axSize = [width*(figBox.x1-figBox.x0),height*(figBox.y1-figBox.y0)]
  root_layer.scale_xy(axSize[0]/svgSize[0],axSize[1]/svgSize[1])
  svgFig.append(root_layer)

def displaySvgFigure(svgFig):
  (width,height) = getSize(svgFig)
  svg = svgFig.to_str()
  tree = etree.fromstring(svg)
  paths = tree.xpath('//*[name()="svg"]')
  paths[0].set('viewBox','0 0 {} {}'.format(width,height))
  svg = etree.tostring(tree).decode('utf-8')
  display(SVG(svg))

def stringifySvgFigure(svgFig):
  svg = svgFig.to_str()
  svg = re.sub(r'\s*viewBox="[^"]+"','',svg.decode("utf-8"))
  return svg

def saveSvgFigure(svgFig,outputFile):
  with open(outputFile,'wt') as fp:
    fp.write( stringifySvgFigure(svgFig) )
