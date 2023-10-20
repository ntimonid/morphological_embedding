

import sys
import os
import json
import nrrd
import numpy
import urllib
import wget
import traceback
import uuid
import gzip
import zlib
import requests
import re
import copy

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import math as m
import scipy as sci
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as Rot
from ipdb import set_trace
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from geopy.distance import geodesic
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.stats import binom
from ismember import ismember
from itertools import combinations
from scipy.linalg import polar
from collections import OrderedDict
from copy import deepcopy
from base64 import b64encode
from json import dumps as json_encode
from IPython.core.display import display, HTML
from IPython.display import Javascript, clear_output #display,
from json import dumps as json_encode, loads as json_decode
from base64 import b64encode,b64decode
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
try:
    from urlparse import urljoin  # Python2
except ImportError:
    from urllib.parse import urljoin

from morphopy.computation import file_manager as fm
from morphopy.neurontree import NeuronTree as nt
from morphopy.neurontree.plotting import show_threeview
from morphopy.computation.feature_presentation import compute_morphometric_statistics
