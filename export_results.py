# -*- coding: utf-8 -*-
"""
Fonction pour créer des exports des résults spour Apply_GSCNN 
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import transforms.transforms as extended_transforms
import torchvision.transforms as standard_transforms
import torch
import gdown
import gdal

import visualize
import apply
from train import args
import time

def plot_graphic(image, cmap = 'Paired', vmin = False): 
  plt.figure(figsize=(20, 10), dpi=80)
  lut = len(np.unique(image))
  if lut>30:
    cmap = cm.get_cmap(cmap, lut=lut)
  if type(vmin) is tuple:
    plt.imshow(np.array(np.flipud(image)), cmap=cmap, vmin=vmin[0], vmax=vmin[1])
  else : 
    plt.imshow(np.array(np.flipud(image)), cmap=cmap)
  plt.axis('off')
  plt.show()
  

#emplacement ou mettre les fichiers temporaires
# path_s = '/content/gdrive/My Drive/mira_project'

def numpy_pil(img, flip=False,norm=True):
  """
  change numpy to pil for the network with normlaization
  in : img
  out : img PIL
  """
  if type(norm) is not tuple: 
    img = (img-img.min())/(img.max()-img.min())
  else:
    img = (img-norm[0])/(norm[1]-norm[0])

  img = (img[:,:]*255).astype(np.uint8)
  
  if flip:
    img = np.flipud(img)
  
  img = Image.fromarray(img)

  return img

def transform_im(img, mean_std = ([0.157, 0.157, 0.157], [0.106, 0.106, 0.106])):
  """
  normlalized pil to mean_normalized torch  
  in : img (pil normalized)
       means_std : ([0.157, 0.157, 0.157], [0.106, 0.106, 0.106])
  out : img torch shape[2,3,x,y]
  """

  trans = standard_transforms.Compose([extended_transforms.ToRGB(),
                                      standard_transforms.ToTensor(), #change PIL image H*X*C to torch C*H*W [0. -1.]
                                      standard_transforms.Normalize(*mean_std)])

  img = trans(img)
  data = torch.unsqueeze(img, 0)
  data = data.repeat(2, 1 ,1, 1)
  
  return data

def plotting_all(img, mean_std, net, path_s):
  #preapare data
  data = transform_im(img, mean_std)

  #afficher l'image d'origine
  plot_graphic(data[0,0], cmap = 'jet')
  
  #calculer les sorties des Aspp
  apply.assp_all([data], net, path_s) 
  data_aspp = apply.load_npy(path_s+'_assp.npy')

  #obtenir les sorties de segmentation
  data_seg = apply.load_npy(path_s+'_seg.npy')
  data_edge = apply.load_npy(path_s+'_edge.npy')

  #calculer les sorties des gates
  apply.gate_all([data], net, path_s)
  data_g3 = apply.load_npy(path_s+'_gate1.npy')
  data_g4 = apply.load_npy(path_s+'_gate2.npy')
  data_g7 = apply.load_npy(path_s+'_gate3.npy')

  #afficher la segmentation de l'algorithme
  plot_graphic(data_seg[0], cmap = 'Paired')

  #afficher les bordures trouvées
  plot_graphic(data_edge[0], cmap = 'Greys')
  
  #afficher les gates
  plot_graphic(data_g3[0], cmap = 'Spectral')
  plot_graphic(data_g4[0], cmap = 'Spectral')
  plot_graphic(data_g7[0], cmap = 'Spectral')

def it_hclus(net, data, mini=6, maxi=16, res=1, lim=0.95, border =16):
  """
  Fonction pour réaliser le clustering sur la sortie de l'ASPP
  """   
  vals, seg_out, edge_out = apply.aspp_output(net, data)
  _, x, y = visualize.prepare_data2cluster(vals.cpu()[0], border=border, res=res)
  
  for i in range(mini,maxi):
  #hierarchical
    start = time.time()
    results = visualize.hclustering_output(vals.cpu()[0], n_clusters=i, distance_threshold=None, border = border, res=res, lim = lim)
    end = time.time()
    tim = end-start

    print(i)
    plot_graphic(results.reshape(x, y), cmap = 'tab20')
    
def import_gdown(path, name):
    """
    import google drive file from link et l'enregistre dans le currnt path
    in : path - le lein drive pour charger l'image
         name - le nom du fichier a donner a l'image (default = 'temp')
    """
    idi = path.split('/')[-2]
    link = 'https://drive.google.com/uc?id={}'.format(idi)
    gdown.download(link, name, quiet=False)
    
def import_tiff(path, name = 'temp'):
  """
  Fonction pour importer une image .tiff
  in : path - le lein drive pour charger l'image
       name - le nom du fichier a donner a l'image (default = 'temp')
  out : l'image chargée
    """
  #importer l'image avec gdown depuis un lien
  name += '.tiff'
  import_gdown(path, name)
  
  #charger l'image au format tuile 
  image = gdal.Open(name)
  image = image.ReadAsArray()

  #verify it's a 2Dd raster
  if len(image.shape) != 2:
    raise TypeError('the file has to be a 2D rasater file')

  #verify it does not contains no NoData values
  if ~np.isnan(image).any() :
    raise ValueError('the file contains NoData values')

  return image