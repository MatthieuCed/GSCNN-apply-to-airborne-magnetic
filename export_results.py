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
import ipywidgets as wid
from IPython.display import display

import visualize
import apply
from train import args
import time
from network import get_model
from datasets import syntmag
from optimizer import restore_snapshot


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
  if np.isnan(image).any() :
    raise ValueError('the file contains NoData values')

  return image


def get_net_use(args):
    print('num class : ', args.dataset_cls.num_classes)
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=None, trunk=args.trunk)
    
    return net

def prepare_net(weight_path, name):
  #name += '.pth'
  import_gdown(weight_path, name)
  output_path = '/content/GSCNN-apply-to-airborne-magnetic/{}'.format(name)

  #prepare net
  args.dataset_cls=syntmag
  args.snapshot = output_path

  #load model and weights
  if torch.cuda.is_available():
    net = apply.load_trained_model(args)
  else:
    net = get_net_use(args)
    net, _ = restore_snapshot(args, net, None, output_path)

  return net

def get_image_trans(net, image, mean_std = None):
  """
  fonction pour obtenir toutes les transformations depuis une image
  et les enregistrer localement
  """
  #transform data - add conditions depending of the used model
  if 'mean_std' not in locals():
      image = (image-image.min())/(image.max()-image.min())
      mean = np.mean(image)
      std = np.std(image)
      mean_std = ([mean, mean, mean],[std, std, std])
  elif type(mean_std)!= tuple:
      image = (image-image.min())/(image.max()-image.min())
      mean = np.mean(image)
      std = np.std(image)
      mean_std = ([mean, mean, mean],[std, std, std])

  img = numpy_pil(image)
  data = transform_im(img, mean_std)

  #apply aspp (get seg, edge, aspp in local)
  apply.assp_all([data], net, '') 

  #get gates 
  apply.gate_all([data], net, '')
  
def clustering_output(border = 16, 
                      res = 1,
                      lim = 0.95,
                      cmap = 'tab20',
                      n_clusters = 8,
                      hclust = False,
                      kmean = True):
  """
  fonction pour faire un clsutering
  """
  data_temp = apply.load_npy('_assp.npy')
  _, x, y = visualize.prepare_data2cluster(data_temp[0], border=border, res=res)

  if hclust:
    clust = visualize.hclustering_output(data_temp[0],
                                        n_clusters = n_clusters,
                                        distance_threshold=None,
                                        border = border,
                                        res = res,
                                        lim = lim)
  elif kmean:
    clust = visualize.kclustering_output(data_temp[0],
                                        n_clusters,
                                        border=border,
                                        lim=lim,
                                        res=res)

  plot_graphic(clust.reshape(x, y), cmap = cmap)
  

def clustering_mapping():
  """
  fonction pour afficher les widgets de la carte par regroupement
  """
  #button
  btn = wid.Button(description='Clustering')

  #nb of clusters
  sld_clu = wid.IntSlider(
      min=2,
      max=30,
      step=1,
      description='Clusters :',
      value=10)

  #choose cluster type
  drp_clu_typ = wid.Dropdown(options = ['hierarchical clustering', 'k means'], description='Clustering :') 
  if drp_clu_typ.value == 'hierarchical clustering':
    hclust, kmean = True, False
  elif drp_clu_typ.value == 'k means':
    hclust, kmean = False, True

  #choose cluster type
  drp_clu_cmap = wid.Dropdown(options = ['tab20', 'tab20b', 'tab20c', 'Pastel1',
                                          'Pastel2', 'Paired', 'Accent', 'Dark2',
                                          'Set1', 'Set2', 'Set3', 'tab10'], description='Colormap :') 

  #choose borders
  sld_brd = wid.IntSlider(
            min=0,
            max=32,
            step=2,
            description='Borders :',
            value=16)

  #choose borders
  sld_res = wid.IntSlider(
            min=1,
            max=2,
            step=1,
            description='Resolution :',
            value=1)

  ##choose PCA
  sld_pca = wid.IntSlider(
            min=50,
            max=100,
            step=2,
            description='% PCA :',
            value=96)
  if sld_pca.value == 1:
    pca=None
  else:
    pca=sld_pca.value/100

  #fonction clustering
  def clustering_fonction(obj):
      clustering_output(border = sld_brd.value,
                          res = sld_res.value,
                          lim = pca,
                          cmap = drp_clu_cmap.value,
                          n_clusters = sld_clu.value,
                          hclust = hclust,
                          kmean = kmean)

  #action du boutton
  btn.on_click(clustering_fonction)

  #create the interface
  display(wid.Label(value="Clustering Mapping | Carte par Regroupement"))
  display(wid.HBox([drp_clu_typ, wid.Label(value="choose the clustering algorithm| choisissez l'algorithme de regroupement")]))
  display(wid.HBox([sld_clu, wid.Label(value="choose the number of clusters | choisissez le nombre de groupes")]))
  display(wid.HBox([sld_pca, wid.Label(value="choose PCA reduction [explanation bellow] | choisissez la réduction en PCA [explication ci-dessous]")]))
  display(wid.HBox([sld_res, wid.Label(value="choose output resolution [explanation bellow] | choisissez la résolution de sortie [explication ci-dessous]")]))
  display(wid.HBox([sld_brd, wid.Label(value="choose the border crop [explanation bellow] | choisissez la réduction des bordures [explication ci-dessous]")]))
  display(wid.HBox([drp_clu_cmap, wid.Label(value="choose the colormap | choisissez la carte des couleurs")]))
  display(btn)