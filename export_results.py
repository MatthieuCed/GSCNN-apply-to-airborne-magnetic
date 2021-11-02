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
from IPython.display import display, clear_output
import matplotlib.patches as mpatches
import skimage.transform
from scipy.ndimage import gaussian_filter

import visualize
import apply
from train import args
import time
from network import get_model
from datasets import syntmag
from optimizer import restore_snapshot

images = {}
nets = {}
used_net = ''

preload_image_path = {'malartic_synt' : 'https://drive.google.com/file/d/1J8UX-fTlHY-NqyF15IYQrfXANHqZJY4H/view?usp=sharing',
                      'malartic' : 'https://drive.google.com/file/d/1UZLueOuQieyl0LN8ZUyK69ktJ0hmCuEV/view?usp=sharing',
                      'val d\'or' : 'https://drive.google.com/file/d/1XJeGl4fjcCBSRdA5NgZnl6OayjBLk4Cu/view?usp=sharing',
                      'ashram' : 'https://drive.google.com/file/d/13hfgcY8f6fScr98VdqSnwdrk53M-cvqT/view?usp=sharing',
                      'eleonor' : 'https://drive.google.com/file/d/1a_SEZmH-ddjdUF8mXTa2p1rd-lvrAEjl/view?usp=sharing',
                      'niobec' : 'https://drive.google.com/file/d/1zPig5w_SvY_VS1XCv_CyF23F5l1c5zWt/view?usp=sharing',
                      '32A' : 'https://drive.google.com/file/d/1bCapwErTpViVXl-QjqqJAX3tUkqyxBJA/view?usp=sharing'}

weight_path = {'weight_syntmag' : 'https://drive.google.com/file/d/1hHFL1Kkex_AdDCHo-Uo0U_kaoBUH0Xsq/view?usp=sharing'}

syntmag_labels = ['Greywackes', 'Dyke', 'Pluton']

qualitative = ['Paired', 'Pastel1', 'Pastel2', 'Accent',
               'Dark2', 'Set1', 'Set2', 'Set3',
               'tab10', 'tab20', 'tab20b', 'tab20c']

diverging = ['Spectral', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'coolwarm', 'bwr', 'seismic']

pusc = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

sequential = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

sequential_2 = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']

def plot_graphic(image, cmap = 'Paired', vmin = False, legend=False, labels=None, title=None): 
  plt.figure(figsize=(20, 10), dpi=80)
  lut = len(np.unique(image))
  if lut>30:
    cmap = cm.get_cmap(cmap, lut=lut)
  if type(vmin) is tuple:
    im = plt.imshow(np.array(image), cmap=cmap, vmin=vmin[0], vmax=vmin[1])
  else : 
    im = plt.imshow(np.array(image), cmap=cmap)
    
  if legend:
    values = np.unique(image.ravel())
    #colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
  if type(title) == str:
      plt.title(title)
      
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
    
def clip_data(data, cutoff = 0.3):
    """
    fonction np.clip pour enlever les valeurs abhérentes
    in : data - le jeu de données
         cutoff - la limite en terme d'écart type multiplié des valeurs abherantes
    out : data corrigée
    """
    #calcul de la moyenne, de la l'écart-type et de la limite de détection
    X = data[~np.isnan(data)]
    
    if type(cutoff)==float:
        mean = np.mean(X)
        std = np.std(X)
        anomal = std * cutoff
        
        #calcul des limites inférieure et supérieure
        lower_lim = mean - anomal
        upper_lim = mean + anomal
    elif type(cutoff)==tuple:
        lower_lim, upper_lim = cutoff
    
    # suppression des valeurs au dessus des limites
    X = np.clip(data, lower_lim, upper_lim)
     
    return X

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
  image = clip_data(image, cutoff = (-750, 1500))
    
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
  
def resize_image(im, rat):
    x, y = im.shape
    x, y  = int(x*rat), int(y*rat)
    
    return skimage.transform.resize(im, (x,y), order=3)

def resize_smooth(im, rat):
    """
    Fonction pour mettre les image à une échelle désirée et pour lisser selon 
    un filtre gaussien
    in : im - l'image à traiter
    
    """    
    im = resize_image(im, rat)    
    im = gaussian_filter(im, 1.2)
    
    return im 
  
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

  #choose cluster type
  drp_clu_cmap = wid.Dropdown(options = qualitative, description='Colormap :') 

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
            description='Reduction :',
            value=1)

  ##choose PCA
  sld_pca = wid.IntSlider(
            min=50,
            max=100,
            step=2,
            description='% PCA :',
            value=96)


  #fonction clustering
  def clustering_fonction(obj):
      clear_output()
      display_menu()
      
      #define clustering type
      if drp_clu_typ.value == 'hierarchical clustering':
          hclust, kmean = True, False
      elif drp_clu_typ.value == 'k means':
          hclust, kmean = False, True
         
      #define PCA
      if sld_pca.value == 100:
          pca=None
      else:
          pca=sld_pca.value/100

      #do the clustering
      clustering_output(border = sld_brd.value,
                          res = sld_res.value,
                          lim = pca,
                          cmap = drp_clu_cmap.value,
                          n_clusters = sld_clu.value,
                          hclust = hclust,
                          kmean = kmean)

  #action du boutton
  btn.on_click(clustering_fonction)
  
  def display_menu():
      #create the interface
      display(wid.Label(value="Clustering Mapping | Carte par Regroupement"))
      display(wid.HBox([drp_clu_typ, wid.Label(value="choose the clustering algorithm | choisissez l'algorithme de regroupement")]))
      display(wid.HBox([sld_clu, wid.Label(value="choose the number of clusters | choisissez le nombre de groupes")]))
      display(wid.HBox([sld_pca, wid.Label(value="choose PCA reduction [explanation bellow] | choisissez la réduction en PCA [explication ci-dessous]")]))
      display(wid.HBox([sld_res, wid.Label(value="choose output reduction of the resolution [explanation bellow] | choisissez la réduction de la résolution de sortie [explication ci-dessous]")]))
      display(wid.HBox([sld_brd, wid.Label(value="choose the border crop [explanation bellow] | choisissez la réduction des bordures [explication ci-dessous]")]))
      display(wid.HBox([drp_clu_cmap, wid.Label(value="choose the colormap | choisissez la carte des couleurs")]))
      display(btn)

  display_menu()

def display_gscnn_outputs():
  def create_checkbox_colormap(description, cmap):
    ch = wid.Checkbox(
        value=False,
        description=description,
        disabled=False,
        indent=False)
    
    dp = wid.Dropdown(options = cmap, description='Colormap :') 

    return ch, dp

  btn = wid.Button(description='Outputs')

  #selection outputs
  ch_seg, cmap_seg = create_checkbox_colormap('segmentation', qualitative)
  ch_brd, cmap_brd = create_checkbox_colormap('borders', ['binary', 'gray'])

  #selection gate 
  cmap = diverging + pusc + sequential +sequential_2
  ch_g1, cmap_g1 = create_checkbox_colormap('gate 1', cmap)
  ch_g2, cmap_g2 = create_checkbox_colormap('gate 2', cmap)
  ch_g3, cmap_g3 = create_checkbox_colormap('gate 3', cmap)

  def display_outputs(obj): 
    clear_output()
    display_menu()
    #create lists
    disp = [ch_seg.value, ch_brd.value, ch_g1.value, ch_g2.value, ch_g3.value]
    data = ['_seg.npy', '_edge.npy', '_gate1.npy', '_gate2.npy', '_gate3.npy']
    cmap = [cmap_seg.value, cmap_brd.value, cmap_g1.value, cmap_g2.value, cmap_g3.value]
    legend = [True, False, False, False, False]
    titles = ['Segmentation', 'Borders', 'Gate 1', 'Gate 2', 'Gate 3']

    if used_net == 'weight_syntmag':
        labels = [syntmag_labels, None, None, None, None]
    
    for i, j, k, l, m, n in zip(data, cmap, disp, legend, labels, titles):
      if k:
        data_temp = apply.load_npy(i)
        plot_graphic(data_temp[0], cmap = j, legend=l, labels=m, title=n) #test

  #button
  btn.on_click(display_outputs)

  #display
  def display_menu():
    display(wid.Label('Algorithm Outputs | Sortie de l\'Algorithme'))
    display(wid.HBox([ch_seg, cmap_seg]))
    display(wid.HBox([ch_brd, cmap_brd]))
    display(wid.Label('Deep representations | Représentations profonde'))
    display(wid.HBox([ch_g1, cmap_g1]))
    display(wid.HBox([ch_g2, cmap_g2]))
    display(wid.HBox([ch_g3, cmap_g3]))
    display(btn)
  display_menu()
  
def import_image(images_in):  
    
  btn = wid.Button(description='Get Image')

  image_link = wid.Textarea(
      value='https://drive.google.com/file/d/1qRh2NO2JIwjFJg2eJb9olg7pN8iHV1fd/view?usp=sharing',
      placeholder='right the google drive link of the image you want to load',
      description='GDrive link:',
      disabled=False)

  name = wid.Textarea(
      value='original',
      placeholder='Name your image',
      description='image name',
      disabled=False)

  cmap = ['jet'] + pusc + diverging + sequential + sequential_2
  color_m = wid.Dropdown(options = cmap, description='Colormap :') 

  btn2 = wid.Button(description='Test Image')
  pre_im = wid.Dropdown(options = preload_image_path.keys(), description='Test Images :') 
    
  size = wid.IntText(value=50,
                        description='Pixel size',
                        disabled=False)
  
  smt = wid.Checkbox(value=False,
                    description='Smooth',
                    disabled=False,
                    indent=False)
    
  def import_im(obj):
    image = import_tiff(image_link.value, name.value)
    rat = size.value/50
    
    if rat != 1:
        if smt.value:
            image = resize_smooth(image, rat)
        else:
            image = resize_image(image, rat)        
    
    #afficher l'image
    #clear_output()
    #display_menu()
    plot_graphic(image, cmap = color_m.value)
    
    #l'enregistrer
    images[name.value]=image
  
  def pre_image(obj):
    #change the values
    image_link.value = preload_image_path[pre_im.value]
    name.value = pre_im.value
    #display menu
    clear_output()
    display_menu()
      
  btn.on_click(import_im)
  btn2.on_click(pre_image)
    
  def display_menu():
      display(wid.Label('Import an image to work on || Importez une image de travail'))
      display(wid.HBox([pre_im, btn2, wid.Label('predefined testing images || images test prédéfinies')]))
      display(wid.Label('or download your own image || ou téléchargez votre image'))
      display(wid.HBox([image_link, wid.Label('Type a shared Google Drive link of the tif image to download here || Lien Google Drive d\'une image .tiff partagée')]))
      display(wid.HBox([name, wid.Label('Name the image to import || Nommez l\'image à importer')]))
      display(wid.HBox([color_m, wid.Label("choose the colormap (display only) | choisissez la couleur de la carte (affichage seulement)")]))
      display(wid.HBox([size, smt, wid.Label("magnetic resolution (m) | résolution levé magnétique (m)")]))
      display(btn)
  
  display_menu()
 
def load_net():
  btn = wid.Button(description='Load')           
  wgh = wid.Dropdown(options = weight_path.keys(), description='Weights') 

  def load_weigth(obj):
    net = prepare_net(weight_path[wgh.value], wgh.value)
    nets[wgh.value] = net

  btn.on_click(load_weigth)

  display(wid.HBox([wgh,wid.Label('load pretrained weights to run the model')]))
  display(btn)
  
def obtain_values():
  
  #select image
  dd_im = wid.Dropdown(options = images.keys(), description='Image') 

  #select weights
  dd_net = wid.Dropdown(options = nets.keys(), description='Model') 
    
  #add standard deviation
  def prepare_values(obj):
    global used_net
    used_net = dd_net.value
    get_image_trans(nets[dd_net.value], images[dd_im.value], mean_std = None)

  btn = wid.Button(description='Process Image')
  btn.on_click(prepare_values)

  display(wid.Label('Choose an image and a model to process it || Choisissez une image et un modèle pour la transformer'))
  display(wid.HBox([dd_im, dd_net]))
  display(btn)
