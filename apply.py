# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:57:44 2021

@author: Moi
"""
from visualize import load_image2torch, aspp_output, get_alphas, \
    load_trained_model, hclustering_output, kclustering_output, return_acmap
import numpy as np
import torch
import gc
from os import remove
import skimage.transform

def data2torch(data):
    """
    fonction pour changer les données de data à un format compatible pour
    les réseaux
    """
    data_temp = []
    for i in data:
        i = load_image2torch(i)
        data_temp.append(i)
        
    return data_temp

def assp_all(data, net, path):
    """
    fonction pour faire le assp sur l'ensemble de la zone
    in : data - les données au format torch,
         net - le modèle chargée avec les poids
         path - l'emplacement ou enregistrer le fichier en .npy
    out : DANS UN FICHIER .NPY im path
          [assp - liste des assp pour chaque données
          seg - liste des seg pour 'ensemble des donéées
          edge - l'ensemble des edges pour les données]
    """
    
    try :
        remove(path+'_assp.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_seg.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_edge.npy')
    except FileNotFoundError:
        pass
    
    f1=open(path+'_assp.npy','ab')
    f2=open(path+'_seg.npy','ab')
    f3=open(path+'_edge.npy','ab')
    
    for i in data:
        asppt, segt, edget = aspp_output(net, i)

        #enregistrer dans un fichier
        np.save(f1, asppt.cpu().numpy()[0])
        np.save(f2, segt.max(1)[1][0].cpu().numpy())
        np.save(f3, edget.max(1)[0][0].cpu().numpy())
        
        torch.cuda.empty_cache()
     
    f1.close() 
    f2.close() 
    f3.close() 
    
def gate_all(data, net, path):
    """
    fonction pour faire le assp sur l'ensemble de la zone
    in : data - les données au format torch,
         net - le modèle chargée avec les poids
         path - l'emplacement ou enregistrer le fichier en .npy
    out : DANS UN FICHIER .NPY im path
          [assp - liste des assp pour chaque données
          seg - liste des seg pour 'ensemble des donéées
          edge - l'ensemble des edges pour les données]
    """
    
    try :
        remove(path+'_gate1.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_gate2.npy')
    except FileNotFoundError:
        pass
    
    try :
        remove(path+'_gate3.npy')
    except FileNotFoundError:
        pass
    
    f1=open(path+'_gate1.npy','ab')
    f2=open(path+'_gate2.npy','ab')
    f3=open(path+'_gate3.npy','ab')
    
    for i in data:
        dsn3, dsn4, dsn7 = get_alphas(net, i)

        #enregistrer dans un fichier
        np.save(f1, dsn3.cpu().numpy()[0, 0, :, :])
        np.save(f2, dsn4.cpu().numpy()[0, 0, :, :])
        np.save(f3, dsn7.cpu().numpy()[0, 0, :, :])
        
        torch.cuda.empty_cache()
     
    f1.close() 
    f2.close() 
    f3.close() 
            
def load_npy(path):
    """
    Fonction pour ouvrir les fichier .npy pour visualisation
    in : path - l'emplacement des données .npy
    out : data - les données chargées
    """
    f =open(path, 'rb')
    data_temp = []
    while True:
        try :
            data_temp.append(np.load(f))
        except ValueError:
            break
         
    f.close()
    data_temp = np.array(data_temp)

    return data_temp

def con_im(data_org1, li, al=42):
    
    data_org = data_org1.copy()
    temp = data_org[0][al:-al,al:-al]
    temp[:] = np.nan
        
    
    for l in li:
        a=l[0]
        b=l[1]
        c1, c2 = l[2]
        
        temp11 = temp
        for i in range(c1):
            temp11 = np.hstack((temp11, temp))
        
        temp12 = temp
        for i in range(c2):
            temp12 = np.hstack((temp12, temp))
            
        ini = np.hstack((temp11, data_org[a][al:-al, al:-al]))
                        
        for i in range(a+1, b):
            ini = np.hstack((ini, data_org[i][al:-al, al:-al]))
            
        ini = np.hstack((ini, temp12))

        if 'ini_t' not in locals(): #locals
            ini_t = ini
        else : 
            ini_t = np.vstack((ini_t, ini))
            
    return ini_t

def resize_image(im, rat):
    x, y = im.shape
    x, y  = int(x*rat), int(y*rat)
    
    return skimage.transform.resize(im, (x,y), order=3)


## %%
# if __name__ == '__osef__':
#     import pickle as pkl
#     import os
#     os.chdir('G:/Mon disque/Colab Notebooks/GSCNN-master')
    
#     #charger les donnéees de malartic
#     path = 'C:/Users/Moi/Documents/Maitrise_INRS/CloudStation/Maitrise_Codes/Un_seg/Z_data/mag_malartic_decoup_prepro.pkl'    
#     with open(path, 'rb') as f:
#         data = pkl.load(f)        
#     data = data[:,:,:,0]
    
#     #reshape the images to be on the net format
#     data = np.array([resize_image(i, 3) for i in data])
    
#     #changer les données à un format torch
#     data = data2torch(data)
    
#     #%%
#     from train import args
#     args.dataset = 'syntmag'
#     args.snapshot = 'G:/Mon disque/Colab Notebooks/GSCNN-master/checkpoints/best_epoch_44_mean-iu_0.59201.pth'
    
#     net = load_trained_model(args)
#     #%%
#     path_s = 'C:/Users/Moi/Documents/Maitrise_INRS/CloudStation/Maitrise_Codes/Un_seg/Z_data/malartic'
#     #%%
#     assp_all(data, net, path_s) 
#     #%%
#     data_aspp = load_npy(path_s+'_assp.npy')
#     #%%
#     data_seg = load_npy(path_s+'_seg.npy')
#     data_edge = load_npy(path_s+'_edge.npy')
    
#     #%%
#     gate_all(data, net, path_s)
#     #%%
#     data_g3 = load_npy(path_s+'_gate1.npy')
#     data_g4 = load_npy(path_s+'_gate2.npy')
#     data_g7 = load_npy(path_s+'_gate3.npy')

#     #%%
#     data_org = [i.cpu().numpy()[0,0,:,:] for i in data]
#     data_seg = [i for i in data_seg]
#     #%%
#     import matplotlib.pyplot as plt
    
#     li = [[0,10, (0,0)],
#           [13,22, (1,0)],
#           [25,34, (1,0)]]
        
#     al = 42

# #%%
#     ini_or = con_im(data_org, li, al=72)
#     #%%
#     ini_g1 = con_im(data_g3, li, al=72)
#     ini_g2 = con_im(data_g4, li, al=72)
#     ini_g3 = con_im(data_g7, li, al=72)
    
#     #%%
#     ini_seg = con_im(data_seg, li, al=72)
#     ini_esge = con_im(data_edge, li, al=72)
    
#     #%%
#     ini_aspp = con_im(data_aspp.transpose(0,2,3,1), li, al=36)
    
#     #%%
#     ini_fin = ini_or[:, 95:528]
    
#     plt.imshow(ini_fin, cmap='Spectral')
#     plt.axis('off')
#     plt.show()
    
#     #%%
#     fin_seg = ini_seg[:, 96:528]
#     cmap = plt.get_cmap('tab10', 4)
#     plt.imshow(fin_seg, cmap=cmap)
#     plt.axis('off')
#     plt.show()

#     #%%
    
#     ini_fin = ini_esge[:, 96:528]
    
#     plt.imshow(ini_fin, cmap='Greys')
#     plt.axis('off')
#     plt.show()
    
    
#     #%%
#     ini_fin = ini_g1[:, 95:528]
    
#     plt.imshow(ini_fin, cmap='Spectral')
#     plt.axis('off')
#     plt.show()
    
#     ini_fin = ini_g2[:, 95:528]
    
#     plt.imshow(ini_fin, cmap='Spectral')
#     plt.axis('off')
#     plt.show()
    
#     ini_fin = ini_g3[:, 95:528]
    
#     plt.imshow(ini_fin, cmap='Spectral')
#     plt.axis('off')
#     plt.show()
    
#     #%%
#     fin_aspp = ini_aspp[:,48:264].astype(np.float16)
#     n_clusters = 6
#     results = clustering_output(torch.from_numpy(fin_aspp), distance_threshold=None, n_clusters=n_clusters, dendo=False)
    
#     #%%
#     ini_fin = ini_or[:, 95:528]
    
#     plt.imshow(ini_fin, cmap='Spectral')
#     plt.axis('off')
#     plt.show()
    
#     #%%
    
    
    
#     #%%
    
#     ini_edg = con_im(data_edge, li, al=42)
#     plt.imshow(ini_edg, cmap='Greys')
#     plt.show()
      
#     ini_seg = con_im(data_seg.astype(float), li, al=42)
#     plt.imshow(ini_seg, cmap='gist_stern')
#     plt.show()

#     #%%
#     ini_g3 = con_im(data_g3, li, al=42)
#     plt.imshow(ini_g3, cmap='gist_stern')
#     plt.show()    

#     ini_g4 = con_im(data_g4, li, al=42)
#     plt.imshow(ini_g4, cmap='gist_stern')
#     plt.show()    
    
#     ini_g7 = con_im(data_g7, li, al=42)
#     plt.imshow(ini_g7, cmap='gist_stern')
#     plt.show()    
    
#     #%%
#     data_aspp = np.swapaxes(np.swapaxes(data_aspp, 1,3), 1, 2)
#     #%%
#     ini_aspp = con_im(data_aspp, li, al=21)
#     #%%
#     result = clustering_output(torch.from_numpy(ini_aspp).cpu(), distance_threshold=None, n_clusters=10, dendo=False)
#     #%%
    
#     vals, seg, edge = return_acmap(net, net.module.mod2.block1.convs.conv2, data[0])