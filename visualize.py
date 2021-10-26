# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:02:05 2021

@author: Moi
"""
import os
try : 
    os.chdir('G:/Mon disque/Colab Notebooks/GSCNN-master')
except OSError:
    pass
    
import loss
from train import args, parser
import network
from utils.misc import prep_experiment
import optimizer
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import skimage.measure

def return_acmap(net, layer, input):
    """
    Fonction pour obtenir la sortie du réseau (net) 
    à une couche choisie (layer)
    en fonction de l'entrée (input)'
    """
    outputs = []  
    input_cuda = input.cuda()    
    
    def hook(module, input, output):
        outputs.append(output)        
        
    layer.register_forward_hook(hook)

    with torch.no_grad():
        seg_out, edge_out = net(input_cuda)
            
    vals = outputs[0]

    return vals, seg_out, edge_out

def show_dendrogram(vals, model, save=False):
    """
    Fonction pour afficher le dendrogram des d'un regroupement agglomératif
    in : vals - les valeures en 1d des images à regrouper
    """
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_,
      counts]).astype(float)
    
    # Plot the corresponding dendrogram
    fig = plt.figure(figsize=(15, 8))
    plt.title('Dendogramme', fontsize = 20)

    dendrogram(linkage_matrix)
    
    plt.xlabel('Filtres')
    plt.ylabel('Similarité')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.show()

def prepare_data2cluster(output, border=6, res=2):
    """
    fonction pour préparer les données aspp à être en cluster:
    elles sont normalisées par la moyenne et la std
    in : output - les données en entrée
         border - la bordure à supprimer (default 6)
    out : outpur prêt à être en cluster
    """  
    #on fait un max_pooling
    vals = maxpooling_aspp(output, res)
    
    #préparation des données
    z, x, y = vals.shape
    x, y = x-border, y-border
    
    #on change les x et y pour qu'ils soient plus en 1d
    vals = vals[:, border//2:-border//2, border//2:-border//2].reshape(z, x*y)
    
    #normalisation
    X = StandardScaler().fit_transform(vals)
    X = np.swapaxes(X , 0, 1)
    
    return X, x, y


def maxpooling_aspp(data, res=2):
    """
    Fonction pour réduire la taille des données et obtenir un max pooling
    permet de : supprimer les trucs moins importants pour le clustering
    et d'alleger le temps de calcul
    in : data - les données à réduire
         res - la résolution vertical et horizontale de la réduction (default =2)
    """
    data_t = []
    
    for i in data:
        data_t.append(skimage.measure.block_reduce(i, (res,res), np.max))
    
    return np.array(data_t)

def kclustering_output(output, n_clusters, border=6, lim=None, res=2):
    """
    fonction pour réaliser un kcluster de sklearn sur les données
    in : output - les données à rassembler
         n_clusters - le nombre de cluster à former 
         border - les bordures à supprimer (default 6)
         lim - la limite de la variance expliquée par le PCA
    out : les données regroupées
    """
    #normalisation 
    X, x, y = prepare_data2cluster(output, border=border, res=res)

    #faire un PCA
    if lim!=None:
        X = calculate_pca(X, lim)
        
        
    #clustering 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    results = kmeans.labels_
    
    # remmetre les données à la taille d'origine
    results = results.reshape(x, y)
    
    return results

def hclustering_output(outputs,
                       n_clusters = None,
                       distance_threshold=1.5,
                       dendo=False,
                       border=6,
                       lim=None,
                       res=2):
    """
    faire le clustering des couches de sorties à  partir d'un regroupement hierarchique
    input : ouptus - les données de l'aspp en entrée
            n_cluster, le nombre de cluster (int ou none si distance_threshold!=None)
            distance_threshold, le seuil de classe (int ou none si n_cluster!=None)
            dendo - montrer le dendogramme (default = False)
            border - les bordures de l'image à enlever (default = 6 )
            lim - la limite de la variance expliquée par le PCA
    out : results - le résultat du regroupement
    """
    #normalisation 
    X, x, y = prepare_data2cluster(outputs, border=border, res=res)
    
    #faire un PCA
    if lim!=None:
        X = calculate_pca(X, lim)
        
    #faire le regroupement
    agglo_model = AgglomerativeClustering(n_clusters = n_clusters,
                                            affinity='euclidean',
                                            linkage='ward',
                                            distance_threshold=distance_threshold)
    results = agglo_model.fit_predict(X)
    
    #montre le dendogramme
    if dendo:
        show_dendrogram(X, agglo_model)
    
    # remmetre les données à la taille d'origine
    results = results.reshape(x, y)
    
    return results

def aspp_output(net, img):
    """
    obtenir les sorties de la couche de aspp
    """
    
    result1, seg_out, edge_out = return_acmap(net, net.module.bot_fine, img)
    result2, seg_out, edge_out = return_acmap(net, net.module.bot_aspp, img)
    result2 = F.interpolate(result2, result1.size()[2:], mode='bicubic',align_corners=True)

    vals = torch.cat([result1, result2], 1)
    
    return vals, seg_out, edge_out

def get_alphas(net, img):
    """
    fonction pour sortir les couches alpha avant les gates du shape stream
    """
    result_dsn3, seg_out, edge_out = return_acmap(net, net.module.dsn3, img)
    result_dsn3 = F.interpolate(result_dsn3, img.size()[2:],
                            mode='bicubic', align_corners=True)
    
    result_dsn4, seg_out, edge_out = return_acmap(net, net.module.dsn4, img)
    result_dsn4 = F.interpolate(result_dsn4, img.size()[2:],
                            mode='bicubic', align_corners=True)
    
    result_dsn7, seg_out, edge_out = return_acmap(net, net.module.dsn7, img)
    result_dsn7 = F.interpolate(result_dsn7, img.size()[2:],
                            mode='bicubic', align_corners=True)
    
    return result_dsn3, result_dsn4, result_dsn7

def load_image2torch(img):
    """
    fonction pour changer une image 1d de npy à torch
    in : image (size = (x,y))
    out : vector(2,3,x, y)
    """
    img = torch.from_numpy(np.repeat(np.expand_dims(np.repeat(np.expand_dims(img, axis=0), 3, axis=0), axis=0), 2, axis=0)).type(torch.FloatTensor)

    return img

from datasets import syntmag

def load_trained_model(args):
    """
    fonction pour charger un modèle en mode eval()
    changer les args pour charger un modèle entrainé
    """
    writer = prep_experiment(args,parser)
    # train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    args.dataset_cls = syntmag
    criterion, criterion_val = None#loss.get_loss(args)
    net = network.get_net(args, criterion)    
    #optim, scheduler = optimizer.get_optimizer(args, net)
    torch.cuda.empty_cache()
    net.eval()
    
    return net

from skimage.transform import resize

def resize_along_axis(img, ls_dic, new_s, border):
    a=0
    temp_ = []
    for i in range(ls_dic[0]):
        temp__ = []
        for j in range(ls_dic[1]):
            im = img[a, j, border:-border, border:-border]
            im = resize(im, new_s)
            temp__.append(im)
        a+=1
            
        temp_.append(np.array(temp__))
    return np.array(temp_)

def calculate_pca(X, lim):
    """
    fonction pour faire un PCA sur les données afin de booster le clustering
    in : X - les données nromalisées
         lim - la varioance qui doit être expliquée par les PCA
    out : les données réduites selon des PCA
    """
    #calculer l'ensemble des PCA
    my_model = PCA(n_components= X.shape[1])
    trans = my_model.fit_transform(X)

    #calculer la variance
    temp = np.array(my_model.explained_variance_ratio_.cumsum())
    nbpca = np.sum(temp<lim)
    
    print('number of PCA : %s'%nbpca)
    #fair un k_clustering 
    #obtenir les variances qui expliquent notre truc
    return trans[:, :nbpca]
    
#%%
if __name__ is '__main__':
    args.dataset = 'syntmag'
    args.snapshot = 'G:/Mon disque/Colab Notebooks/GSCNN-master/checkpoints/best_epoch_44_mean-iu_0.59201.pth'
    
    # charger le modèle
    net = load_trained_model(args)
       
    #charger les images originales
    img=np.load('G:/Mon disque/mira_project/original.npy') 
    
    #normaliser l'image entre 0 et 1
    img = img[0, :, :, 0]
    #%%
    img = (img-(np))/(2000-(-500)) 
    
    #%%
    #afficher l'image
    plt.imshow(img, cmap='Spectral')
    plt.title('Données Originales')
    plt.axis('off')
    plt.show()
    
    #la transformer en tensor
    img = load_image2torch(img)   

    #charger les lithologies
    from skimage import feature
    lit=np.load('G:/Mon disque/mira_project/litho_original.npy')
    lit = lit[:, :, 0]
    imto = feature.canny(image=lit,
                            sigma=2,
                            low_threshold=0.1,
                            high_threshold=0.3)
    
    #afficher les lithologies
    cmap = plt.get_cmap('Spectral', 10)
    cb = plt.imshow(lit, cmap=cmap)
    plt.title('Lithologie')
    plt.axis('off')
    plt.show()
    
    import cv2
    #afficher les bordures trouvées par canny
    imte = (img.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)
    imt = cv2.Canny(imte[0],1,15)
    plt.imshow(imt, cmap='Greys')
    plt.title('Canny délimitation')
    plt.axis('off')
    plt.show()
    
    
    #%%
    
    #afficher les contacts connues
    plt.imshow(imto, cmap='Greys')
    plt.title('Contacts')
    plt.axis('off')
    plt.show()
        
    #%%
    #préparer les gates
    dsn3, dsn4, dsn7 = get_alphas(net, img)

    plt.imshow(dsn3[0,0,:,:].cpu(), cmap='Spectral')
    plt.axis('off')
    plt.show()

    plt.imshow(dsn4[0,0,:,:].cpu(), cmap='Spectral')
    plt.axis('off')
    plt.show()
    
    plt.imshow(dsn7[0,0,:,:].cpu(), cmap='Spectral')
    plt.axis('off')
    plt.show()
    
    #%%
    #essayer un clustering plus rapide que le Klearn
    vals, seg_out, edge_out = aspp_output(net, img)  

    outputs = vals.cpu().numpy()[0]
    #14
    n_clusters = 14
    # results = clustering_output(vals[1].cpu().numpy(), distance_threshold=None, n_clusters=n_clusters, dendo=False)
    # z, x, y = outputs.shape
    # x, y = x-6, y-6
    # vals = np.swapaxes(outputs[:, 3:-3, 3:-3].reshape(z, x*y), 0, 1)


    
    #%%

    
    
    

        
        
    