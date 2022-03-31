# GSCNN APPLIED TO MAGNETIC DATA

original code from : https://github.com/nv-tlabs/GSCNN
(Towaki Takikawa, David Acuna, Varun Jampani, Sanja Fidler : [Paper](https://arxiv.org/abs/1907.05740)]).

The description of our implementation and the data augmentation workflow is available [in Arxiv](http://arxiv.org/abs/2110.14440)

An interface to apply the code to airborne magnetic data is available on [Google Colab](https://colab.research.google.com/github/MatthieuCed/GSCNN-apply-to-airborne-magnetic/blob/main/Preliminary_Mapping_GSCNN.ipynb)

### Download datas (pretrained weights and database)

 from: https://drive.google.com/drive/folders/1CrdDffu_KftxgBjG7QF5YrobL-oRar4s?usp=sharing
 
 it contains: 
 - the trained weights
 - the trained history
 - the data in zip format (to unzip)
 - real data for testing (mag_similar_[...] an mag_model_pp.pkl)
 - the data of the original 3d model (original.npy)
 - data form grenville
 
 ### Results 
 
 To see the results, open the Results_GSCNN.ipynb (google colab)
 Change the path of the files to open, to correspond to yours, copied in a google drive
 
 ### Training
 
 Open run_GSCNN.ipynb (in google colab)
 Change the config.py to correspond to the files at your location
 

