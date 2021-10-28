# GSCNN APPLIED TO MAGNETIC DATA

original code from : https://github.com/nv-tlabs/GSCNN
(Towaki Takikawa, David Acuna, Varun Jampani, Sanja Fidler : [Paper](https://arxiv.org/abs/1907.05740)])

our implementation and the description of the data augmentation process is available at : [Paper](http://arxiv.org/abs/2110.14440)

an interface to apply the code to airborne magnetic data is available on [Google Colab](https://colab.research.google.com/drive/1YHyJ1xAbIyEgzEL--srEbJXj-qLMpsgc?usp=sharing)

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
 
 To see the results, open the Results_GSCNN.ipynb
 Change the path of the files to open, to correspond to yours, copied in a google drive
 
 ### Training
 
 Open run_GSCNN.ipynb
 Change the config.py to correspond to jthe files at your location
 

