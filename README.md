# Sampling possible reconstructions of undersampled acquisitions in MR imaging
by Kerem C. Tezcan, Neerav Karani, Christian F. Baumgartner and Ender Konukoglu

All authors are with the Computer Vision Lab, ETH Zürich, Switzerland. CFB is also with the Machine Learning in Medical Image Analysis Group, University of Tübingen.

## Paper
The main idea is to sampling multiple reconstructions instead of only reconstructing a single image for undersampled MR imaging. This way one can find multiple solutions to the ill-posed inverse problem and can characterize the uncertainty in the solutions as well. Notice that this is a different approach than obtaining model uncertainty by Bayesian networks/dropout or modeling aleatoric uncertainty by predicting errors with heteroscedastic models.


The paper is under review. A preprint version can be found here: https://arxiv.org/abs/2010.00042.


## Results
Here we show the results for the proposed l-MALA method as well as the compared methods. Notice the figure is a gif, showing multiple samples for the sampling methods. You can click on the gif and then zoom in to see the structural changes in your browser with Ctrl+mouse scroll or in your browser settings. You can find more examples in the gifs folder.

 The image presented is from a subject from the Human Connectome Project dataset, retrospectively undersampled at a factor of 5.

![plot](./gifs/gif_vol4_usfact5_kspns0.gif)

### Shared results as numpy files
To view the results of the reconstruction and the sampling in python without having to run the code, we share here the numpy files of two examples, one from the HCP dataset and one from the the in-house measured dataset. These can be downloaded from: https://polybox.ethz.ch/index.php/s/9uLZaHlgV2MTBDS (around 644.7 MB).

## Code
### How to use:
1. Get the repo into your computer/server.
2. Make an environment using the requirements file, use the minimal_pip with pip to get only the necessary libraries or the other two to get more comprehensive environments (though these might include old libraires and may be more error prone).
3. Download the trained models from the link below and put them into the location as shown below.
4. Create the folder structure as shown below.


### Folder structure
The code is published here as well. It assumes such a folder structure: 
```
project
└─── sampling/
│    └─── Code/
│    └─── example_data/
│         └─── hcp_image/ 
│         └─── usz_image/         
│    └─── gifs/
│    README.md
│
└─── results/
│    └─── hcp/
│         └─── reconstruction/
│         └─── samples/
│              └─── decoder_samples/
│     └─── usz/
│          └─── reconstruction/
│          └─── samples/
│               └─── decoder_samples/
│            
└─── trained_models/ 
│    └─── covariances_emp_prior/
```
This structure will otherwise be created to some extent, but the code generally needs this structure to read and write files, so would be wise to create these by hand.

### Trained models
This repo contains only the contents of the "./sampling/" folder, the rest can be downloaded or generated. The trained models including the parameters of the empirical latent distribution can be downloaded at https://polybox.ethz.ch/index.php/s/KtqM19ttB40hX8R (around 616.9 MB) and should be placed into a folder as shown above.
