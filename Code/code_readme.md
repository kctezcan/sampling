## Here we explain each file:
1. run_reconsamp_hcp.py: this file is a script to run an example from the HCP dataset. It first loads the example slice, does the necessary preprocessing, then runs a deep density prior (DDP) reconstruction, then runs the sampling to obtain the decoder output samples (x ~ p(x|z)), then runs the sampling to obtain the proper samples (x ~ p(x|y,z)).
2. run_reconsamp_usz.py: this file is a script to run an example from the in-house dataset, measured at the University Hospital Zurich (USZ). It follows the same steps as the one above.
3. vaerecon5.py, vaerecon6.py: these are the functions for the DDP reconstruction. The difference is minimal between the two, one has the functionality of bias field correction as well.
4. vaesampling.py: this is the function that runs the decoder output sampling, implementing the l-MALA method, i.e. the MCMC sampling.
5. definevae2.py: this file creates the VAE graph necessary for the DDP reconstruction and loads its trained weights and returns the necessary parts of the VAE for reconstruction.
6. definevae_2d_v1_mri_nocov_fullim_conz_homodyn_varsize_f2.py: this file creates the 2D latent space VAE graph necessary for the l-MALA sampling and loads its trained weights and returns the necessary structures from the VAE for the sampling.
