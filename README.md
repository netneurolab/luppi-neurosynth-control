#  Transitions between cognitive topographies: contributions of network structure, chemoarchitecture, and diagnostic categories.
Authors: A.I. Luppi, S.P. Singleton, J.Y. Hansen, K.W. Jamison, D. Bzdok, A. Kuceyeski, R.F. Betzel, & B. Misic.

This repository provides code to illustrate the central method in Luppi et al., "Transitions between cognitive topographies: contributions of network structure, chemoarchitecture, and diagnostic categories." Nature Biomedical Engineering (2024) ([preprint](https://www.biorxiv.org/content/10.1101/2023.03.16.532981v1)).

It was developed in MATLAB 2019a by Andrea Luppi from the the [Network Neuroscience Lab](netneurolab.github.io/) at the Montreal Neurological Institute, McGill University.

This code relies on MATLAB code from the [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet) for MATLAB by Rubinov and Sporns (2010) NeuroImage.
For additional plotting functionality, also include in your MATLAB path the [ENIGMA Toolbox](https://github.com/MICA-MNI/ENIGMA.git) by Lariviere et al. (2021) Nature Methods.

The study investigates how network control energy in the brain depends on the network, but also the start and destination states, and the control strategy.
In this code, we change these three elements one at a time.

## Repository Structure
### Main script
The main file is [neurosynth_control_4GitHub.m](neurosynth_control_4GitHub.m)
This script should work out of the box, if run from the parent directory. However, the user may also customise some parameters provided in the SET PARAMETERS section (see comments in the code for details)
To run, ensure you are in the main directory of the repo.

#### Part I: changing the start and destination states
The core innovation from [Luppi et al (2024) NatBME](https://www.biorxiv.org/content/10.1101/2023.03.16.532981v1) is to use start and target states defined as [NeuroSynth](https://neurosynth.org/) meta-analytic maps associated with different terms from the cognitive neuroscience literature.

#### Part II: changing the network topology
Next we can change the network topology, by using different kinds of rewiring: degree-preserving (Maslov-Sneppen) rewiring, and a more stringent geometry-preserving rewiring that preserves both degree and connection length, to account for spatial embedding in the brain

#### Part III: Heterogeneous control
To use heterogeneous controls, we can add or subtract (or replace) values from the identity matrix, for example according to an empirical map of interest;
In this example we use the cortical thickness map.

### `data`
The [data](data/) folder contains all the data you need to make this code run: 
- `structural_connectome_DesikanKilliany68.m` - an empirical structural connectome from diffusion tractography (consensus connectome from 100 HCP participants) in the 68-ROI Desikan-Killiany cortical atlas
- `Euclidean_distances_DesikanKilliany68.m` - Euclidean distances between ROIs of the Desikan-Killiany atlas
- `NeuroSynth_maps_and_terms_DesikanKilliany68.m` - meta-analytic maps (in Desikan-Killiany parcellation) pertaining to 123 cognitive terms from [NeuroSynth](https://neurosynth.org/), as well as the terms themselves, and a subset of terms that we use for visualisation purposes in the paper

### `utils`
The [utils](utils/) folder contains support functions:
- `fcn_optimalControlContinuous.m` - this function (from Shi Gu's [GitHub repo](https://github.com/gushiapi/Dynamic-Trajectory.git), lightly edited by A.Luppi) computes network control energy between a start and target state
- `fcn_match_length_degree_distribution.m` - function from Rick Betzel to generate geometry-preserving nulls, from a network and matrix of Euclidean distances
- `fcn_randmio_und.m` - this is the [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet) function to generate degree-preserving (Maslov-Sneppen) rewired nulls
- `fcn_quick_mat_plot.m` - plot a matrix as heatmap with optional labels for rows and/or columns
- `rdbu_sym.m` - makes a colormap with blues for negatives and reds for positives; required the [ENIGMA Toolbox](https://github.com/MICA-MNI/ENIGMA.git) to be on the MATLAB path

## Contact Information
For questions, please email: [al857@cam.ac.uk](al857@cam.ac.uk).





