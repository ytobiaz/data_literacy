# Spatio-Temporal Graph Neural Network for Urban Spaces: Interpolating Citywide Traffic Volume

This repository contains the implementation of the Graph Neural Network for Urban Interpolation (GNNUI), as described in the corresponding arXiv publication. The repository includes the code, data processing scripts, and training examples necessary to reproduce the results presented in the paper.

## Citation

If you use this repository, please cite the following paper:
Kaiser, S. K., Rodrigues, F., Azevedo Lima, C., & Kaack, L.H. (2025). Spatio-Temporal Graph Neural Network for Urban Spaces: Interpolating Citywide Traffic Volume. [published on arXiv].


## Repository Structure

- **`src/`**: Contains the main implementation of GNNUI, including adjacency matrix generation, data processing, and utility functions.
- **`data/`**: Directory for raw, processed, and prepared data. See the "Data" section below for details.
- **`bld/`**: Directory for saving trained models and elements necessary for basline computation.
- **`train.ipynb`**: Example notebook for training the GNNUI model.

## Data

The data required to reproduce the results can be downloaded from Zenodo, as mentioned in the paper. The repository assumes the following directory structure for the data:

- **`data/raw/`**: Contains the raw data files (e.g., Strava cycling data, NYC taxi data), as downloaded from Zenodo.
- **`data/processed/`**: Contains processed data files, formatted for use with GNNUI.
- **`data/data_prepared/`**: Contains additional prepared data files, beyond the baseline data, for reproducing results.

### Instructions for Data Preparation

1. Download the raw data from Zenodo and place it in the `data/raw/` directory.

## Training

The main training function is implemented in the repository. An example of how to train the model is provided in the `train.ipynb` notebook. The parameters for training are explained in detail in the paper.

## Contact

For questions or issues, please contact the corresponding author:  
**Silke K. Kaiser**  
Email: [s.kaiser@phd.hertie-school.de](mailto:s.kaiser@phd.hertie-school.de)
