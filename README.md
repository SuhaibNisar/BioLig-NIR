
This repository (**BioLig-NIR**) contains the code required to generate results for the study (Near-infrared spectroscopy for rapid compositional analysis of cellulose pulps after fractionation with protic ionic liquids) for developing near infrared (NIR) spectroscopy-based compositional models for biomass samples, primarily those fractionated with protic ionic liquids. 

There are 2 Jupyter notebooks to show how the results in the study were generated:
- **ModelTraining_Notebook.ipynb**: In this notebook, the optimal models were developed by considering different spectral preprocessing techniques and spectral wavelength ranges.
- **ModelUncertainty_Notebook.ipynb**: In this notebook, the developed models were tested on different samples to understand their performance further

There are 2 spreadsheet files:
- **Training_Testing_Data_ModelDevelopment.xlsx**: This spreadsheet file holds the data for the model training procedure
- **Data_Uncertainty.xlsx**: This spreadsheet file holds the data for the model evaluation procedure for the second notebook

The repository contains 4 folders:
- **Import_Scripts**: This folder holds scripts all the functions needed, with the file titles representing the classes of functions within each file
- **Training_Results**: This folder holds the results of the model training procedure over several preprocessing techniques and wavelength ranges. It also contains figures comparing different cases
- **UncertaintyResults**: This folder contains the results of the uncertainty estimation along with figures including parity and violin plots for different sample types

**Required packages**:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- openpyxl
- pickle
- jupyter

This repository was developed within the Optimisation Methods for Green Applications group at Imperial College London (omega-icl: https://github.com/omega-icl).

For any queries, please contact Suhaib Nisar at .