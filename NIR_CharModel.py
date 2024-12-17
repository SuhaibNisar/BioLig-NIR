import sys
from NIR_ImportExport import *
from NIR_Metrics import *
from NIR_Plots import *
from NIR_Preprocessing import *
from NIR_Training import *
import os

# Get the path of the directory where the script is located
script_path = os.path.dirname(__file__)
path_prev = script_path.partition("\\Import_Scripts")[0]

file_name=path_prev + "/Training_Testing_Data_ModelDevelopment.xlsx" # Imports the training and testing data excel from the previous folder
_, wavelength, spectra_raw=file_import(file_name,'Reflectance') # Extracts the spectral data and the wavelength

training_set = pd.read_excel(file_name, sheet_name="Names",header=0).loc[:,'Training?'].values=="Y" # Gets the indices of the training set
testing_set = pd.read_excel(file_name, sheet_name="Names",header=0).loc[:,'Testing?'].values=="Y" # Gets the indices of the testing set
comp_dataset = pd.read_excel(file_name, sheet_name="Compositions",header=0) ; comp_headers=list(comp_dataset.columns[0:3]) # Gets the compositions
comp_vals=comp_dataset.loc[:,comp_headers].values ; comp_vals_train=comp_vals[training_set] ; comp_vals_test=comp_vals[testing_set] # Segments the compositions into training and testing datasets as defined in the excel file

spectra_raw_avg=spectra_avg(spectra_raw, num_repeats=4) # Averages the spectral data (Assumes 4 repeats)
# The below 3 lines generate a set of dummy preprocessed data to get the number of preprocessing methods done
low_nm_test = 800 ; up_nm_test = 2500
_, X_train_test, _ = preprocessing_combined(low_nm_test, up_nm_test, spectra_raw_avg, wavelength, spectra_raw, training_set, testing_set)
ncomp = len(comp_headers) ; npreproc = X_train_test.shape[2]

# specifies the wavelength ranges to be investigated (user-defined)
low_nm_range=np.array([800, 948, 1100, 1248, 1400, 1500]) ; up_nm=2500

# Capability for PLS, PCA and SVR has been included - all_vector defines whether univariate or multivariate models will be developed
# Some more options given in the 2 commented lines - 
# max_n_comp_vec defines the the maximum number of components/SVR value to be studied to - MUST EQUAL n_methods + n_all_vector
methods = ['PLS','PLS']; all_vector = [True, False] ; max_n_comp_vec = [25, 25]
# methods = ['PLS','PLS','PCA','PCA','SVR','Standard-SVR']; all_vector = [True, False, True, False, False, False]
# methods = ['PLS','PLS','PCA','PCA']; n_methods = len(methods) ; all_vector = [True, False, True, False] ; max_n_comp_vec = [25, 25, 25, 25]

n_methods = len(methods) 
output_component_wise = np.zeros((ncomp*low_nm_range.shape[0],n_methods*npreproc,16)) # Generates an empty numpy array to hold the results
print("Import complete")

for i in range(low_nm_range.shape[0]):
    low_nm = low_nm_range[i] # Gets the wavelength range for this run
    print(f'Running wavelength range {low_nm}-{up_nm}')
    wavelength_red, X_train, X_test = preprocessing_combined(low_nm, up_nm, spectra_raw_avg, wavelength, spectra_raw, training_set, testing_set)

    for j in range(n_methods):
        print(f'Running model method {j+1}/{n_methods}')
        all = all_vector[j]
        output = preprocess (X_train, comp_vals_train, X_test, comp_vals_test, methods[j], all, max_n_comp=max_n_comp_vec[j], plot_progressval=False, plot_response=False, eval_metrics = True, criterion_train='Haaland')
        l1 = list(range(1,10+6))

        for k in l1:
            output_component_wise[i*ncomp:(i+1)*ncomp, j*npreproc:(j+1)*npreproc, k] = output[k]

        if all:
            output_component_wise[i*ncomp, j*npreproc:(j+1)*npreproc, 0] = output[0]
        else:
            output_component_wise[i*ncomp:(i+1)*ncomp, j*npreproc:(j+1)*npreproc, 0] = output[0]

# Specification of technique names and preprocessing labels for output in excel file
technique_str = ['PLS Multivariate','PLS Univariate','PCA Multivariate','PCA Univariate']
# technique_str = ['PLS Multivariate','PLS Univariate','PCA Multivariate','PCA Univariate','SVR Univariate','Standard-SVR Univariate']
preprocess_labels=['Raw','SNV','MSC','SNV(Detrend)','1st Derivative','2nd Derivative','Centered', \
    'SG1-SNV','SG1-MSC','SG2-SNV','SG2-MSC']

#Specification of file name for saving
filename=path_prev+'/Training_Results/CompositionResults_charmodel.xlsx'
writer = pd.ExcelWriter(filename, engine='xlsxwriter')
writer.save()

# Function to save results to excel file
_ = results_excel_export_complete(filename, output_component_wise, preprocess_labels, technique_str, low_nm_range, comp_headers)

# Generation of dummy excel file to release previous file for opening
filename=path_prev+'/Training_Results/CompositionReset.xlsx'
writer = pd.ExcelWriter(filename, engine='xlsxwriter')
writer.save()
