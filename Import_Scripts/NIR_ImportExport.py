from sys import stdout
try:
    import pandas as pd
    import numpy as np
    import pickle as pkl
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import pandas as pd
    import numpy as np
    import pickle as pkl

def file_import(file_name,sheet_name,num_points=426):
    """
    Imports spectral data from an Excel file and returns the spectral data in absorbance and wavelength.
    Parameters:
    file_name (str): The path to the Excel file containing the spectral data.
    sheet_name (str): The name of the sheet in the Excel file to import data from.
    num_points (int, optional): The number of data points to import. Default is 426.
    Returns:
    tuple: A tuple containing:
        - dataset (DataFrame): The entire dataset imported from the Excel file.
        - wavelength (ndarray): An array of wavelength values.
        - spectra_raw (ndarray): An array of spectral data in absorbance.
    """
    dataset = pd.read_excel(file_name, sheet_name=sheet_name,header=0)
    dataset.columns=dataset.iloc[0]
    dataset = dataset.drop(labels=0, axis=0)
    wavelength=dataset.loc[:num_points,['Wavelength (nm)']].values[:,0]
    
    if sheet_name=='Reflectance':
        spectra_raw=dataset.iloc[:num_points,1::2].values.astype(float) # Excel file format must have repeating series of wavelength0reflectant
        spectra_raw[spectra_raw<=0] = 2 # corrects spectra with negative reflectance to a very small number
        spectra_raw=-1*np.log10(0.01*spectra_raw) # converts reflectance to absorbance
    else:
        spectra_raw=dataset.iloc[:num_points,1::2].values
    return dataset, wavelength, spectra_raw

def results_excel_export_complete(filename,output,preprocess_labels, technique_str, nm_range, comp_headers):
    """
    Exports modeling results to an Excel file with multiple sheets for each metric.
    Parameters:
        filename (str): The path to the Excel file where results will be saved.
        output (numpy.ndarray): A 3D array containing the output metrics.
        preprocess_labels (list): A list of labels for preprocessing steps.
        technique_str (list): A list of technique names used in the analysis.
        nm_range (numpy.ndarray): An array representing the wavelength range.
        comp_headers (list): A list of component headers.
    Returns:
        int: The starting column index for the next set of data.
    """
    labels=['n_comp','rmse_c_comp','rmse_cv_comp','rmse_val_comp', 'r2_c_comp',\
         'r2_cv_comp','r2_val_comp','bias_val','se_val','rpd_val', 'mae_cv','mae_val','mode_error_cv','mode_error_val','std_error_cv','std_error_val']
    ncomp = len(comp_headers)
    n_techniques = len(technique_str) ; n_metrics = output.shape[2] ; n_preprocess = len(preprocess_labels)
    row_label = np.tile(nm_range,ncomp) ; row_label = sorted(row_label)
    preprocess_labels = preprocess_labels * n_techniques
    comp_labels = comp_headers * nm_range.shape[0]
    
    df1 = pd.DataFrame(np.repeat(technique_str,n_preprocess).reshape(1,-1))
    df2 = pd.DataFrame(comp_labels) ; df2.index = row_label
    for i in range(n_metrics):
        metric = output[:,:,i]
        df = pd.DataFrame(metric)
        with pd.ExcelWriter(filename, engine="openpyxl", mode='a',if_sheet_exists='overlay') as writer:  
            df1.to_excel(writer, sheet_name=labels[i],index=False, header=preprocess_labels, startrow = 0, startcol = 2)
            df2.to_excel(writer, sheet_name=labels[i], index=True, header=False, startrow=2)
            df.to_excel(writer, sheet_name=labels[i],index=False, header=False, startrow = 2, startcol=2)
    
    startcolnext = len(preprocess_labels)
    print("Results saved!")
    return startcolnext