try:
    from sys import stdout
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter, detrend
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter, detrend

def spectra_avg(spectra_raw,num_repeats=4):
    """
    Compute the average spectra from raw spectral data.

    Parameters:
        spectra_raw (np.ndarray): A 2D array containing raw spectral measurements with shape (features, total_samples).
        num_repeats (int, optional): The number of repeated measurements for each sample. Defaults to 4.

    Returns:
        np.ndarray: A 2D array of averaged spectral data with shape (features, total_samples / num_repeats).
    """
    num_samples=np.int64(spectra_raw.shape[1]/num_repeats)
    spectra_raw_avg=np.zeros((spectra_raw.shape[0],num_samples))
    for i in range(num_samples):
        spectra_raw_avg[:,i]=np.mean(spectra_raw[:,num_repeats*i:num_repeats*(i+1)],axis=1)
    return spectra_raw_avg

def spectra_red_range(low_nm, high_nm, spectra, wavelength, spectra_raw):
    """
    Extracts a red spectral range from the given spectra data.

    Parameters:
        low_nm (int or float): The lower bound of the wavelength range in nanometers.
        high_nm (int or float): The upper bound of the wavelength range in nanometers.
        spectra (ndarray): The averaged spectral data.
        wavelength (ndarray): The array of wavelength values corresponding to the spectra.
        spectra_raw (ndarray): The raw spectral data.

    Returns:
        tuple:
            wavelength_red (ndarray): The wavelengths within the specified red range.
            spectra_avg_red (ndarray): The averaged spectral data within the red range.
            spectra_raw_red (ndarray): The raw spectral data within the red range.
    """
    nm_lower=np.where(wavelength == low_nm)[0][0]+1
    nm_upper=np.where(wavelength == high_nm)[0][0]
    spectra_raw_red=spectra_raw[nm_upper:nm_lower,:]
    spectra_avg_red=spectra[nm_upper:nm_lower,:]
    wavelength_red=wavelength[nm_upper:nm_lower]
    return wavelength_red, spectra_avg_red, spectra_raw_red

def spectra_smoothing(spectra,w,p,deriv):
    """
    Applies Savitzky-Golay filter to smooth the input spectra.

    Parameters:
        spectra (array-like): The spectral data to be smoothed.
        w (int): The length of the filter window (must be a positive odd integer).
        p (int): The order of the polynomial used to fit the samples.
        deriv (int): The order of the derivative to compute.

    Returns:
        array-like: The smoothed spectral data.
    """
    spectra_smooth=savgol_filter(spectra, w, polyorder = p, deriv=deriv, axis = 0)
    return spectra_smooth

def snv(input_data):
    '''Input Data: (N_features x N_samples)
    Returns SNV, Centered Data'''
    # Define a new array and populate it with the corrected data  
    data_mean=np.mean(input_data,axis=0)
    data_std=np.std(input_data,axis=0)
    data_centered = input_data - data_mean
    output_data=(input_data-data_mean)/data_std
 
    return output_data, data_centered

def msc(input_data, reference=None):
    '''Input Data: (N_features x N_samples) 
    Perform Multiplicative scatter correction'''
 
    # mean centre correction

    #input_data_corr = input_data - np.mean(input_data,axis=0)
    input_data_corr = input_data[:,:]

    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data_corr, axis=1)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data_corr)
    print(input_data_corr.shape)
    for i in range(input_data_corr.shape[1]):
        # Run regression
        fit = np.polyfit(ref, input_data_corr[:,i], deg=1)
        # Apply correction
        data_msc[:,i] = (input_data_corr[:,i] - fit[1]) / fit[0]
 
    return (data_msc, ref)

def preprocessing_combined(low_nm, up_nm, spectra_raw_avg, wavelength, spectra_raw, calibration_set, validation_set):
    """
    Preprocesses spectral data by applying various preprocessing techniques to both calibration and validation datasets.
    Parameters:
        low_nm (float): Lower bound of the wavelength range in nanometers.
        up_nm (float): Upper bound of the wavelength range in nanometers.
        spectra_raw_avg (ndarray): Averaged raw spectral data.
        wavelength (ndarray): Array of wavelength values.
        spectra_raw (ndarray): Raw spectral data.
        calibration_set (list or ndarray): Indices of the calibration samples.
        validation_set (list or ndarray): Indices of the validation samples.
    Returns:
        tuple:
            wavelength_red (ndarray): Reduced wavelength array after selecting the specified range.
            X_train (ndarray): Preprocessed training data stack with multiple preprocessing steps applied.
            X_test (ndarray): Preprocessed testing data stack with multiple preprocessing steps applied.
    """
    wavelength_red, spectra_red, _= spectra_red_range(low_nm, up_nm, spectra_raw_avg, wavelength, spectra_raw)

    w = 4*4+1 ; p = 2*2
    spectra_smooth=spectra_smoothing(spectra_red,w,p,deriv=0)

    X_data=spectra_smooth[:,:] ; X_data=X_data[:,calibration_set] 

    Xsnv, Xcenter = snv(X_data) ; Xsnv_detrend=detrend(Xsnv,axis=0)

    Xmsc,ref=msc(X_data)

    w = 15  ; p = 2
    X_deriv=X_data[:] ;  X_savgol1=spectra_smoothing(X_deriv,w,p,deriv=1)
    X_savgol1snv=spectra_smoothing(Xsnv,w,p,deriv=1) ; X_savgol1msc=spectra_smoothing(Xmsc,w,p,deriv=1)

    w = 23 ; p = 3 ; X_savgol2=spectra_smoothing(X_deriv,w,p,deriv=2)
    X_savgol2snv=spectra_smoothing(Xsnv,w,p,deriv=2) ; X_savgol2msc=spectra_smoothing(Xmsc,w,p,deriv=2)

    #Restricting to validation data
    X_val=spectra_smooth[:,validation_set]

    #Validation Preprocessing
    Xsnv_val, Xcenter_val = snv(X_val)
    Xsnv_detrend_val=detrend(Xsnv_val,axis=0) ; Xmsc_val,ref=msc(X_val,ref)

    w = 15 ; p = 2
    X_savgol1_val=spectra_smoothing(X_val,w,p,deriv=1)
    X_savgol1snv_val=spectra_smoothing(Xsnv_val,w,p,deriv=1)
    X_savgol1msc_val=spectra_smoothing(Xmsc_val,w,p,deriv=1)

    w = 23 ; p = 3
    X_savgol2_val=spectra_smoothing(X_val,w,p,deriv=2)
    X_savgol2snv_val=spectra_smoothing(Xsnv_val,w,p,deriv=2)
    X_savgol2msc_val=spectra_smoothing(Xmsc_val,w,p,deriv=2)

    X_train = np.stack((X_data.T,Xsnv.T,Xmsc.T,Xsnv_detrend.T,X_savgol1.T,X_savgol2.T, Xcenter.T, X_savgol1snv.T, X_savgol1msc.T, X_savgol2snv.T, X_savgol2msc.T),axis=2)
    X_test = np.stack((X_val.T,Xsnv_val.T,Xmsc_val.T,Xsnv_detrend_val.T,X_savgol1_val.T,X_savgol2_val.T, Xcenter_val.T, X_savgol1snv_val.T, X_savgol1msc_val.T, X_savgol2snv_val.T, X_savgol2msc_val.T),axis=2)

    return wavelength_red, X_train, X_test

