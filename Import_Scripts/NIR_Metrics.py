from sys import stdout
try:
    import numpy as np
    from scipy.signal import savgol_filter, detrend
    from scipy.signal.windows import general_gaussian
    from scipy.stats import f
    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_squared_error, max_error, mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, cross_val_score
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import numpy as np
    from scipy.signal import savgol_filter, detrend
    from scipy.signal.windows import general_gaussian
    from scipy.stats import f
    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_squared_error, max_error, mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, cross_val_score

def mse(y,y_pred,n_lv=None,mode='optimise'):
    '''function to calculate RMSE for different modes - optimise (default) returns a weighted RMSE
    y and y_pred must be (n_samples x n_components)
    pls-calibration returns rmse_c, which has a different formula
    returns both RMSE vector for individual components as well as overall calculated RMSE'''
    if y.shape[1]>1:
        y=y
        y_pred=y_pred
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    n=y.shape[0]
    rmse=np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        std=np.std(y[:,i])
        rmse[i]=np.sqrt(np.sum(np.power((y[:,i]-y_pred[:,i]),2))/n)
        if mode =='optimise':
            rmse[i]=rmse[i]/std
        elif mode=='pls-calibration':
            rmse[i]=np.sqrt(np.sum(np.power((y[:,i]-y_pred[:,i]),2))/(n-(n_lv+1)))
    rmse_overall=np.sum(rmse)

    return rmse, rmse_overall

def r2(y,y_pred):
    '''function to calculate; y and y_pred must be (n_samples x n_components)
    returns both R2 vector for individual components as well as overall calculated R2'''
    if y.shape[1]>1:
        y=y
        y_pred=y_pred
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    
    r2=np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        r2[i]=1-(np.sum(np.power((y[:,i]-y_pred[:,i]),2))/np.sum(np.power((y[:,i]-np.mean(y[:,i])),2)))
    r2_overall=np.mean(r2)

    return r2, r2_overall

def bias(y,y_pred):
    '''function to calculate bias; y and y_pred must be (n_samples x n_components)
    returns both bias vector for individual components as well as overall calculated bias'''
    if y.shape[1]>1:
        y=y
        y_pred=y_pred
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    
    bias=np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        bias[i]=np.mean(y_pred[:,i]-y[:,i])
    bias_overall=np.mean(bias)

    return bias, bias_overall

def standard_error(rmse,bias):
    '''function to calculate SEP; rmse and bias must be (n_components x 1)
    returns both standard error vector for individual components as well as overall calculated SE'''
    
    standard_error=np.sqrt(np.square(rmse)-np.square(bias))
    standard_error_overall=np.mean(standard_error)

    return standard_error, standard_error_overall

def rpd(standard_error, y):
    '''function to calculate RPD; standard_error must be (n_components x 1) and y (n_samples x n_components)
    returns both RPD vector for individual components as well as overall calculated RPD'''
    std = np.std(y, axis = 0)
    rpd = np.divide(std,standard_error)
    rpd_overall=np.mean(rpd)

    return rpd, rpd_overall

def max_error_all(y,y_pred):
    '''function to calculate max error; y and y_pred must be (n_samples x n_components)
    returns both max error vector for individual components as well as overall calculated max error'''
    if y.shape[1]>1:
        y=y
        y_pred=y_pred
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    
    max_error=np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        max_error[i]=np.max(np.abs(y[:,i]-y_pred[:,i]))

    return max_error

def mode_error_all(y,y_pred):
    '''function to calculate mode of the error distribution; y and y_pred must be (n_samples x n_components)
    returns mode error vector for individual components and standard deviation of the error distribution for individual components'''
    if y.shape[1]>1:
        y=y
        y_pred=y_pred
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_pred=y_pred.reshape((-1,1))

    err_data =  np.abs(y-y_pred)    
    iqr = np.percentile(err_data, 75,axis=0) - np.percentile(err_data, 25,axis=0)
    n = y.shape[0]
    binwidth = 2 * iqr / (n**(1/3))

    mode_error=np.zeros(y.shape[1]) ; std_error=np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        bins=np.arange(0,max(err_data[:,i])+1.5,binwidth[i])
        hist, bin_edges = np.histogram(err_data[:,i], bins=bins)
        max_bin = np.argmax(hist)
        # The mode is the value at the center of the bin with the maximum count
        mode_error[i] = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
        std_error[i] = np.std(err_data[:,i])
    return mode_error, std_error

def metrics(y,y_test,y_c,y_cv,y_val):
    """
    Compute various regression metrics for model evaluation.
    Parameters:
        y (array-like): True target values.
        y_test (array-like): True target values for the test set.
        y_c (array-like): Predicted values from component C.
        y_cv (array-like): Predicted values from cross-validation.
        y_val (array-like): Predicted values from validation.
    Returns:
        tuple: A tuple containing the following metrics:
            - rmse_c_comp (float): RMSE for component C comparison.
            - rmse_c (float): Regular RMSE for component C.
            - rmse_cv_comp (float): RMSE for cross-validation comparison.
            - rmse_cv (float): Regular RMSE for cross-validation.
            - rmse_val_comp (float): RMSE for validation comparison.
            - rmse_val (float): Regular RMSE for validation.
            - r2_c_comp (float): R-squared for component C comparison.
            - r2_c (float): Regular R-squared for component C.
            - r2_cv_comp (float): R-squared for cross-validation comparison.
            - r2_cv (float): Regular R-squared for cross-validation.
            - r2_val_comp (float): R-squared for validation comparison.
            - r2_val (float): Regular R-squared for validation.
            - bias_val (float): Bias of the validation predictions.
            - se_val (float): Standard error of the validation RMSE and bias.
            - rpd_val (float): Residual Predictive Deviation for validation.
            - mae_cv (array-like): Mean Absolute Errors for cross-validation.
            - mae_val (array-like): Mean Absolute Errors for validation.
            - mode_error_cv (float): Mode error for cross-validation.
            - mode_error_val (float): Mode error for validation.
            - std_error_cv (float): Standard error for cross-validation mode error.
            - std_error_val (float): Standard error for validation mode error.
    """
    rmse_c_comp,rmse_c = mse(y,y_c,mode='regular')
    rmse_cv_comp,rmse_cv = mse(y,y_cv,mode='regular')
    rmse_val_comp,rmse_val = mse(y_test,y_val,mode='regular')
    r2_c_comp,r2_c = r2(y,y_c)
    r2_cv_comp,r2_cv = r2(y,y_cv)
    r2_val_comp,r2_val = r2(y_test,y_val)
    
    bias_val, _ = bias(y_test, y_val)
    se_val, _ = standard_error(rmse_val_comp, bias_val)
    rpd_val, _ = rpd(se_val, y)

    mode_error_cv, std_error_cv = mode_error_all(y,y_cv)
    mode_error_val, std_error_val = mode_error_all(y_test,y_val)

    mae_cv = mean_absolute_error(y,y_cv, multioutput='raw_values')
    mae_val = mean_absolute_error(y_test,y_val, multioutput='raw_values')

    return rmse_c_comp, rmse_c, rmse_cv_comp, rmse_cv, rmse_val_comp, rmse_val,\
         r2_c_comp, r2_c, r2_cv_comp, r2_cv, r2_val_comp, r2_val,\
         bias_val, se_val, rpd_val, mae_cv, mae_val, mode_error_cv, mode_error_val, std_error_cv, std_error_val

