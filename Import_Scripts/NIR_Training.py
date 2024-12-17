from sys import stdout
try:
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
    from sklearn.svm import SVR
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from NIR_Metrics import *
    from NIR_Plots import *
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
    from sklearn.svm import SVR
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from NIR_Metrics import *
    from NIR_Plots import *

def optimise_calibration(X,y,n_comp,method,all=True,scale=False, kernel='rbf', max_C=1000, eps=0.5, max_trees = 1000, criterion='RMSE', q=0.25, plot_progressval=False, save = False, savename = '', X_test = None, y_test = None):
    """
    Optimizes calibration for Near-Infrared (NIR) spectroscopy models.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input features for calibration.
    y : array-like, shape (n_samples, n_components)
        The target values corresponding to each sample.
    n_comp : int
        The maximum number of components to consider for optimization.
    method : str
        The decomposition/regression method to use. Options include 'PCA', 'PLS', 'SVR', 'Standard-SVR', 'RandomForest'.
    all : bool, default=True
        Whether to optimize over all components collectively (True) or individually per component (False).
    scale : bool, default=False
        Whether to scale the data before applying PLS regression.
    kernel : str, default='rbf'
        The kernel type to be used in SVR. Ignored if the method is not 'SVR' or 'Standard-SVR'.
    max_C : float, default=1000
        The maximum value of the regularization parameter C to consider in SVR methods.
    eps : float, default=0.5
        The epsilon parameter in SVR. Ignored if the method is not 'SVR' or 'Standard-SVR'.
    max_trees : int, default=1000
        The maximum number of trees to consider in RandomForestRegressor.
    criterion : str, default='RMSE'
        The criterion used for determining the optimal number of components. Options include 'RMSE', 'Wold', 'Haaland'.
    q : float, default=0.25
        The quantile value used in the 'Haaland' criterion.
    plot_progressval : bool, default=False
        Whether to plot the progress of the optimization.
    save : bool, default=False
        Whether to save the progress plot.
    savename : str, default=''
        The filename to use when saving the progress plot.
    X_test : array-like, shape (n_test_samples, n_features), optional
        The input features for testing the calibration.
    y_test : array-like, shape (n_test_samples, n_components), optional
        The target values for testing the calibration.
    Returns
    -------
    ideal : int or array-like
        The optimal number of components (or hyperparameter values) determined by the optimization criterion.
    rmse_c : numpy.ndarray
        The root mean squared error for calibration.
    rmse_cv : numpy.ndarray
        The root mean squared error from cross-validation.
    """
    '''General function for optimised NIR calibration -  X (N_samples x N_features), y (N_samples x N_components)
    mode - 'PCA' or 'PLS'; all - over all components or individually?
    Returns optimal # of components, rmse_calibration, and rmse_cv vectors'''

    if all==True:
        rmse_c=np.zeros(n_comp)
        wold_r=np.zeros(n_comp)
    else:
        rmse_c=np.zeros((n_comp,y.shape[1]))
        wold_r=np.zeros((n_comp-1,y.shape[1]))

    rmse_cv=np.zeros_like(rmse_c)
    rmse_c_plot=np.zeros_like(rmse_c)
    rmse_cv_plot=np.zeros_like(rmse_cv)
    rmse_test = np.zeros_like(rmse_cv_plot)
    rss_cv=np.zeros_like(rmse_c)

    if method == 'PCA':
            pipe = Pipeline([('scaler', StandardScaler()), ('decomp', PCA()), ('reg',LinearRegression())])
        
    elif method == 'PLS':
        pipe=PLSRegression(scale=scale, max_iter=2000)
    elif method == 'SVR':
        pipe=SVR(epsilon=eps,kernel=kernel)
    elif method == 'Standard-SVR':
        pipe = Pipeline([('scaler', StandardScaler()), ('reg',SVR(epsilon=eps,kernel=kernel))])
    elif method == 'RandomForest':
        pipe = RandomForestRegressor()

    cv = LeaveOneOut()
    n=y.shape[0]
    iter_range=np.arange(0,n_comp)
    C_range=np.linspace(1,max_C,n_comp)
    trees_range = np.linspace(10, max_trees, n_comp, dtype=int)

    for i in iter_range:      

        if method == 'PCA':
            pipe.set_params(**{'decomp__n_components': np.int64(i)+1})
        elif method == 'PLS':
            pipe.set_params(**{'n_components': np.int64(i)+1})
        elif method == 'SVR':
            pipe.set_params(**{'C': C_range[i]})
        elif method == 'Standard-SVR':
            pipe.set_params(**{'reg__C': C_range[i]})
        elif method == 'RandomForest':
            pipe.set_params(**{'n_estimators': trees_range[i]})


        if all == True:
            pipe.fit(X,y)
            y_c = pipe.predict(X)
            y_cv = cross_val_predict(pipe, X, y, cv=cv)
            _,rmse_c[i] = mse(y,y_c,mode='optimise')
            _,rmse_cv[i] = mse(y,y_cv,mode='optimise')
            _,rmse_c_plot[i] = mse(y,y_c,mode='regular')
            _,rmse_cv_plot[i] = mse(y,y_cv,mode='regular')
            rss_cv[i] = np.sum(np.sum((y-y_cv)**2, axis=0)/np.var(y,axis=0))

            if (X_test is None)==False:
                y_test_pred = pipe.predict(X_test)
                _, rmse_test[i] = mse(y_test, y_test_pred, mode = 'optimise')
            if i!=0:
                wold_r[i-1]=np.divide(rss_cv[i],rss_cv[i-1])
        
        else:
            for j in range(y.shape[1]):
                y_fit=y[:,j]
                pipe.fit(X,y_fit)
                y_c = pipe.predict(X).reshape((-1,1))
                y_cv = cross_val_predict(pipe, X, y_fit, cv=cv).reshape((-1,1))
                y_fit=y_fit.reshape((-1,1))
                _,rmse_c[i,j] = mse(y_fit,y_c,mode='optimise')
                _,rmse_cv[i,j] = mse(y_fit,y_cv,mode='optimise')
                _,rmse_c_plot[i,j] = mse(y_fit,y_c,mode='regular')
                _,rmse_cv_plot[i,j] = mse(y_fit,y_cv,mode='regular')
                rss_cv[i,j]=n*(np.power(rmse_cv_plot[i,j],2))
                if (X_test is None) == False:
                    y_test_pred = pipe.predict(X_test)
                    _, rmse_test[i,j] = mse(y_test[:,j], y_test_pred, mode = 'regular')
            if i!=0:
                wold_r[i-1,:]=np.divide(rss_cv[i,:],rss_cv[i-1,:])
    
    rss_min=np.min(rss_cv,axis=0)
    if criterion== 'RMSE':
        ideal_n=np.argmin(rmse_cv,axis=0)
    elif criterion == 'Wold':
        ideal_n=np.argmax(wold_r>1,axis=0)
    elif criterion =='Haaland':
        rss_ratio=np.divide(rss_cv,rss_min)
        f_crit = f.ppf(1-q,n,n, loc=0, scale=1)
        ideal_n=np.argmax(rss_ratio<f_crit,axis=0)
    
    if method == 'SVR' or method == 'Standard-SVR':
        if type(ideal_n)==np.int64:
            ideal=C_range[ideal_n]
        else:
            ideal=np.zeros((y.shape[1]))
            for j in range(y.shape[1]):
                ideal[j]=C_range[ideal_n[j]]
    elif method == 'RandomForest':
        if type(ideal_n)==np.int64:
            ideal=trees_range[ideal_n]
        else:
            ideal=np.zeros((y.shape[1]))
            for j in range(y.shape[1]):
                ideal[j]=trees_range[ideal_n[j]]
    else:
        ideal=np.int64(ideal_n+1)
    
    if plot_progressval==True:
        plot_progress(iter_range,rmse_c,rmse_cv, rmse_test, ideal, save, savename)

    return ideal, rmse_c, rmse_cv


def simple_pls_pcr(X, y, X_test, y_test, n_comp, method, all = True, kernel='rbf', eps=0.5, scale = False, plot_response = False, plot_val=False, \
    eval_metrics=True, save = False, savename = 'output.png', graphtitle = '',save_val=False, savename_val='output.png', labels=None):
    """
    General function for optimized NIR calibration.
    Parameters:
        X (array-like): Training data features with shape (N_samples, N_features).
        y (array-like): Training targets with shape (N_samples, N_components).
        X_test (array-like): Test data features.
        y_test (array-like): Test data targets.
        n_comp (int or array-like): Number of components to use. Must be a vector for individual mode.
        method (str): Regression method to use. Options include 'PCA', 'PLS', 'SVR', 'Standard-SVR', or 'RandomForest'.
        all (bool, optional): If True, apply the method over all components simultaneously. If False, apply individually per component. Defaults to True.
        kernel (str, optional): Kernel type for SVR. Defaults to 'rbf'.
        eps (float, optional): Epsilon parameter for SVR. Defaults to 0.5.
        scale (bool, optional): Whether to scale data when using PLS. Defaults to False.
        plot_response (bool, optional): If True, plot the response of the calibration. Defaults to False.
        plot_val (bool, optional): If True, plot the validation results. Defaults to False.
        eval_metrics (bool, optional): If True, evaluate and return performance metrics. Defaults to True.
        save (bool, optional): If True, save the calibration plot. Defaults to False.
        savename (str, optional): Filename for saving the calibration plot. Defaults to 'output.png'.
        graphtitle (str, optional): Title for the plots. Defaults to ''.
        save_val (bool, optional): If True, save the validation plot. Defaults to False.
        savename_val (str, optional): Filename for saving the validation plot. Defaults to 'output.png'.
        labels (list, optional): Labels for the plots. Defaults to None.
    Returns:
        If eval_metrics is False:
            tuple: (y_c, y_cv, y_val)
                y_c (array-like): Calibration predictions.
                y_cv (array-like): Cross-validated predictions.
                y_val (array-like): Validation predictions.
        If eval_metrics is True:
            tuple: (rmse_c_comp, rmse_c, rmse_cv_comp, rmse_cv, rmse_val_comp, rmse_val,
                    r2_c_comp, r2_c, r2_cv_comp, r2_cv, r2_val_comp, r2_val,
                    bias_val, se_val, rpd_val)
                Performance metrics for calibration, cross-validation, and validation datasets.
    """
    '''General function for optimised NIR calibration -  X (N_samples x N_features), y (N_samples x N_components)
    mode - 'PCA' or 'PLS'; all - over all components or individually?
    n_comp MUST be a vector for individual mode
    Returns y_c, y_cv, y_val if eval_metrics is False
    if eval_metrics is True: Returns rmse_c_comp, rmse_c, rmse_cv_comp, rmse_cv, rmse_val_comp, rmse_val, r2_c_comp, r2_c, r2_cv_comp, r2_cv, r2_val_comp, r2_val, bias_val, se_val, rpd_val'''
    cv = LeaveOneOut()
    
    if method == 'PCA':
            pipe = Pipeline([('scaler', StandardScaler()), ('decomp', PCA()), ('reg',LinearRegression())])
    elif method == 'PLS':
        pipe=PLSRegression(scale=scale, max_iter=2000)
    elif method == 'SVR':
        pipe=SVR(epsilon=eps,kernel=kernel)
    elif method == 'Standard-SVR':
        pipe = Pipeline([('scaler', StandardScaler()), ('reg',SVR(epsilon=eps,kernel=kernel))])    
    elif method == 'RandomForest':
        pipe = RandomForestRegressor()

    if all == True:
        if isinstance(np.int64(n_comp),np.int64):
            ncomp=np.int64(n_comp)
        elif method == 'SVR':
            pipe.set_params(**{'C': n_comp})
        elif method == 'RandomForest':
            pipe.set_params(**{'n_estimators': n_comp})
        else:
            ncomp=np.int64(n_comp[0])
        
        if method == 'PCA':
            pipe.set_params(**{'decomp__n_components': ncomp})
        elif method == 'PLS':
            pipe.set_params(**{'n_components': ncomp})
        pipe.fit(X,y)
        y_c = pipe.predict(X)
        y_cv = cross_val_predict(pipe, X, y, cv=cv)
        y_val = pipe.predict(X_test)

    else:
        y_c=np.zeros(y.shape)
        y_cv=np.zeros(y.shape)
        y_val=np.zeros(y_test.shape)
        
        for j in range(y.shape[1]):
            
            if method == 'PCA':
                pipe.set_params(**{'decomp__n_components': np.int64(n_comp[j])})
            elif method == 'PLS':
                pipe.set_params(**{'n_components': np.int64(n_comp[j])})
            elif method == 'SVR':
                pipe.set_params(**{'C': n_comp[j]})
            elif method == 'Standard-SVR':
                pipe.set_params(**{'reg__C': n_comp[j]})
            elif method == 'RandomForest':
                pipe.set_params(**{'n_estimators': n_comp[j]})
            
            y_fit=y[:,j]
            pipe.fit(X,y_fit)
            y_c[:,j] = pipe.predict(X).reshape((-1,))
            y_cv[:,j] = cross_val_predict(pipe, X, y_fit, cv=cv).reshape((-1,))
            y_val[:,j] = pipe.predict(X_test).reshape((-1,))

    r2_cv_comp,r2_cv = r2(y,y_cv)
    r2_val_comp,r2_val = r2(y_test,y_val)
    
    if plot_response is True:
        CV_response_plot(y,y_cv,y_c,r2_cv, r2_cv_comp, labels, save=save, savename=savename, graphtitle=graphtitle)
    
    if plot_response is True:
        Val_response_plot(y_test,y_val,r2_val, r2_val_comp, labels, save=save_val, savename=savename_val, graphtitle=graphtitle)
    
    if eval_metrics is False:
        return y_c, y_cv, y_val
    else:
        return metrics(y,y_test,y_c,y_cv,y_val)


def preprocess (preprocess_data, comp_vals_calib, preprocess_test, comp_vals_val, method, all, max_n_comp = 25, scale=False, plot_progressval=False, plot_response = False, eval_metrics = True, criterion_train='RMSE'):
    """
    Preprocesses calibration and validation data using specified methods and parameters.
    This function performs preprocessing on calibration and validation datasets, optimizes the number of components
    based on the provided method and criterion, and optionally evaluates performance metrics.
    Parameters
    ----------
    preprocess_data : np.ndarray
        3D array containing the calibration data to be preprocessed.
    comp_vals_calib : np.ndarray
        2D array of calibration component values.
    preprocess_test : np.ndarray
        3D array containing the test data to be preprocessed.
    comp_vals_val : np.ndarray
        2D array of validation component values.
    method : str
        The method used for calibration and preprocessing (e.g., 'PLS', 'PCR').
    all : bool
        If True, a single number of components is optimized across all datasets; 
        if False, components are optimized per dataset.
    max_n_comp : int, optional
        The maximum number of components to consider during optimization (default is 25).
    scale : bool, optional
        Whether to scale the data before processing (default is False).
    plot_progressval : bool, optional
        If True, plots the progress during calibration optimization (default is False).
    plot_response : bool, optional
        If True, plots the response variables after processing (default is False).
    eval_metrics : bool, optional
        If True, calculates and returns evaluation metrics; 
        if False, returns processed data arrays (default is True).
    criterion_train : str, optional
        The criterion used for training optimization (default is 'RMSE').
    Returns
    -------
    Depending on the value of `eval_metrics`:
        If `eval_metrics` is True:
            tuple
                A tuple containing:
                    n_comp : np.ndarray
                        Array containing the number of components selected for each dataset.
                    rmse_c_comp : np.ndarray
                        Root Mean Square Error for calibration data.
                    rmse_cv_comp : np.ndarray
                        Root Mean Square Error for cross-validation data.
                    rmse_val_comp : np.ndarray
                        Root Mean Square Error for validation data.
                    r2_c_comp : np.ndarray
                        R-squared values for calibration data.
                    r2_cv_comp : np.ndarray
                        R-squared values for cross-validation data.
                    r2_val_comp : np.ndarray
                        R-squared values for validation data.
                    bias_val : np.ndarray
                        Bias values for validation data.
                    se_val : np.ndarray
                        Standard error for validation data.
                    rpd_val : np.ndarray
                        Residual Predictive Deviation for validation data.
                    mae_cv : np.ndarray
                        Mean Absolute Error for cross-validation data.
                    mae_val : np.ndarray
                        Mean Absolute Error for validation data.
                    mode_error_cv : np.ndarray
                        Mode error for cross-validation data.
                    mode_error_val : np.ndarray
                        Mode error for validation data.
                    std_error_cv : np.ndarray
                        Standard deviation of error for cross-validation data.
                    std_error_val : np.ndarray
                        Standard deviation of error for validation data.
        If `eval_metrics` is False:
            tuple
                A tuple containing:
                    y_c : np.ndarray
                        Processed calibration data.
                    y_cv : np.ndarray
                        Processed cross-validation data.
                    y_val : np.ndarray
                        Processed validation data.
    """
    comp=comp_vals_calib.shape[1]
    n_train = comp_vals_calib.shape[0]
    n_test = comp_vals_val.shape[0]

    num_data=preprocess_data.shape[2]
    if all == True:
        n_comp=np.zeros((1,num_data))
    else:
        n_comp=np.zeros((comp,num_data))
    
    if eval_metrics is True:
        r2_c_comp=np.zeros((comp,num_data)); r2_cv_comp=np.zeros((comp,num_data)); r2_val_comp=np.zeros((comp,num_data))
        rmse_c_comp=np.zeros((comp,num_data)); rmse_cv_comp=np.zeros((comp,num_data));  rmse_val_comp=np.zeros((comp,num_data))
        bias_val=np.zeros((comp,num_data)); se_val=np.zeros((comp,num_data)); rpd_val=np.zeros((comp,num_data))
        mae_cv=np.zeros((comp,num_data)); mae_val=np.zeros((comp,num_data)); mode_error_cv=np.zeros((comp,num_data)); mode_error_val=np.zeros((comp,num_data))
        std_error_cv=np.zeros((comp,num_data)); std_error_val=np.zeros((comp,num_data))
    else:
        y_c = np.zeros((n_train,comp,num_data)); y_cv = np.zeros((n_train,comp,num_data)); y_val = np.zeros((n_test,comp,num_data))

    y = comp_vals_calib[:]; y_test = comp_vals_val[:]

    for n in range(num_data):
        X=preprocess_data[:,:,n]
        X_test=preprocess_test[:,:,n]
        n_comp[:,n], _, _ = optimise_calibration(X, y, max_n_comp, method,all=all,scale=scale,plot_progressval=plot_progressval,criterion=criterion_train)
        output = simple_pls_pcr(X, y, X_test, y_test, n_comp[:,n], method, all = all, scale = scale, \
            plot_response = plot_response, eval_metrics=eval_metrics)
        if eval_metrics is True:
            rmse_c_comp[:,n] = output[0]; rmse_cv_comp[:,n] = output[2]; rmse_val_comp[:,n] = output[4]
            r2_c_comp[:,n] = output[6]; r2_cv_comp[:,n] = output[8]; r2_val_comp[:,n] = output[10]
            bias_val[:,n] = output[12]; se_val[:,n] = output[13]; rpd_val[:,n] = output[14]
            mae_cv[:,n] = output[15] ; mae_val[:,n] = output[16]; mode_error_cv[:,n] = output[17]; mode_error_val[:,n] = output[18]; std_error_cv[:,n] = output[19]; std_error_val[:,n] = output[20]
        else:
            y_c[:,:,n], y_cv[:,:,n], y_val[:,:,n] = output

    if eval_metrics is True:
        return n_comp, rmse_c_comp, rmse_cv_comp, rmse_val_comp, r2_c_comp, r2_cv_comp, r2_val_comp,  bias_val, se_val, rpd_val\
            , mae_cv, mae_val, mode_error_cv, mode_error_val, std_error_cv, std_error_val
    else:
        return y_c, y_cv, y_val

def ideal_model(X, y, X_test, y_test, n_comp, method, all = True, kernel='rbf', eps=0.5, scale = False, eval_metrics=True):
    """
    General function for optimized NIR calibration.
    Parameters:
        X (np.ndarray): Training data features, shape (N_samples, N_features).
        y (np.ndarray): Training data targets, shape (N_samples, N_components).
        X_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test data targets.
        n_comp (int or list of int): Number of components to use.
            - If `all` is True and not using SVR or RandomForest, `n_comp` must be a single integer.
            - If `all` is False, `n_comp` must be a list of integers, one for each component.
        method (str): Method to use for calibration. Options include:
            - 'PCA'
            - 'PLS'
            - 'SVR'
            - 'Standard-SVR'
            - 'RandomForest'
        all (bool, optional): If True, applies the method over all components simultaneously.
            If False, applies the method to each component individually. Defaults to True.
        kernel (str, optional): Kernel type for SVR. Applicable if `method` is 'SVR' or 'Standard-SVR'.
            Defaults to 'rbf'.
        eps (float, optional): Epsilon parameter for SVR. Applicable if `method` is 'SVR' or 'Standard-SVR'.
            Defaults to 0.5.
        scale (bool, optional): If True, scales the data for PLSRegression. Applicable if `method` is 'PLS'.
            Defaults to False.
        eval_metrics (bool, optional): If True, computes evaluation metrics.
            If False, returns only predictions. Defaults to True.
    Returns:
        tuple:
            models (list): Fitted models for each component or a single model if `all` is True.
            predictions (tuple):
                If `eval_metrics` is False:
                    - y_c (np.ndarray): Predicted targets on training data.
                    - y_cv (np.ndarray): Cross-validated predicted targets.
                    - y_val (np.ndarray): Predicted targets on test data.
                If `eval_metrics` is True:
                    - rmse_c_comp (list): RMSE for each component on training data.
                    - rmse_c (float): Overall RMSE on training data.
                    - rmse_cv_comp (list): RMSE for each component in cross-validation.
                    - rmse_cv (float): Overall RMSE in cross-validation.
                    - r2_c_comp (list): R² for each component on training data.
                    - r2_c (float): Overall R² on training data.
                    - r2_cv_comp (list): R² for each component in cross-validation.
                    - r2_cv (float): Overall R² in cross-validation.
                    - r2_val_comp (list): R² for each component on test data.
                    - r2_val (float): Overall R² on test data.
                    - bias_val (float): Bias on test data.
                    - se_val (float): Standard error on test data.
                    - rpd_val (float): RPD on test data.
    """
    '''General function for optimised NIR calibration -  X (N_samples x N_features), y (N_samples x N_components)
    mode - 'PCA' or 'PLS'; all - over all components or individually?
    n_comp MUST be a vector for individual mode
    Returns y_c, y_cv, y_val if eval_metrics is False
    if eval_metrics is True: Returns rmse_c_comp, rmse_c, rmse_cv_comp, rmse_cv, rmse_val_comp, rmse_val, r2_c_comp, r2_c, r2_cv_comp, r2_cv, r2_val_comp, r2_val, bias_val, se_val, rpd_val'''
    cv = LeaveOneOut()
    models = []
    
    if method == 'PCA':
            pipe = Pipeline([('scaler', StandardScaler()), ('decomp', PCA()), ('reg',LinearRegression())])
    elif method == 'PLS':
        pipe=PLSRegression(scale=scale, max_iter=2000)
    elif method == 'SVR':
        pipe=SVR(epsilon=eps,kernel=kernel)
    elif method == 'Standard-SVR':
        pipe = Pipeline([('scaler', StandardScaler()), ('reg',SVR(epsilon=eps,kernel=kernel))])    
    elif method == 'RandomForest':
        pipe = RandomForestRegressor()

    if all == True:
        if isinstance(np.int64(n_comp),np.int64):
            ncomp=np.int64(n_comp)
        elif method == 'SVR':
            pipe.set_params(**{'C': n_comp})
        elif method == 'RandomForest':
            pipe.set_params(**{'n_estimators': n_comp})
        else:
            ncomp=np.int64(n_comp[0])
        
        if method == 'PCA':
            pipe.set_params(**{'decomp__n_components': ncomp})
        elif method == 'PLS':
            pipe.set_params(**{'n_components': ncomp})
        pipe.fit(X,y)
        y_c = pipe.predict(X)
        y_cv = cross_val_predict(pipe, X, y, cv=cv)
        y_val = pipe.predict(X_test)
        models.append(pipe)

    else:
        y_c=np.zeros(y.shape)
        y_cv=np.zeros(y.shape)
        y_val=np.zeros(y_test.shape)
        
        for j in range(y.shape[1]):
            
            if method == 'PCA':
                pipe.set_params(**{'decomp__n_components': np.int64(n_comp[j])})
            elif method == 'PLS':
                pipe.set_params(**{'n_components': np.int64(n_comp[j])})
            elif method == 'SVR':
                pipe.set_params(**{'C': n_comp[j]})
            elif method == 'Standard-SVR':
                pipe.set_params(**{'reg__C': n_comp[j]})
            elif method == 'RandomForest':
                pipe.set_params(**{'n_estimators': n_comp[j]})
            
            y_fit=y[:,j]
            pipe.fit(X,y_fit)
            y_c[:,j] = pipe.predict(X).reshape((-1,))
            y_cv[:,j] = cross_val_predict(pipe, X, y_fit, cv=cv).reshape((-1,))
            y_val[:,j] = pipe.predict(X_test).reshape((-1,))
            models.append(pipe)
    
    if eval_metrics is False:
        return models , (y_c,y_cv,y_val)
    else:
        return models, metrics(y,y_test,y_c,y_cv,y_val)