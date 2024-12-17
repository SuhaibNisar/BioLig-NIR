try:
    from sys import stdout
    import pandas as pd
    import numpy as np
    import copy
    from scipy.signal import savgol_filter, detrend
    from scipy.signal.windows import general_gaussian
    from scipy.stats import f, t
    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict,cross_val_score
    from NIR_Metrics import *
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
    from matplotlib import rcParams, cycler
    import matplotlib.patches as mpatches
    from matplotlib import colors
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from scipy.stats import f, t
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter, detrend
    from scipy.signal.windows import general_gaussian
    from scipy.stats import f
    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict,cross_val_score
    from NIR_Metrics import *
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
    from matplotlib import rcParams, cycler
    import matplotlib.patches as mpatches
    from matplotlib import colors


plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600
##Figure Settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.1
rcParams['axes.labelpad'] = 5.0
rcParams['axes.labelsize'] = 12.5
rcParams['axes.titlesize'] = 13.5

plot_color_cycle = cycler('color', ['#0000a7', '#e0a24c', '#861b1d', '#008176','darkgray'])
rcParams['axes.prop_cycle'] = plot_color_cycle
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
rcParams["legend.frameon"] = False
rcParams["legend.title_fontsize"] = 7
rcParams["legend.borderpad"] = 0
rcParams.update({"figure.figsize" : (6.4,4.8),
                 "figure.subplot.left" : 0.177, "figure.subplot.right" : 0.946,
                 "figure.subplot.bottom" : 0.156, "figure.subplot.top" : 0.965,
                 "axes.autolimit_mode" : "round_numbers",
                 "xtick.major.size"     : 7,
                 "xtick.minor.size"     : 3.5,
                 "xtick.major.width"    : 1.1,
                 "xtick.minor.width"    : 1.1,
                 "xtick.major.pad"      : 5,
                 "xtick.minor.visible"  : True,
                 'xtick.labelsize'      : 11,
                 "ytick.major.size"     : 7,
                 "ytick.minor.size"     : 3.5,
                 "ytick.major.width"    : 1.1,
                 "ytick.minor.width"    : 1.1,
                 "ytick.major.pad"      : 5,
                 "ytick.minor.visible"  : True,
                 'ytick.labelsize'      : 11,
                 "lines.markersize" : 10,
                 "lines.markerfacecolor" : "none",
                 "lines.markeredgewidth"  : 0.8})

def plot_progress(component,rmse_c,rmse_cv, rmse_val, haaland_n, save=False, savename=''):
    mse_min=np.argmin(rmse_cv,axis=0)+1
    component +=1
    fig=plt.figure()
    ax1 = plt.subplot(111)
    plt.plot(component,rmse_c, '-v', color = '#e0a24c', mfc='#e0a24c', label = 'Calibration')
    plt.plot(haaland_n, rmse_c[haaland_n-1], 'P', ms=10, mfc='orangered')
    plt.plot(mse_min, rmse_c[mse_min-1], 'X', ms=10, mfc='cornflowerblue')
    plt.plot(component,rmse_cv, '-^', color = '#0000a7', mfc='#0000a7', label = 'Cross-validation')
    plt.plot(haaland_n, rmse_cv[haaland_n-1], 'P', ms=10, mfc='orangered')
    plt.plot(mse_min, rmse_cv[mse_min-1], 'X', ms=10, mfc='cornflowerblue')
    plt.plot(component,rmse_val, '-o', color = '#861b1d', mfc='#861b1d', label = 'Testing')
    plt.plot(haaland_n, rmse_val[haaland_n-1], 'P', ms=10, mfc='orangered')
    plt.plot(mse_min, rmse_val[mse_min-1], 'X', ms=10, mfc='cornflowerblue')


    plt.plot([mse_min, mse_min],ax1.get_ylim(),'--', color = 'cornflowerblue')
    plt.plot([haaland_n, haaland_n],ax1.get_ylim(),'--', color = 'orangered')
    plt.text(mse_min+0.1,ax1.get_ylim()[1]*0.8,'Minimum\nCross-Validation', ha='left', color = 'cornflowerblue')
    plt.text(haaland_n-0.1,ax1.get_ylim()[1]*0.7,'Haaland\nCriterion', ha='right', color = 'orangered')
    plt.ylabel('Weighted Root Mean Square Error (RMSE)')

    plt.xlabel('Number of Latent Variables')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(n=2))    
    plt.legend()
    if save:
        plt.savefig(savename+'_progress.png', pad_inches=0,bbox_inches='tight')


def CV_response_plot(y,y_cv,y_c,score_cv,r2_cv_comp,labels=None,save=False,savename='output.png', graphtitle=''):
    """
    Plot regression results and figures of merit.
    Parameters:
        y (numpy.ndarray): Actual measured values. Shape (n_samples, n_features).
        y_cv (numpy.ndarray): Cross-validated predicted values.
        y_c (numpy.ndarray): Another set of predicted values for comparison.
        score_cv (float): Cross-validation score.
        r2_cv_comp (list or numpy.ndarray): R² values for each component in cross-validation.
        labels (list, optional): Labels for each feature. Defaults to ['Glucan', 'Hemicellulose', 'Lignin'].
        save (bool, optional): If True, saves the plot to a file. Defaults to False.
        savename (str, optional): Filename to save the plot. Defaults to 'output.png'.
        graphtitle (str, optional): Title of the graph. Defaults to ''.
    Returns:
        None. Displays a plot and optionally saves it to a file.
    """
    # Plot regression and figures of merit
    if y.shape[1]>1:
        y=y
        y_c=y_c
        y_cv=y_cv
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_c=y_c.reshape((-1,1))
        y_cv=y_cv.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_c=y_c.reshape((-1,1))
        y_cv=y_cv.reshape((-1,1))
    n,dim=y.shape
    if labels==None:
        labels = ['Glucan','Hemicellulose','Lignin']

    # Fit a line to the CV vs response
    fig, ax = plt.subplots(figsize=(6, 5))
    cycler = ax._get_lines.prop_cycler
    cycler = ["green","#d9d900","navy"]
    for i in range(dim):
        col = cycler[i]
        ax.scatter(y[:,i], y_c[:,i], s=70, edgecolors=col, facecolors='none', label=labels[i]+' - $R^{2}_{CV}$= '+str(round(r2_cv_comp[i],3)))
        z = np.polyfit(y[:,i], y_cv[:,i], 1)
        #Plot the best fit line
        ax.plot(y[:,i], np.polyval(z,y[:,i]), linewidth=1, color = col)
    #Plot the ideal 1:1 line
    ax.plot(np.linspace(np.min(y),np.max(y),10), np.linspace(np.min(y),np.max(y),10), color='black', linestyle='dashed', linewidth=2, label='Parity line')
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.xlabel('Measured (%)')
    plt.ylabel('Predicted (%)')
    plt.legend()
    ax.set_xlim([(min(np.min(y),np.min(y_c))-0.5), (max(np.max(y),np.max(y_c))+0.5)])
    ax.set_ylim([(min(np.min(y),np.min(y_c))-0.5), (max(np.max(y),np.max(y_c))+0.5)])
    if save:
        plt.savefig(savename, pad_inches=0,bbox_inches='tight') 
    plt.show()    

def Val_response_plot(y_test,y_val,score_val,r2_val_comp,labels=None,save=False,savename='output.png', graphtitle=''):
    """
    Plot regression results and figures of merit.
    Parameters:
        y_test (numpy.ndarray): Actual measured values. Shape should be (n_samples, n_features).
        y_val (numpy.ndarray): Predicted validation values. Shape should match y_test.
        score_val (float): Validation score metric.
        r2_val_comp (list or numpy.ndarray): R² values for each component.
        labels (list, optional): Labels for each component. Defaults to ['Glucan', 'Hemicellulose', 'Lignin'].
        save (bool, optional): If True, saves the plot to a file. Defaults to False.
        savename (str, optional): Filename for saving the plot. Defaults to 'output.png'.
        graphtitle (str, optional): Title of the graph. Defaults to ''.
    Returns:
        None
    """
    # Plot regression and figures of merit
    if y_test.shape[1]>1:
        y_test=y_test
        y_val=y_val
    elif y_test.shape[1]==1:
        y_test=y_test.reshape((-1,1))
        y_val=y_val.reshape((-1,1))
    else:
        y_test=y_test.reshape((-1,1))
        y_val=y_val.reshape((-1,1))
    n,dim=y_test.shape
    if labels==None:
        labels=['Glucan','Hemicellulose','Lignin']

    # Fit a line to the Val vs response
    fig, ax = plt.subplots(figsize=(6, 5))
    cycler = ["green","#d9d900","navy"]
    for i in range(dim):
        col = cycler[i]
        ax.scatter(y_test[:,i], y_val[:,i], s=70, edgecolors=col, facecolors='none', label=labels[i]+' - $R^{2}_{test}$= '+str(round(r2_val_comp[i],3)))
        z = np.polyfit(y_test[:,i], y_val[:,i], 1)
        #Plot the best fit line
        ax.plot(y_test[:,i], np.polyval(z,y_test[:,i]), linewidth=1, color = col)
    #Plot the ideal 1:1 line
    ax.plot(np.linspace(np.min(y_test),np.max(y_test),10), np.linspace(np.min(y_test),np.max(y_test),10), color='black', linestyle='dashed', linewidth=2, label='Parity line')
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.xlabel('Measured (%)')
    plt.ylabel('Predicted (%)')
    plt.legend()
    ax.set_xlim([(min(np.min(y_test),np.min(y_val))-0.5), 100])
    ax.set_ylim([(min(np.min(y_test),np.min(y_val))-0.5), 100])
    if save:
        plt.savefig(savename, pad_inches=0,bbox_inches='tight') 
    plt.show() 

def CV_response_plot_type(y,y_cv,y_c,type_calib,labels='',save=False,savename='output.png', graphtitle=''):
    """
    Generate a cross-validation response plot categorized by calibration types.
    Parameters:
        y (numpy.ndarray): Actual measured values.
        y_cv (numpy.ndarray): Cross-validated predicted values.
        y_c (numpy.ndarray): Corrected or alternative predicted values.
        type_calib (numpy.ndarray): Categories for calibration types.
        labels (str, optional): Label prefix for the best fit line. Defaults to ''.
        save (bool, optional): Whether to save the plot as a file. Defaults to False.
        savename (str, optional): Filename to save the plot. Defaults to 'output.png'.
        graphtitle (str, optional): Title of the graph. Defaults to ''.
    Returns:
        None: Displays the plot and optionally saves it to a file.
    """
    # Plot regression and figures of merit
    if y.shape[1]>1:
        y=y
        y_c=y_c
        y_cv=y_cv
    elif y.shape[1]==1:
        y=y.reshape((-1,1))
        y_c=y_c.reshape((-1,1))
        y_cv=y_cv.reshape((-1,1))
    else:
        y=y.reshape((-1,1))
        y_c=y_c.reshape((-1,1))
        y_cv=y_cv.reshape((-1,1))
    n,dim=y.shape
    _,r2_cv_comp = r2(y,y_cv)
    type_cat=np.unique(type_calib)

    # Fit a line to the CV vs response
    fig, ax = plt.subplots(figsize=(6, 5))
    cycler = ax._get_lines.prop_cycler
    for i in type_cat:
        col=next(cycler)['color']
        mask=type_calib==i
        ax.scatter(y[mask], y_c[mask], s=70, edgecolors=col, facecolors='none', label=i)
    y=y.reshape((-1,))
    z = np.polyfit(y, y_cv, 1)
    #Plot the best fit line
    ax.plot(y, np.polyval(z,y), linewidth=1, color = 'indigo',label=labels+' - $R^{2}_{CV}$= '+str(round(r2_cv_comp,3)))
    #Plot the ideal 1:1 line
    ax.plot(np.linspace(np.min(y),np.max(y),10), np.linspace(np.min(y),np.max(y),10), color='black', linestyle='dashed', linewidth=2, label='Parity line')
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    #plt.title(graphtitle + '$R^{2}_{CV}$: '+str(round(score_cv,3)))
    plt.xlabel('Measured (%)')
    plt.ylabel('Predicted (%)')
    plt.legend()
    ax.set_xlim([(min(np.min(y),np.min(y_c))-0.5), (max(np.max(y),np.max(y_c))+0.5)])
    ax.set_ylim([(min(np.min(y),np.min(y_c))-0.5), (max(np.max(y),np.max(y_c))+0.5)])
    if save:
        plt.savefig(savename, pad_inches=0,bbox_inches='tight') 
    plt.show()    

def Val_response_plot_type(y_test,y_val,type_valid,labels='',save=False,savename='output.png', graphtitle=''):
    """
    Generates and displays a validation response plot with regression and figures of merit.
    This function creates a scatter plot of measured versus predicted values, categorized by type.
    It fits a regression line, plots the ideal parity line, and optionally saves the plot as an image.
    Parameters
    ----------
    y_test : array-like
        Measured values. Should be a 1D or 2D array with one column.
    y_val : array-like
        Predicted or validation values. Should be a 1D or 2D array with one column, matching y_test.
    type_valid : array-like
        Categorical labels used to group the data points in the plot.
    labels : str, optional
        Label for the regression line in the plot legend (default is '').
    save : bool, optional
        If True, saves the plot to a file specified by savename (default is False).
    savename : str, optional
        The filename for saving the plot (default is 'output.png').
    graphtitle : str, optional
        Title of the graph (currently not displayed as plt.title is commented out).
    Returns
    -------
    None
    Raises
    ------
    ValueError
        If y_test and y_val do not have compatible shapes.
    """
    # Plot regression and figures of merit
    if y_test.shape[1]>1:
        y_test=y_test
        y_val=y_val
    elif y_test.shape[1]==1:
        y_test=y_test.reshape((-1,1))
        y_val=y_val.reshape((-1,1))
    else:
        y_test=y_test.reshape((-1,1))
        y_val=y_val.reshape((-1,1))
    n,dim=y_test.shape
    
    _,r2_val_comp = r2(y_test,y_val)   
    type_cat=np.unique(type_valid)

    # Fit a line to the Val vs response
    fig, ax = plt.subplots(figsize=(6, 5))
    cycler = ax._get_lines.prop_cycler
    for i in type_cat:
        col=next(cycler)['color']
        mask = type_valid==i
        ax.scatter(y_test[mask], y_val[mask], s=70, edgecolors=col, facecolors='none', label=i)
    y_test=y_test.reshape((-1,))
    z = np.polyfit(y_test, y_val, 1)
    #Plot the best fit line
    ax.plot(y_test, np.polyval(z,y_test), linewidth=1, color = 'indigo', label=labels+' - $R^{2}_{Val}$= '+str(round(r2_val_comp,3)))
    #Plot the ideal 1:1 line
    ax.plot(np.linspace(np.min(y_test),np.max(y_test),10), np.linspace(np.min(y_test),np.max(y_test),10), color='black', linestyle='dashed', linewidth=2, label='Parity line')
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    #plt.title(graphtitle + '$R^{2}_{Validation}$: '+str(round(score_val,3)))
    plt.xlabel('Measured (%)')
    plt.ylabel('Predicted (%)')
    plt.legend()
    ax.set_xlim([(min(np.min(y_test),np.min(y_val))-0.5), (max(np.max(y_test),np.max(y_val))+0.5)])
    ax.set_ylim([(min(np.min(y_test),np.min(y_val))-0.5), (max(np.max(y_test),np.max(y_val))+0.5)])
    if save:
        plt.savefig(savename, pad_inches=0,bbox_inches='tight') 
    plt.show()

def typeseveritylabels(type_calib,type_valid,severity_calib,severity_valid,plot):
    """
    Selects and returns calibration and validation labels based on the specified plot type.

    Parameters:
        type_calib (list): Calibration labels for types.
        type_valid (list): Validation labels for types.
        severity_calib (list): Calibration labels for severity levels.
        severity_valid (list): Validation labels for severity levels.
        plot (str): The type of plot to generate ('type' or 'severity').

    Returns:
        tuple: A tuple containing the selected calibration labels and validation labels.

    Raises:
        Exception: If an invalid plot type is provided.
    """
    if plot == 'type':
        calib_labels=type_calib
        valid_labels=type_valid
    elif plot == 'severity':
        calib_labels=severity_calib
        valid_labels=severity_valid
    else:
        raise Exception("Incorrect plot type given")
    return calib_labels, valid_labels

def response_plot_type(y,y_pred,type_calib,labels='',subsc = 'cal',save=False,savename='output', display_fig = True):
    """
    Plots regression results and figures of merit.
    This function generates scatter plots of measured versus predicted values, 
    categorized by calibration type. It fits a regression line, plots an ideal 
    parity line, and displays figures of merit such as R² and RMSE for each dimension. 
    The plot can be saved to a file and/or displayed.
    Parameters:
        y (np.ndarray): Array of measured values with shape (n_samples, n_dimensions).
        y_pred (np.ndarray): Array of predicted values with shape (n_samples, n_dimensions).
        type_calib (np.ndarray): Array of calibration types/categories with shape (n_samples,).
        labels (str or list of str, optional): Labels for each dimension. Defaults to ''.
        subsc (str, optional): Subscript identifier for figures of merit. Defaults to 'cal'.
        save (bool, optional): Whether to save the plot as a file. Defaults to False.
        savename (str, optional): Filename for saving the plot. Defaults to 'output'.
        display_fig (bool, optional): Whether to display the plot. Defaults to True.
    Returns:
        None
    """
    # Plot regression and figures of merit
    _,dim=y.shape
    r2_comp,_ = r2(y,y_pred)
    rmse_comp,_ = mse(y,y_pred,mode='regular')
    type_cat=np.unique(type_calib)

    # Fit a line to the CV vs response
    fig=plt.figure(figsize=(6*dim,5)); axes=fig.subplots(1,dim)
    for ind,ax in enumerate(axes):
        cycler = ax._get_lines.prop_cycler
        for i in type_cat:
            col=next(cycler)['color']
            mask=type_calib==i
            ax.scatter(y[mask,ind], y_pred[mask,ind], s=70, edgecolors=col, facecolors='none', label=i)
        z = np.polyfit(y[:,ind], y_pred[:,ind], 1)
        #Plot the best fit line
        ax.plot(y[:,ind], np.polyval(z,y[:,ind]), linewidth=1, color = 'indigo')
        #Plot the ideal 1:1 line
        ax.plot(np.linspace(np.min(y[:,ind]),np.max(y[:,ind]),10), np.linspace(np.min(y[:,ind]),np.max(y[:,ind]),10), color='black', linestyle='dashed', linewidth=2, label='Parity line')
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        ax.set_xlabel('Measured (%)')
        ax.set_ylabel('Predicted (%)')
        ax.set_xlim([(min(np.min(y[:,ind]),np.min(y_pred[:,ind]))-0.5), (max(np.max(y[:,ind]),np.max(y_pred[:,ind]))+0.5)])
        ax.set_ylim([(min(np.min(y[:,ind]),np.min(y_pred[:,ind]))-0.5), (max(np.max(y[:,ind]),np.max(y_pred[:,ind]))+0.5)])

        extra_handle = plt.Line2D([], [], linestyle='')
        extra_label = ['$R^{2}$'+'$_{'+subsc+'}$'+'= '+str(round(r2_comp[ind],3))]
        extra_label.append('$RMSE$'+'$_{'+subsc+'}$'+'= '+str(round(rmse_comp[ind],3))+' %')
        ax.legend([extra_handle, extra_handle], extra_label, loc='upper left', title=labels[ind], title_fontproperties = {'size': 11.5, 'weight':'bold'})

    handles, labels_leg = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels_leg, ncol = type_cat.shape[0]+2, loc ='upper center',fontsize=12.5, borderpad = -0.35)
    if save:
        plt.savefig(savename+'LabelledParity.png', pad_inches=0,bbox_inches='tight')
    if display_fig:
        plt.show()
    else:
        plt.close() 
    # plt.show()    

def rmse_type_plot(y,y_c,y_cv,y_test,y_val,type_calib,type_valid,labels,mae_plot = False, save = False, savename = '',display_fig = True):
    """
    Generate bar plots of RMSE or MAE for different types across calibration, cross-validation, and testing datasets.
    Parameters
    ----------
    y : array-like
        True target values for calibration.
    y_c : array-like
        Predicted values for calibration.
    y_cv : array-like
        Predicted values for cross-validation.
    y_test : array-like
        True target values for testing.
    y_val : array-like
        Predicted values for testing.
    type_calib : array-like
        Type categories for calibration data.
    type_valid : array-like
        Type categories for validation data.
    labels : list
        List of labels for each target variable.
    mae_plot : bool, optional
        If True, plot Mean Absolute Error instead of RMSE. Default is False.
    save : bool, optional
        If True, save the plot as an png file. Default is False.
    savename : str, optional
        Filename for saving the plot (without extension). Default is ''.
    display_fig : bool, optional
        If True, display the figure. If False, close the figure. Default is True.
    Returns
    -------
    None
    """
    type_cat=np.unique(type_calib)
    mae_c_ind = np.zeros((len(labels),len(type_cat))) ; rmse_c_ind = np.zeros((len(labels),len(type_cat)))
    mae_cv_ind = np.zeros_like(mae_c_ind) ; rmse_cv_ind = np.zeros_like(rmse_c_ind)
    mae_val_ind = np.zeros_like(mae_c_ind) ; rmse_val_ind = np.zeros_like(rmse_c_ind)

    for i in range(len(type_cat)):
        mask = type_calib == type_cat[i]
        mae_c_ind[:,i] = mean_absolute_error(y[mask,:],y_c[mask,:], multioutput='raw_values') ; rmse_c_ind[:,i],_ = mse(y[mask,:],y_c[mask,:], mode='regular')
        mae_cv_ind[:,i] = mean_absolute_error(y[mask,:],y_cv[mask,:], multioutput='raw_values') ; rmse_cv_ind[:,i],_ = mse(y[mask,:],y_cv[mask,:], mode='regular')
        mask2 = type_valid == type_cat[i]
        mae_val_ind[:,i] = mean_absolute_error(y_test[mask2,:],y_val[mask2,:], multioutput='raw_values') ; rmse_val_ind[:,i],_ = mse(y_test[mask2,:],y_val[mask2,:], mode='regular')

    # Data
    x = np.arange(len(type_cat))  # Values on the x-axis
    if mae_plot:
        overalldata = [mae_c_ind,mae_cv_ind,mae_val_ind]
    else:
        overalldata = [rmse_c_ind,rmse_cv_ind,rmse_val_ind]

    bar_labels = ['Calibration','Cross-Validation','Testing']
    # Plotting
    width = 0.2  # Width of each bar
    widths = width*np.arange(-1,2)
    fig, ax = plt.subplots(nrows=3,sharex=True)
    for i in range(3):
        for j in range(len(labels)):
            ax[j].bar(x + widths[i], overalldata[i][j,:], width, label = bar_labels[j])
            ax[j].set_xticks(x)
            ax[j].set_xticklabels(type_cat)
            ax[j].legend([],[],title=labels[j], title_fontproperties = {'size': 11.5, 'weight':'bold'})
            if mae_plot:
                ax[j].set_ylabel('MAE (%)')
            else:
                ax[j].set_ylabel('RMSE (%)')
            ax[j].xaxis.set_minor_locator(AutoMinorLocator(n=1))
            ax[j].yaxis.set_minor_locator(AutoMinorLocator(n=1))
            ax[j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fig.legend(bar_labels, loc='upper center', bbox_to_anchor=(0.5, 1.035), ncol = 3)
    if save:
        plt.savefig(savename+'.png', pad_inches=0,bbox_inches='tight')
    if display_fig:
        plt.show()
    else:
        plt.close()

def normalised_error_plots(cv_dataset, val_dataset, preprocess_labels, wavelength_labels, datasets_highlight, wavelengths_highlight, save = False, save_name = ''):
    """
    Plots normalised error scatter plots for cross-validation and validation datasets.
    Parameters:
    cv_dataset (numpy.ndarray): Cross-validation dataset, expected to be a 2D array where rows correspond to different wavelengths.
    val_dataset (numpy.ndarray): Validation dataset, expected to be a 2D array where rows correspond to different wavelengths.
    preprocess_labels (list): List of preprocessing labels corresponding to the columns of the datasets.
    wavelength_labels (numpy.ndarray): Array of wavelength labels corresponding to the rows of the datasets.
    datasets_highlight (list): List of preprocessing labels to highlight in the plot.
    wavelengths_highlight (list): List of wavelengths to highlight in the plot.
    save (bool, optional): If True, the plot will be saved as an png file. Default is False.
    save_name (str, optional): The name of the file to save the plot. Default is an empty string.
    Returns:
    None
    """
    fig=plt.figure(figsize=(5.7,4)); axes=fig.subplots(1,1)
    col = ['#0000a7', '#4d85c5', '#e0a24c', '#d3592a', '#861b1d', '#008176']
    highlight_col = ['greenyellow','red', 'blue', 'orange', 'purple', 'cyan']

    larger_dataset = cv_dataset if np.max(cv_dataset) > np.max(val_dataset) else val_dataset ; larger_dataset = larger_dataset.flatten()
    smaller_dataset = cv_dataset if np.min(cv_dataset) < np.min(val_dataset) else val_dataset ; smaller_dataset = smaller_dataset.flatten()
    upper_lim = [t.interval(0.75, len(larger_dataset) - 1, loc=np.mean(larger_dataset), scale=np.std(larger_dataset, ddof=1) / len(larger_dataset)**0.5)[1]]
    bottom_lim = [0.9*min(smaller_dataset)]

    for i in range(6):
        xplot = cv_dataset[i,:]; yplot = val_dataset[i,:]
        axes.scatter(xplot.flatten(), yplot.flatten(),c=col[i],label = str(int(wavelength_labels[i]))+' nm', s=50, edgecolor= 'k')

    for j in range(len(datasets_highlight)):
        ind = preprocess_labels.index(datasets_highlight[j]) ; wavelength_ind = np.where(wavelength_labels == wavelengths_highlight[j])[0][0]
        axes.scatter(cv_dataset[wavelength_ind,ind],val_dataset[wavelength_ind,ind],edgecolor=highlight_col[j], marker = 'o', s=125, facecolor = 'none', linestyle = 'dashed', linewidth = 2.75)

    axes.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    axes.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    axes.set_xlim(bottom_lim[0],upper_lim[0]) ; axes.set_ylim(bottom_lim[0],upper_lim[0])
    handles, labels = axes.get_legend_handles_labels()
    axes.set(xlabel='Cross-Validation Error (Normalised)', ylabel='Testing Error (Normalised)')

    axes.legend(handles, labels, loc="upper left", title_fontproperties = {'size': 11.5, 'weight':'bold'})
    if save:
        plt.savefig(save_name + '.png', pad_inches=0,bbox_inches='tight')
    plt.show()

def component_error_plots(cv_dataset, val_dataset, preprocess_labels, wavelength_labels, datasets_highlight, wavelengths_highlight, species_labels, xlabel = 'Cross-Validation Error (MAE)', ylabel='Testing Error (MAE)', save=False, save_name=''):
    """
    Plots the cross-validation and testing errors for different components and highlights specific datasets and wavelengths.
    Parameters:
    cv_dataset (numpy.ndarray): Cross-validation dataset with shape (n_samples, n_features).
    val_dataset (numpy.ndarray): Validation dataset with shape (n_samples, n_features).
    preprocess_labels (list): List of preprocessing labels corresponding to the datasets.
    wavelength_labels (list): List of wavelength labels.
    datasets_highlight (list): List of datasets to highlight.
    wavelengths_highlight (list): List of wavelengths to highlight.
    species_labels (list): List of species labels for the plot titles.
    xlabel (str, optional): Label for the x-axis. Default is 'Cross-Validation Error (MAE)'.
    ylabel (str, optional): Label for the y-axis. Default is 'Testing Error (MAE)'.
    save (bool, optional): Whether to save the plot as an png file. Default is False.
    save_name (str, optional): Name of the file to save the plot. Default is an empty string.
    Returns:
    None
    """

    col = ['#0000a7', '#4d85c5', '#e0a24c', '#d3592a', '#861b1d', '#008176']
    highlight_col = ['greenyellow','red', 'blue', 'orange', 'purple', 'cyan']
    fig = plt.figure(figsize=(15, 4.3))
    axes = fig.subplots(1, 3)

    for i in range(3):
        cvplot_all = cv_dataset[i::3, :] ; valplot_all = val_dataset[i::3, :]
        for j in range(6):
            xplot = cvplot_all[j, :]
            yplot = valplot_all[j, :]
            axes[i].scatter(xplot.flatten(), yplot.flatten(), c=col[j], label=str(int(wavelength_labels[j])) + ' nm', s=50, edgecolor='k')

        for k in range(len(datasets_highlight)):
            ind = preprocess_labels.index(datasets_highlight[k])
            wavelength_ind = np.where(wavelength_labels == wavelengths_highlight[k])[0][0]
            axes[i].scatter(cv_dataset[i::3][wavelength_ind, ind], val_dataset[i::3][wavelength_ind, ind], edgecolor=highlight_col[k], marker='o', s=125, facecolor='none', linestyle='dashed', linewidth=2.75)

        axes[i].xaxis.set_minor_locator(AutoMinorLocator(n=2))
        axes[i].yaxis.set_minor_locator(AutoMinorLocator(n=2))
        larger_dataset = cvplot_all if np.max(cvplot_all) > np.max(valplot_all) else valplot_all ; larger_dataset = larger_dataset.flatten()
        smaller_dataset = cvplot_all if np.min(cvplot_all) < np.min(valplot_all) else valplot_all ; smaller_dataset = smaller_dataset.flatten()
        upper_lim = t.interval(0.75, len(larger_dataset) - 1, loc=np.mean(larger_dataset), scale=np.std(larger_dataset, ddof=1) / len(larger_dataset)**0.5)[1]
        bottom_lim = 0.9*min(smaller_dataset)
        axes[i].set_xlim(bottom_lim, upper_lim)
        axes[i].set_ylim(bottom_lim, upper_lim)
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].set(xlabel=xlabel, ylabel=ylabel)
        axes[i].legend([], [], title=species_labels[i], title_fontproperties={'size': 11.5, 'weight': 'bold'})

    fig.tight_layout(pad=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc='upper center', fontsize=12.5, borderaxespad=0.1, borderpad=-0.15)
    if save:
        plt.savefig(save_name + '.png', pad_inches=0, bbox_inches='tight')
    plt.show()