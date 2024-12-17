try:
    import numpy as np
    from scipy.optimize import fsolve
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import numpy as np
    from scipy.optimize import fsolve

from NIR_Plots import *
import pickle as pkl
import PLS
from PLS import *
import re

def uncertainty_estimation(pls, X_test, n_points = 100):
    """
    Estimate prediction uncertainties using a PLS model.
    This function computes the predicted values, uncertainty intervals, and density estimates 
    for each test sample using a Partial Least Squares (PLS) model.
    Parameters
    ----------
    pls : object
        A trained PLS model with `predict`, `prediction_cdf`, and `boot_pdf` methods.
    X_test : numpy.ndarray
        Test dataset of shape (n_samples, n_features).
    n_points : int, optional
        Number of points to compute the uncertainty interval (default is 100).
    Returns
    -------
    tuple
        A tuple containing:
            y_hat (numpy.ndarray): Predicted values of shape (n_samples, 3).
            y_pos (numpy.ndarray): Positions spanning the uncertainty interval, shape (n_samples, n_points, 3).
            y_dens (numpy.ndarray): Density estimates at each position, shape (n_samples, n_points, 3).
    """
    n_test = X_test.shape[0]
    y_hat = np.zeros((n_test,3))
    y_pos = np.zeros((n_test, n_points, 3))
    y_dens = np.zeros((n_test,n_points,3))

    for i in range(n_test):
        print(i)
        y_hat[i] = pls.predict(X_test[i])
        f = lambda y: pls.prediction_cdf(X_test[i], y) - 0.025
        g = lambda y: pls.prediction_cdf(X_test[i], y) - 0.975
        y_min = fsolve(f, y_hat[i])
        y_max = fsolve(g, y_hat[i])

        y_pos[i] = np.linspace(y_min, y_max, n_points).reshape(n_points,3)
        for j in range(n_points):
            y_dens[i,j,:] = pls.boot_pdf(X_test[i],y_pos[i,j,:])

    return (y_hat, y_pos, y_dens)

def violin_plot(y_pos_plot, y_dens_plot, y_hat_plot, y_test_plot, plot_labels, uncertainty_up, uncertainty_down, y_comp_err, sort_uncertainty = True, scale = 1, label_rotation = 50, time_plt = [],
                            ha_plt = 'right', common_string = None, multiline_label = None, mass_closure = True, legend_loc = 'best', save_fig = False, save_name = '', subplot_labels = []):
    """
    Creates a violin plot with error bars, experimental data points, and mass closure indicators.
    Parameters
    ----------
    y_pos_plot : ndarray
        Array containing position values for violin plot construction.
    y_dens_plot : ndarray
        Array containing density values for violin plot construction.
    y_hat_plot : ndarray
        Array containing predicted values.
    y_test_plot : ndarray
        Array containing experimental/test data points.
    plot_labels : array-like
        Labels for x-axis.
    uncertainty_up : ndarray
        Upper uncertainty bounds.
    uncertainty_down : ndarray
        Lower uncertainty bounds.
    y_comp_err : ndarray
        Experimental error bars.
    sort_uncertainty : bool, optional
        If True, sorts plots by maximum uncertainty (default True).
    scale : float, optional
        Scaling factor for density plots (default 1).
    label_rotation : int, optional
        Rotation angle for x-axis labels (default 50).
    time_plt : array-like, optional
        Time points for x-axis (default empty list).
    ha_plt : str, optional
        Horizontal alignment for x-axis labels (default 'right').
    common_string : str or list, optional
        String(s) to remove from plot labels (default None).
    multiline_label : str, optional
        String to add line break after in labels (default None).
    mass_closure : bool, optional
        If True, shows mass closure indicators (default True).
    legend_loc : str, optional
        Location of the legend (default 'best').
    save_fig : bool, optional
        If True, saves the figure (default False).
    save_name : str, optional
        Prefix for saved figure filename (default '').
    subplot_labels : list, optional
        Labels for subplots (default empty list).
    Returns
    -------
    None
        Displays the plot and optionally saves it as a PNG file.
    Notes
    -----
    The plot shows three components (Glucan, Hemicellulose, Lignin) with violin plots
    representing distribution, diamond markers for predictions, circles for experimental
    data, and error bars for uncertainties. Mass closure is indicated with red markers
    if enabled.
    """
    mass_closure_plot = y_hat_plot.sum(axis=1)
    plot_titles = ["Glucan (Cellulose)","Hemicellulose","Lignin"] ; ind_all = [0,1,2]
    colour_all = ["darkgreen","#d9d900","navy","black","black","red"] ; rcParams['markers.fillstyle'] = 'full'
    err_colour_all = ["lightgreen","darkolivegreen","deepskyblue","black","black","red"]
    weight_all = ['bold','bold','bold','normal','normal','normal']
    
    n_test_plot = y_pos_plot.shape[0]
    if sort_uncertainty:
        l = np.argsort(np.max(y_dens_plot,axis=1)[:,0])[::-1] ; x_labels = plot_labels[l]
    else:
        l = list(range(0,n_test_plot)) ; x_labels = plot_labels[l]

    if common_string!=None:
        x_labels = [w.replace(' - 2', '') for w in x_labels]
        if isinstance(common_string,list):
            for i in range(len(common_string)):
                x_labels = [w.replace(common_string[i], '') for w in x_labels]
        else:
            x_labels = [w.replace(common_string, '') for w in x_labels]
    
    if multiline_label!=None:
        x_labels = [re.sub(multiline_label, multiline_label+'\n', w) for w in x_labels]

    if len(time_plt) == 0:
        time_plt = np.arange(n_test_plot) ; xlabel = "" ; l_label = list(range(n_test_plot+1))
    else:
        x_labels = [str(i) for i in time_plt] ; xlabel = 'Time (h)' ; l_label = np.insert(time_plt+1,0,0)
    
    fig,ax1 = plt.subplots(figsize = (6.5+0.5*n_test_plot,5))
    for ind in ind_all:
        for i in range(n_test_plot):
            rgb_col = colors.to_rgba(colour_all[ind]) ; darkened_rgb_color = tuple(comp * 0.7 for comp in rgb_col) ; dark_col = colors.to_hex(darkened_rgb_color)
            lightened_rgb_color = tuple(comp * 0.5 for comp in rgb_col) ; light_col = colors.to_hex(lightened_rgb_color)
            ax1.fill_betweenx(y_pos_plot[l[i],:,ind], y_dens_plot[l[i],:,ind]*scale + time_plt[i]  + 1 , -y_dens_plot[l[i],:,ind]*scale+ time_plt[i] + 1, alpha = 0.5, color = colour_all[ind], edgecolor = dark_col)
            if np.any(y_test_plot[l[i]]):
                ax1.scatter(time_plt[i]+1, y_test_plot[l[i],ind], marker='o', edgecolors='black',color = colour_all[ind], s= 50)
                ax1.errorbar(time_plt[i]+1, y_test_plot[l[i],ind], yerr=y_comp_err[l[i],ind], fmt='none', ecolor = err_colour_all[ind], elinewidth = 1.5, capsize = 5, linestyle = '--')
        ax1.scatter(time_plt+1, y_hat_plot[l,ind], marker='D', edgecolor = dark_col, facecolor = 'none', s=50, linewidth = 2.5)
        ax1.errorbar(time_plt+1, y_hat_plot[l,ind], yerr=[uncertainty_down[l,ind], uncertainty_up[l,ind]], fmt='none', ecolor = dark_col, elinewidth = 1.5, capsize = 5)

    for j in plot_titles:
        plt.plot([], [], ' ', label=j)

    plt.scatter([], [], marker='D', edgecolor='black', facecolor = 'lightgrey', s=20, label="Nominal Predictions", linewidth = 2.5)
    if np.any(y_test_plot):
        plt.scatter([], [], marker='o', color='black', s=30, label="Experimental Data")

    if mass_closure:
        for i in range(n_test_plot):
            plt.scatter(time_plt[i]+1,mass_closure_plot[i], marker = '_', color='red')
            ax1.text(time_plt[i]+0.93, mass_closure_plot[i]+3, str(int(mass_closure_plot[i])), color = 'red')
        plt.scatter([], [], marker='_', color='red', s=30, label="Mass Closure")

    ax1.set_ylabel("Content (%)")
    x_labels = np.insert(x_labels,0," ") ; 
    plt.xticks(l_label, x_labels, rotation = label_rotation, ha = ha_plt)
    plt.xlabel(xlabel)
    plt.axhline(y=0, color = "black")
    ax1.autoscale(axis="x")
    if n_test_plot<=6:
        leg = plt.legend(framealpha = 0, loc = legend_loc, ncol=3, bbox_to_anchor=(0.87,1.15), columnspacing=0.8)
        if len(subplot_labels)>0:
            ax1.text(0.12, 0.96, subplot_labels[0], horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    else:
        leg = plt.legend(framealpha = 0, loc = legend_loc, ncol=6, bbox_to_anchor=(0.92,1.075), columnspacing=0.8)
        if len(subplot_labels)>0:
            ax1.text(0.12, 0.975, subplot_labels[0], horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    
    for text,col, weight in zip(leg.get_texts(),colour_all, weight_all):
        text.set_fontweight(weight)
        if text.get_text() == 'Mass Closure':
            plt.setp(text, color = 'red')
        else:
            plt.setp(text, color = col) 
    ax1.xaxis.set_minor_locator(AutoMinorLocator(n=1))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_name + "Uncertainty.png", pad_inches = 0, bbox_inches = "tight")
    plt.show()

def spectra_minmax(spectra_raw,num_repeats=4):
    """
    Calculate minimum and maximum values for repeated spectra measurements.

    This function takes a raw spectra matrix where each sample has multiple repeat measurements
    and calculates the minimum and maximum values across the repeats for each sample.

    Parameters
    ----------
    spectra_raw : numpy.ndarray
        Raw spectra matrix where rows are wavelengths and columns are measurements.
        The measurements should be organized as consecutive repeat measurements for each sample.
    num_repeats : int, optional
        Number of repeat measurements per sample (default is 4)

    Returns
    -------
    spectra_raw_min : numpy.ndarray
        Matrix containing minimum values across repeats for each sample.
        Shape is (n_wavelengths, n_samples)
    spectra_raw_max : numpy.ndarray
        Matrix containing maximum values across repeats for each sample.
        Shape is (n_wavelengths, n_samples)

    Notes
    -----
    The input spectra_raw should be organized such that the repeat measurements
    for each sample are in consecutive columns. For example, with num_repeats=4:
        - Columns 0-3: repeats for sample 1
        - Columns 4-7: repeats for sample 2
        And so on.
    """
    num_samples=np.int64(spectra_raw.shape[1]/num_repeats)
    spectra_raw_min=np.zeros((spectra_raw.shape[0],num_samples))
    spectra_raw_max=np.zeros((spectra_raw.shape[0],num_samples))
    for i in range(num_samples):
        spectra_raw_min[:,i]=np.min(spectra_raw[:,num_repeats*i:num_repeats*(i+1)],axis=1)
        spectra_raw_max[:,i]=np.max(spectra_raw[:,num_repeats*i:num_repeats*(i+1)],axis=1)
    return spectra_raw_min, spectra_raw_max

def uncertainty_estimation_spectra_measurement(pls, spectra, spectra_min, spectra_max):
    """
    Estimates the uncertainty in spectral measurements using a PLS model.
    Parameters:
        pls (object): A trained Partial Least Squares (PLS) regression model with a `predict` method.
        spectra (ndarray): An array of spectra measurements with shape (n_samples, n_features).
        spectra_min (ndarray): An array of minimum spectra measurements with shape (n_samples, n_features).
        spectra_max (ndarray): An array of maximum spectra measurements with shape (n_samples, n_features).
    Returns:
        tuple:
            y_hat (ndarray): Predicted values from the PLS model for the input spectra.
            spectral_uncertainty_up (ndarray): Upper uncertainty estimates for the predicted values.
            spectral_uncertainty_down (ndarray): Lower uncertainty estimates for the predicted values.
    """
    y_hat = np.zeros((spectra.shape[0],3)); y_hat_min = np.zeros((spectra_min.shape[0],3)) ; y_hat_max = np.zeros((spectra_max.shape[0],3))
    for i in range(spectra_min.shape[0]):
        y_hat[i] = pls.predict(spectra[i,:])
        y_hat_min[i] = pls.predict(spectra_min[i,:])
        y_hat_max[i] = pls.predict(spectra_max[i,:])
    
    spectral_uncertainty_up, spectral_uncertainty_down = uncertainty_estimation_spectra_calculation(y_hat, y_hat_min, y_hat_max)
    return y_hat, spectral_uncertainty_up, spectral_uncertainty_down


def uncertainty_estimation_spectra_calculation(y, y_min, y_max):
    """
    This function estimates the uncertainty of the model prediction for each sample in the validation set.
    The uncertainty is calculated as the sum of the absolute difference between the prediction of the model trained on the minimum and maximum spectra and the prediction of the model trained on the raw spectra.
    The uncertainty is calculated for each sample
    """
    opposite_sign = np.sign(y_min - y) == -np.sign(y_max - y)
    uncertainty_up = np.zeros_like(y) ; uncertainty_down = np.zeros_like(y)
    for i in range(len(opposite_sign)):
        for j in range(opposite_sign.shape[1]):
            if opposite_sign[i,j]:
                if y_min[i,j] > y_max[i,j]:
                    uncertainty_up[i,j] = np.abs(y_min[i,j] - y[i,j])
                    uncertainty_down[i,j] = np.abs(y_max[i,j] - y[i,j])
                else:
                    uncertainty_up[i,j] = np.abs(y_max[i,j] - y[i,j])
                    uncertainty_down[i,j] = np.abs(y_min[i,j] - y[i,j])
            else:
                if y_min[i,j] > y[i,j]:
                    uncertainty_up[i,j] = max(np.abs(y_min[i,j] - y[i,j]), np.abs(y_max[i,j] - y[i,j]))
                    uncertainty_down[i,j] = 1E-5
                else:
                    uncertainty_down[i,j] = max(np.abs(y_min[i,j] - y[i,j]), np.abs(y_max[i,j] - y[i,j]))
                    uncertainty_up[i,j] = 1E-5
    
    return uncertainty_up, uncertainty_down

def plot_parity_with_uncertainty(y, y_c, uncertainty_down, uncertainty_up, labels, legend_title, save=False, savename=''):
    """
    Plots a parity plot with uncertainty bounds for multiple components.
    This function creates a scatter plot comparing measured vs predicted values for multiple
    components, including error bars for uncertainty and trend lines. It also displays R² values
    for each component and a parity line for reference.
    Parameters
    ----------
    y : numpy.ndarray
        2D array of measured values where each column represents a different component
    y_c : numpy.ndarray
        2D array of predicted values corresponding to y
    uncertainty_down : numpy.ndarray
        2D array of lower uncertainty bounds for each prediction
    uncertainty_up : numpy.ndarray
        2D array of upper uncertainty bounds for each prediction
    labels : list
        List of strings containing names for each component
    legend_title : str
        Title for the plot legend
    save : bool, optional
        If True, saves the plot to a file. Default is False
    savename : str, optional
        Path and filename for saving the plot. Only used if save=True. Default is ''
    Returns
    -------
    None
        Displays the plot and optionally saves it to a file
    Notes
    -----
    - The function uses a predefined color cycle: green, yellow, and navy
    - Error bars are displayed as vertical lines
    - A trend line is fitted for each component
    - The parity line (y=x) is shown as a black dashed line
    - Plot limits are automatically set based on data range
    """

    r2_comp = np.array([r2_score(y[:, i], y_c[:, i]) for i in range(y.shape[1])])
    scale = 10
    fig, ax = plt.subplots(figsize=(6, 5))
    cycler = ['green', '#d9d900', 'navy']

    for i in range(len(labels)):
        col = cycler[i]
        rgb_col = colors.to_rgba(col)
        darkened_rgb_color = tuple(comp * 0.8 for comp in rgb_col)
        dark_col = colors.to_hex(darkened_rgb_color)
        ax.scatter(y[:, i], y_c[:, i], s=50, edgecolors=col, facecolors='none', label=f'{labels[i]} - $R^{{2}}_{{train}}$= {round(r2_comp[i], 3)}')
        ax.errorbar(y[:, i], y_c[:, i], xerr=0, yerr=[uncertainty_down[:, i], uncertainty_up[:, i]], fmt='none', ecolor=col, elinewidth=1, capsize=2)
        z = np.polyfit(y[:, i], y_c[:, i], 1)
        ax.plot(y[:, i], np.polyval(z, y[:, i]), linewidth=1, color=col)

    ax.plot(np.linspace(np.min(y), np.max(y), 10), np.linspace(np.min(y), np.max(y), 10), color='black', linestyle='dashed', linewidth=2, label='Parity line')
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))

    plt.xlabel('Measured (%)')
    plt.ylabel('Predicted (%)')
    ax.legend(loc='upper left', title=legend_title, title_fontproperties={'size': 12, 'weight': 'bold'})
    ax.set_xlim([(min(np.min(y), np.min(y_c)) - 0.5), (max(np.max(y), np.max(y_c)) + 0.5)])
    ax.set_ylim([(min(np.min(y), np.min(y_c)) - 0.5), (max(np.max(y), np.max(y_c)) + 0.5)])
    if save:
        plt.savefig(savename, pad_inches=0, bbox_inches='tight')
    plt.show()

def violin_plot_compare(y_pos_plot_all, y_dens_plot_all, y_hat_plot_all, y_test_plot, plot_labels, uncertainty_up_all, uncertainty_down_all, y_comp_err, sort_uncertainty = True, scale = 1, label_rotation = 50, time_plt = [],
                            ha_plt = 'right', common_string = None, multiline_label = None, mass_closure = True, legend_loc = 'best', save_fig = False, save_name = '', subplot_labels = []):
    """
    Generates a violin plot to compare predicted and experimental data with uncertainties.
    Parameters:
    -----------
    y_pos_plot_all : list of np.ndarray
        List of arrays containing y positions for the violin plots.
    y_dens_plot_all : list of np.ndarray
        List of arrays containing y densities for the violin plots.
    y_hat_plot_all : list of np.ndarray
        List of arrays containing predicted values.
    y_test_plot : np.ndarray
        Array containing experimental data.
    plot_labels : list of str
        List of labels for the x-axis.
    uncertainty_up_all : list of np.ndarray
        List of arrays containing upper uncertainty bounds.
    uncertainty_down_all : list of np.ndarray
        List of arrays containing lower uncertainty bounds.
    y_comp_err : np.ndarray
        Array containing comparison errors.
    sort_uncertainty : bool, optional
        If True, sorts the data by maximum density. Default is True.
    scale : float, optional
        Scale factor for the density plots. Default is 1.
    label_rotation : int, optional
        Rotation angle for x-axis labels. Default is 50.
    time_plt : list, optional
        List of time points for the x-axis. Default is an empty list.
    ha_plt : str, optional
        Horizontal alignment for x-axis labels. Default is 'right'.
    common_string : str or list of str, optional
        Common string(s) to be removed from plot labels. Default is None.
    multiline_label : str, optional
        String to be used for multiline labels. Default is None.
    mass_closure : bool, optional
        If True, includes mass closure in the plot. Default is True.
    legend_loc : str, optional
        Location of the legend. Default is 'best'.
    save_fig : bool, optional
        If True, saves the figure. Default is False.
    save_name : str, optional
        Name of the file to save the figure. Default is an empty string.
    subplot_labels : list of str, optional
        List of labels for subplots. Default is an empty list.
    Returns:
    --------
    None
    """
    plot_titles = ["Glucan (Cellulose)","Hemicellulose","Lignin"] ; ind_all = [0,1,2]
    colour_all = ["darkgreen","#d9d900","navy","navy","navy","black"] ; rcParams['markers.fillstyle'] = 'full'
    err_colour_all = ["lightgreen","darkolivegreen","deepskyblue","black","black","red"]
    weight_all = ['bold','bold','bold','normal','normal','normal']
    
    hatch_compare = [None, 'xxxx'] ; position_adjust = [-0.1, 0.1] ; marker_compare = ['D','s']

    n_test_plot = y_pos_plot_all[0].shape[0]
    if sort_uncertainty:
        l = np.argsort(np.max(y_dens_plot,axis=1)[:,0])[::-1] ; x_labels = plot_labels[l]
    else:
        l = list(range(0,n_test_plot)) ; x_labels = plot_labels[l]

    if common_string!=None:
        x_labels = [w.replace(' - 2', '') for w in x_labels]
        if isinstance(common_string,list):
            for i in range(len(common_string)):
                x_labels = [w.replace(common_string[i], '') for w in x_labels]
        else:
            x_labels = [w.replace(common_string, '') for w in x_labels]
    
    if multiline_label!=None:
        x_labels = [re.sub(multiline_label, multiline_label+'\n', w) for w in x_labels]

    if len(time_plt) == 0:
        time_plt = np.arange(n_test_plot) ; xlabel = "" ; l_label = list(range(n_test_plot+1))
    else:
        x_labels = [str(i) for i in time_plt] ; xlabel = 'Time (h)' ; l_label = np.insert(time_plt+1,0,0)
    
    fig,ax1 = plt.subplots(figsize = (6.5+0.5*n_test_plot,5))

    for k in range(2):
        
        y_pos_plot = y_pos_plot_all[k] ; y_dens_plot = y_dens_plot_all[k] ; y_hat_plot = y_hat_plot_all[k] ;uncertainty_up = uncertainty_up_all[k] ; uncertainty_down = uncertainty_down_all[k]
        mass_closure_plot = y_hat_plot.sum(axis=1)
        for ind in ind_all:
            colour_fill = [colour_all[ind], 'none']
            for i in range(n_test_plot):
                rgb_col = colors.to_rgba(colour_all[ind]) ; darkened_rgb_color = tuple(comp * 0.7 for comp in rgb_col) ; dark_col = colors.to_hex(darkened_rgb_color)
                lightened_rgb_color = tuple(comp * 0.5 for comp in rgb_col) ; light_col = colors.to_hex(lightened_rgb_color)
                ax1.fill_betweenx(y_pos_plot[l[i],:,ind], y_dens_plot[l[i],:,ind]*scale + time_plt[i]  + 1 + position_adjust[k] , -y_dens_plot[l[i],:,ind]*scale+ time_plt[i] + 1 + position_adjust[k], alpha = 0.5, facecolor = colour_fill[k], edgecolor = dark_col, hatch = hatch_compare[k])
                if np.any(y_test_plot[l[i]]):
                    ax1.scatter(time_plt[i]+1, y_test_plot[l[i],ind], marker='o', edgecolors='black',color = colour_all[ind], s= 50)
                    ax1.errorbar(time_plt[i]+1, y_test_plot[l[i],ind], yerr=y_comp_err[l[i],ind], fmt='none', ecolor = err_colour_all[ind], elinewidth = 1.5, capsize = 5, linestyle = '--')
            ax1.scatter(time_plt+1 + position_adjust[k], y_hat_plot[l,ind], marker=marker_compare[k], edgecolor = dark_col, facecolor = 'none', s=50, linewidth = 2.5, hatch = hatch_compare[k])
            ax1.errorbar(time_plt+1 + position_adjust[k], y_hat_plot[l,ind], yerr=[uncertainty_down[l,ind], uncertainty_up[l,ind]], fmt='none', ecolor = dark_col, elinewidth = 1.5, capsize = 5)

    for j in plot_titles:
        plt.plot([], [], ' ', label=j)

    if np.any(y_test_plot):
        plt.scatter([], [], marker='o', color='black', s=30, label="Experimental Data")

    if mass_closure:
        for i in range(n_test_plot):
            plt.scatter(time_plt[i]+1,mass_closure_plot[i], marker = '_', color='red')
            ax1.text(time_plt[i]+0.93, mass_closure_plot[i]+3, str(int(mass_closure_plot[i])), color = 'red')
        plt.scatter([], [], marker='_', color='red', s=30, label="Mass Closure")

    plt.scatter([], [], marker='D', color='black', s=20, label="Predictions A", linewidth = 2.5)
    patch_1 = mpatches.Patch(edgecolor='black', facecolor = 'lightgrey', label='Uncertainty A')
    patch_2 = mpatches.Patch(edgecolor='black', facecolor = 'none', label='Uncertainty B', hatch = 'xxxx')

    handles, labels = ax1.get_legend_handles_labels()
    lab_extra1 = plt.scatter([], [], marker='s', color='black', s=30, label="Predictions B")

    handles.append(patch_1) ; handles.append(lab_extra1); handles.append(patch_2)
    labels.append('Uncertainty A') ; labels.append("Predictions B"); labels.append('Uncertainty B')

    ax1.set_ylabel("Content (%)")
    x_labels = np.insert(x_labels,0," ") ; 
    plt.xticks(l_label, x_labels, rotation = label_rotation, ha = ha_plt)
    plt.xlabel(xlabel)
    plt.axhline(y=0, color = "black")
    ax1.autoscale(axis="x")
    leg = plt.legend(handles, labels, framealpha = 0, loc = legend_loc, ncol=4, bbox_to_anchor=(0.87,1.15), columnspacing=0.8)
    if len(subplot_labels)>0:
        ax1.text(0.15, 0.96, subplot_labels[0], horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')

    for text,col, weight in zip(leg.get_texts(),colour_all, weight_all):
        text.set_fontweight(weight)
        if text.get_text() == 'Mass Closure':
            plt.setp(text, color = 'red')
        else:
            plt.setp(text, color = col) 
    ax1.xaxis.set_minor_locator(AutoMinorLocator(n=1))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_name + "Uncertainty.png", pad_inches = 0, bbox_inches = "tight")
    plt.show()