try:
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    from matplotlib import rcParams, cycler
    import matplotlib.font_manager as font_manager
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])


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

## wavlengths for absorption bands of biomass components

def abs_bands(plt,choice):
    '''function to plot the biomass absorption bands based on choice of bands to be plotted
    in order, give as True/False: [lignin, cellulose, water, polysaccharides]'''
    lig_min=np.array([1685, 1440, 1420, 1170])-1
    lig_max=lig_min+2
    lig=np.array([1672, 1446.5, 1220])
    #lig=np.stack((lig,lig))

    cell_min=np.array([1427, 1480, 1548, 1592, 1780])-1
    cell_max=cell_min+2
    cell=np.array([1480, 1548, 1592, 1780])
    #cell=np.stack((cell,cell))
    #plot_lig=np.array([0,0.3])

    water_min=np.array([1780, 1920])-1
    water_max=water_min+2

    sach_min=np.array([1724, 1430, 1669, 1724, 1920, 2090, 2270, 2329, 2488])-1
    sach_max=sach_min+2
    sach=np.array([1724,1930, 2090, 2270, 2329, 2488])
    #sach=np.stack((sach,sach))
    plot_sach=np.array([0,0.5])

    if choice[0]:
        #for i in range(np.shape(lig_min)[0]):
        #    plt.axvspan(lig_min[i],lig_max[i],color='yellow', label='lignin')
        for i in lig:
            plt.plot([i,i],[0,0.3],c='yellow', label='lignin')
    
    if choice[1]:
        #for i in range(np.shape(cell_min)[0]):
        #    plt.axvspan(cell_min[i],cell_max[i],color='m', label='cellulose')
        for i in cell:
            plt.plot([i,i],[0,0.3],'--',c='navy', label='cellulose')
    
    if choice[2]:
        for i in range(np.shape(water_min)[0]):
            plt.axvspan(water_min[i],water_max[i],color='r', label='water')
    
    if choice[3]:
        #for i in range(np.shape(sach_min)[0]):
        #    plt.axvspan(sach_min[i],sach_max[i],color='c', label='polysaccharides')
        for i in sach:
            plt.plot([i,i],[0,0.5],':m', label='polysaccharides')


def abs_bands_IL(plt):
    IL_min=np.array([1190, 1270, 1360, 1380, 1400, 1430, 1470, 1700, 1750, 2250, 2320,2400])-1
    IL_max=IL_min+2
    for i in range(np.shape(IL_min)[0]):
        plt.axvspan(IL_min[i],IL_max[i],color='c', label='TEA')

#function to ensure consistent legends
def legend(plt, font_legend,loc='best'):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, prop=font_legend,loc=loc)