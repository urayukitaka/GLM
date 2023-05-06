'''
utils

各種共通関数
'''
################################
# Libraries
################################
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt

################################
# Functions
################################

# Prepare directory
def prepare_savedir(ROOT:str, name:str)->str:
    '''
    Args
        ROOT (str) : Directory for save
        name (str) : Save directory name
    '''

    # define directory
    savedir = os.path.join(ROOT, name)
    # make direcotry
    os.makedirs(savedir, exist_ok=True)

    return savedir

def getdate():
    '''
    Args
        None
    '''

    return dt.now().strftime("%Y%m%d_%H%M")


# matplot graph setting
def graph_setting():
    '''
    Args
        None
    '''
    # setting
    plt.rcParams["figure.dpi"] = 100 # dpi(dots per inch, resolution )

    # font setting
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    # xticls setting
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["xtick.color"] = "black"
    # minor xticks setting
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["xtick.minor.size"] = 2.0
    plt.rcParams["xtick.minor.width"] = 0.6

    # yticks setting
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["ytick.color"] = "black"
    # minor xticks setting
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.minor.size"] = 2.0
    plt.rcParams["ytick.minor.width"] = 0.6

    # axis
    plt.rcParams["axes.labelsize"] = 12 # axes label font size
    plt.rcParams["axes.linewidth"] = 1.0 # outline width
    plt.rcParams["axes.grid"] = True # grid, true or false
    plt.rcParams["axes.edgecolor"] = "black"

    # grid
    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.linewidth"] = 0.1

    # face color
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # legend
    plt.rcParams["legend.loc"] = "best" # position of legend
    plt.rcParams["legend.frameon"] = True # frame of legend
    plt.rcParams["legend.framealpha"] = 1.0 # transparency of frame
    plt.rcParams["legend.facecolor"] = "white" # back color of legend
    plt.rcParams["legend.edgecolor"] = "black" # edge color
    plt.rcParams["legend.fancybox"] = True # edge type