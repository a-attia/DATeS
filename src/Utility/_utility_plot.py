
#
# ########################################################################################## #
#                                                                                            #
#   DATeS: Data Assimilation Testing Suite.                                                  #
#                                                                                            #
#   Copyright (C) 2016  A. Sandu, A. Attia, P. Tranquilli, S.R. Glandon,                     #
#   M. Narayanamurthi, A. Sarshar, Computational Science Laboratory (CSL), Virginia Tech.    #
#                                                                                            #
#   Website: http://csl.cs.vt.edu/                                                           #
#   Phone: 540-231-6186                                                                      #
#                                                                                            #
#   This program is subject to the terms of the Virginia Tech Non-Commercial/Commercial      #
#   License. Using the software constitutes an implicit agreement with the terms of the      #
#   license. You should have received a copy of the Virginia Tech Non-Commercial License     #
#   with this program; if not, please contact the computational Science Laboratory to        #
#   obtain it.                                                                               #
#                                                                                            #
# ########################################################################################## #
#   S set of functions to manipulate matplotlib plots/figures                                #
# ########################################################################################## #
#


"""
    A module providing classes and functions that handle url-related functionalities; such as downloading files, etc.
"""

try:
    from matplotlib import figure as _figure
except ImportError:
    print("Failed to Import from matplotlib. Plotting won't work in this session... Proceeding ...")



def make_ticklabels_invisible(fig):
    """
    Hide all tickable elements on the passed matplotlib figure

    Args:
        fig: a matplotlib figure handle

    Returns:
        None

    """
    assert isinstance(fig, _figure.Figure), "This function is valid only for matplotlib figures!"
    for i, ax in enumerate(fig.axes):
        ax.set_adjustable('box-forced')
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

def make_titles_invisible(fig):
    """
    Hide all titles/subtitles on the passed matplotlib figure

    Args:
        fig: a matplotlib figure handle

    Returns:
        None

    """
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)
