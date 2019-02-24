
"""
This is a simple script containing very simple functions to enhance the shape of matplotlib plots.
This is intended to be used to modify current workspace
"""

from matplotlib import rc


def enhance_plots(font_size=18, font_family='normal', font_weight='bold', use_tex=True):
    """
    Initial setup of all plotting settings; font size, width, family, latex etc.
    This acts on matplotlib.rc
    """
    # set fonts, and colors:
    font = {'family':font_family,
            'weight':font_weight,
            'size': font_size}
    rc('font', **font)
    rc('text', usetex=use_tex)
    #

def show_grids(ax, showmajor=True,
               majorcolor='gray',
               majorlinestyle='-',
               majorlinewidth=0.5,
               showminor=True,
               minorcolor='silver',
               minorlinestyle=':',
               minorlinewidth=0.5
               ):
    """ given an axis ax, show the major and minor grids, based on passed settings
    """
    if not (showmajor or showminor):
        pass
    else:
        #
        # Customize the major grid
        if showmajor:
            ax.grid(which='major', linestyle=majorlinestyle, linewidth=majorlinewidth, color=majorcolor)
        #
        # Customize the minor grid
        if showminor:
            # Turn on the minor TICKS, which are required for the minor GRID
            ax.minorticks_on()
            ax.grid(which='minor', linestyle=minorlinestyle, linewidth=minorlinewidth, color=minorcolor)
            #
    return ax


