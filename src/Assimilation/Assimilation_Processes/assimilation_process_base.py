
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
#



class AssimilationProcess(object):
    """
    assimilation_process_base.AssimilationProcess:
    A base class for an assimilation process.
    An assimilation process here refers to repeating an assimilation cycle
    (filtering/smoothing/hybrid) over a specific observation/assimilation timespan

    A base class implementing common features in an assimilation process.
    Classes should inherit this one include: FilteringProcess, SmoothingProcess, HybridProcess, etc.
    
    Args:
        assimilation_configs: dict,
        A dictionary containing assimilation configurations.
            
        output_configs (default None); a dictionary containing screen/file output configurations:
            
    """

    def __init__(self, *argv, **kwargs):
        
        raise NotImplementedError

    def __repr__(self):
        """
        return a nice string description of the AssimilationProcess class
        """
        raise NotImplementedError

    def __str__(self):
        """
        return a nice string description of the AssimilationProcess class
        """
        return __repr__(self)

    def assimilation_cycle(self, *argv, **kwargs):
        """
        carry out assimilation for the next assimilation cycle. This should be either a filtering step, 
        a smoothing step or a hybrid step.
        This method should be called by self.recursive_assimilation_process() to carry out the next 
        assimilation cycle.
        
        Args:
        
        Returns:
        
        """
        raise NotImplementedError

    def recursive_assimilation_process(self, *argv, **kwargs):
        """
        Loop over all assimilation cycles and return/save results (forecast, analysis, observations, etc.) 
        for all the assimilation cycles based on the passed configurations.
        The printing/saving should be provided by the assimilation class. 
        All this process should do is to manipulate the settings of the assimilation object for ouputting.
        assimilation cycle.
        
        Args:
        
        Returns:
        
        """        
        raise NotImplementedError
        
        
