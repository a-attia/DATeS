
"""
============================================================================================
=                                                                                          =
=   DATeS: Data Assimilation Testing Suite.                                                =
=                                                                                          =
=   Copyright (C) 2016  A. Sandu, A. Attia, P. Tranquilli, S.R. Glandon,                   =
=   M. Narayanamurthi, A. Sarshar, Computational Science Laboratory (CSL), Virginia Tech.  =
=                                                                                          =
=   Website: http://csl.cs.vt.edu/                                                         =
=   Phone: 540-231-6186                                                                    =
=                                                                                          =
=   This program is subject to the terms of the Virginia Tech Non-Commercial/Commercial    =
=   License. Using the software constitutes an implicit agreement with the terms of the    =
=   license. You should have received a copy of the Virginia Tech Non-Commercial License   =
=   with this program; if not, please contact the computational Science Laboratory to      =
=   obtain it.                                                                             =
=                                                                                          =
============================================================================================
********************************************************************************************
*   ....................................................................................   *
*    AssimilationProcess.FilteringProcess:                                                 *
*    A class implementing functionalities of a filtering process.                          *
*    A filtering process here refers to repeating a filtering cycle over a specific        *
*    observation/assimilation timespan                                                     *
*   ....................................................................................   *
*   Note:                                                                                  *
*        The main methods here update the DA items and update the appropriate configs      *
*        dictionaries in the filter object.                                                *
*        The filter object itself is responsible for input/output.                         *
********************************************************************************************
*                                                                                          *
********************************************************************************************
"""

from filtering_process import *
from state_vector_base import StateVectorBase
from observation_vector_base import ObservationVectorBase
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import gaussian_process
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

import dates_utility as utility
#import matplotlib.pyplot as plt
import scipy.special as sp

import sys
import os
import random
import copy
import scipy
import numpy as np
import scipy.stats as stats
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64



class AdaptiveFilteringProcess(FilteringProcess):
    """
    THIS IS A DERIVED CLASS FROM "FilteringProcess";
    A class implementing the steps of a filtering process.
    This recursively apply filtering on several assimilation cycles given assimilation/observation timespan.
    """
        
    def __init__(self, loc_radii_pool, featured_state_indexes, featured_correlation_indexes,
                       test_phase_initial_index=None, use_same_loc_radius=True, num_training_trials=1, assimilation_configs=None, output_configs=None):
        """
        AdaptiveFilteringProcess class constructor. 
        The input configurarions are used to control the assimilation process behavious over several cycles.
        
        Parameters
        ----------
        loc_radii_pool: an iterable (e.g. a list) that contains initial randge of adaptive localization radii
        eatured_state_indexes: the indexes of the state vector that will be invistigated as features (as per Azam's request)
        use_same_loc_radius (default True): use the same localization radius for all states (at all gridpoints)
        num_training_trials: number of times per cycle the machine will be trained; 
                             this is used only if use_same_loc_radius is set to False.
        
        assimilation_configs: dict,
            A dictionary containing assimilation configurations.
            Supported configuarations:
            --------------------------
                * filter (default None): filter object
                * obs_checkpoints (default None): iterable containing an observation timespan 
                   - thsese are the time instances at which observation (synthetic or not) are given/generated
                * da_checkpoints (default None): : iterable containing an assimilation timespan 
                   - thsese are the time instances at which filtering is carried out
                   - If same as obs_checkpoints, assimilation is synchronous. 
                * da_time_spacing (default None): Type of spacing between consecutive assimilation cycles.
                   - Used (along with (num_filtering_cycles) if both obs_checkpoints, da_checkpoints are None.
                * num_filtering_cycles (default None): Used with 'da_time_spacing'only
                * synchronous (default None): Now it has no effect. this will be removed.
                * ref_initial_condition (default None): model.state_vector object used as a true initial state
                * ref_initial_time (default 0):  the time at which 'ref_initial_condition' is defined
                * forecast_first (default True): Forecast then Analysis or Analysis then Forecast steps
                * random_seed (default None): An integer to reset the random seed of Numpy.random
                  - resets the seed of random number generator before carrying out any filtering steps.
                  
        Returns
        ----------
        None
        
        """
        FilteringProcess.__init__(self, assimilation_configs, output_configs)
        #
        random_seed = self.assimilation_configs['random_seed']
        if random_seed is not None:
            assert isinstance(random_seed, int), "'random_seed' MUST be either None or an integer"
            self._random_seed = random_seed
        else:
            self._random_seed = None
        self._random_state = None  # to hold the state of the random number generator
        
        # Here we create a dictionary with all features required for the Machine-Learning algorithm;
        # This will be a list where each entry is a list of dictionaries (list of lists of dictionaries);
        # each entry is a list containing dictionaries, where each dictionary contains features corresponding to a single localization radius:
        # The leangth of this list will be equal to the number of assimilation cycles
        self.ML_features = []
        try:
            self.loc_radii_pool = list(loc_radii_pool)
        except(TypeError):
            print("loc_radii_pool has to be an iterable, e.g. a list or one-D numpy array")
            raise
        if len(self.loc_radii_pool) == 1:
            self._same_loc_radius = True
        else:
            self._same_loc_radius = use_same_loc_radius
        self._num_training_trials = num_training_trials
        self.featured_state_indexes = featured_state_indexes
        self.featured_correlation_indexes = featured_correlation_indexes
        self.test_phase_initial_index = test_phase_initial_index+1
        
        
        # DO NOT PUT IT IN THE SECOND BRANCH (TESTING PHASE) AS IT WILL RESET IT and it's parameters will be lost!
        ### Learning algorithms we have tried
        self.logistic_regression_model = linear_model.LogisticRegression(C=1.0)
        self.linear_regression_model=linear_model.LinearRegression()
        self.lasso_regression_model=linear_model.Lasso()
        self.forest_regression_model=RandomForestRegressor(max_depth=4, random_state=2)
        self.NearestNeighbors_regression_model=KNeighborsRegressor(n_neighbors=3, weights='distance' )
        #self.NN_regression_model= MLPRegressor(solver='lbfgs', alpha=1e-5, random_state=1)
        self.GNB_regression_model=GaussianNB()
        #self.GPR_regression_model=gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        self.SVM_regression_model=svm.SVR()
        self.SGD_regression_model=linear_model.SGDRegressor(loss='squared_epsilon_insensitive', n_iter=5000, penalty='elasticnet')
        self.PassiveAgressive_regression_model=linear_model.PassiveAggressiveRegressor()
        self.GradientBoosting_regression_model =GradientBoostingRegressor()
        
        ### counter for re-tuning the learning machine which is used for online learning
        self.tune_value = 1000 # every tune_value assimilation cycles we increment the learning in online learning, if you don't want to use online learning put a very large value here
        self.count_tune = 0 #counter for tuning
        # number of features in the training dataset
        self.nFeatures = 6 + len(self.featured_state_indexes) + len(self.featured_correlation_indexes)-1
        print ("number of features", self.nFeatures)
        # number of target variables the machine should estimate 
        self.nTarget=1 
        self.ds_toAdd = np.empty((self.tune_value, self.nFeatures)) # dataset to add is used during the online learning for incremental learning
        
        ### measuring the error in estimation
        self.count_iter=0
        self.estimatedRadii=[]
        self.err_estimationRMSE=[]
        self.testRadii=[]
        self.err_estimationRadii=[]
        self.klMeasure=[]
        self.klMeasure1=[]
        #
    
    #
    def recursive_assimilation_process(self, observations_list=None, obs_checkpoints=None, da_checkpoints=None, update_ref_here=False):
        """
        Loop over all assimilation cycles and output/save results (forecast, analysis, observations) 
        for all the assimilation cycles.
        
        Parameters
        ----------
        observations_list (default None): list of obs.observation_vector objects,
            A list containing observaiton vectors at specific obs_checkpoints to use for sequential filtering
            If not None, len(observations_list) must be equal to len(obs_checkpoints).
            If it is None, synthetic observations should be created sequentially
            
        obs_checkpoints (default None): iterable containing an observation timespan 
            Thsese are the time instances at which observation (synthetic or not) are given/generated
        
        da_checkpoints (default None): iterable containing an assimilation timespan 
            Thsese are the time instances at which filtering is carried out.
            If same as obs_checkpoints, assimilation is synchronous.
            
        update_ref_here (default False): bool,
            A flag to decide to whether to update the reference state here, 
            or request to updated it inside the filter.
                  
        Returns
        ----------
        None
        
        """
        # Retrieve the state of the random number generator:
        if self._random_seed is not None:
            self._random_state = np.random.get_state()
            np.random.seed(self._random_seed)
            
        # Call parent method
        FilteringProcess.recursive_assimilation_process(self, observations_list=observations_list, 
                                                        obs_checkpoints=obs_checkpoints, 
                                                        da_checkpoints=da_checkpoints, 
                                                        update_ref_here=update_ref_here
                                                        ) 
        
        # Reset the state of the random number generator:
        if self._random_seed is not None:
            np.random.set_state(self._random_state)
        #
        # Add functionality after the end of the whole process here:
        # The code added here will depend on what you want; we can discuss that when you attempt to add new stuff
        pass
        

    #
    def assimilation_cycle(self,
                           analysis_time,
                           forecast_time,
                           observation_time,
                           assimilation_timespan,
                           observation,
                           scr_output=False,
                           file_output=False,
                           update_reference=True
                           ):
        """
        Carry out filtering for the next assimilation cycle.
        Given the assimilation cycle is carried out by applying filtering using filter_object.filtering_cycle().
        filter_object.filtering_cycle() itself should carry out the two steps: forecast, analysis.
        This function modifies the configurations dictionary of the filter object to carry out a single 
        filtering cycle.
        
        Parameters
        ----------
        analysis_time: scalar,
            The time instance at which analysis step of the filter is carried out
            
        forecast_time: scalar,
            The time instance at which forecast step of the filter is carried out
        
        observation_time: scalar,
            The time instance at which observation is available
        
        assimilation_timespan: an iterable containing the beginning and the end of the assimilation cycle i.e. len=2 
            A timespan for the filtering process on the current assimilation cycle.
            - analysis_time, obs_time, and forecast_time all must be within in that timespan limits
            - The reference_state is always given at the beginning of the current timespan and
              updated here or inside the filter cycle while calculating RMSE.
          
          
        observation: model.observation_vector object,
            The observaiton vector to use for assimilation/filtering
            
        scr_output (default False): bool,
            A boolean flag to cotrol whether to output results to the screen or not
            The per-cycle screen output options are set by the filter configs.
            
        file_output (default False): bool,
            A boolean flag to cotrol whether to save results to file or not. 
            The per-cycle file output options are set by the filter configs.
            
        update_reference (default False): bool,
            A flag to decide to whether to request updating the reference state in by the filter.
                                   
        Returns
        ----------
        None
            
        """
        # get and update the configurations of the filter. Both output and filter configs are handled here
        # Update filter configs
        if self.filter.filter_configs['forecast_first'] != self.assimilation_configs['forecast_first']:
            self.filter.filter_configs['forecast_first'] != self.assimilation_configs['forecast_first']
        self.filter.filter_configs['observation_time'] = observation_time
        self.filter.filter_configs['observation'] = observation
        self.filter.filter_configs['timespan'] = assimilation_timespan
        self.filter.filter_configs['analysis_time'] = analysis_time
        self.filter.filter_configs['forecast_time'] = forecast_time

        # Update output configs (based on flag and iterations)
        if self.filter.output_configs['scr_output'] != scr_output:
            self.filter.output_configs['scr_output'] = scr_output
        if self.filter.output_configs['file_output'] != file_output:
            self.filter.output_configs['file_output'] = file_output

        # Now start the filtering cycle; 
        self.adaptive_filtering_cycle(update_reference=update_reference)
        # Add functionality after the end of the each assimilation cycle here:
        # The code added here will depend on what you want; we can discuss that when you attempt to add new stuff
        pass
        

    def adaptive_filtering_cycle(self, update_reference=True):
        """
        This method is a modification to filters_base.filtering_cycle
        Carry out a single assimilation cycle. Forecast followed by analysis or the other way around.
        All required variables are obtained from 'filter_configs' dictionary.
        This base method is designed to work in both ensemble, and standard framework. 
        You can override it for sure in your filter.
        
        Parameters
        ----------
        update_reference (default True): bool,
            A flag to decide whether to update the reference state in the filter or not.
                  
        Returns
        ----------
        None
            
        """
        filter_obj = self.filter
        model = filter_obj.filter_configs['model']
        timespan = filter_obj.filter_configs['timespan']
        observation_time = filter_obj.filter_configs['observation_time']
        forecast_time = filter_obj.filter_configs['forecast_time']
        analysis_time = filter_obj.filter_configs['analysis_time']
        forecast_first = filter_obj.filter_configs['forecast_first']
        apply_preprocessing = filter_obj.filter_configs['apply_preprocessing']
        apply_postprocessing = filter_obj.filter_configs['apply_postprocessing']
        #forecast_ensemble = filter_obj.filter_configs['forecast_ensemble']
                
        # calculate the initial RMSE: this is the RMSE before assimilation at the first entry of the time-span
        try:
            reference_time = filter_obj.filter_configs['reference_time']
        except(KeyError, ValueError, AttributeError, NameError):
            raise ValueError("Couldn't find the reference time in the configurations dictionary")
        else:
            if reference_time != timespan[0]:
                raise ValueError("Reference time does not match the initial time of the reference time-span!")

        #
        # -------------------------------------------------------
        # Ahmed Remark: Time Stamp [08:24 pm, July 16, 2017]
        # -------------------------------------------------------
        # Some models, such as the current QG-1.5 implementation in DATeS, force Dirichlet boundary conditions with value 0;
        # In such case, the truth, and the analysis ensemble have the same value. In this case, 
        # we are at the mercy of how Numpy (or alternative packages) rank equal elements, and also will be dependent on how rank_hist function is implemented!
        # We can modify first_var, var_skp, and las_var to take care of it, but
        #
        # A better approach, is as follows: These models (mentioned above) should provide the indexes (as required by DATeS), in the model_configs dictionary.
        # I have modified the rank_hist function to handle such cases. Now it works fine.
        # -------------------------------------------------------        
        try:
            model_boundary_indexes = model.model_configs['boundary_indexes']
        except(NameError, AttributeError, KeyError):
            # this handles the cases if 'model_configs' is not defined, and if 'boundary_indexes' is not a valid key.
            model_boundary_indexes = None
            pass
        except:
            print("Failed while trying to retrieve the model boundary indexes. The exception raised in quite UNEXPECTED! ")
            raise
        # -------------------------------------------------------
        
        #
        #
        if forecast_first:
            # Analysis ensemble should be given first in this case
            try:
                initial_state = filter_obj.filter_configs['analysis_state']
            except(KeyError, ValueError, AttributeError, NameError):
                try:
                    initial_state = utility.ensemble_mean(filter_obj.filter_configs['analysis_ensemble'])
                except:
                    raise ValueError("Couldn't find either analysis state or analysis ensemble while forecast should be done first!")
            finally:
                if initial_state is None:
                    try:
                        initial_state = utility.ensemble_mean(filter_obj.filter_configs['analysis_ensemble'])
                    except:
                        raise ValueError("Couldn't find either analysis state or analysis ensemble while forecast should be done first!")
        else:
            # Forecast ensemble should be given first in this case
            try:
                initial_state = filter_obj.filter_configs['forecast_state']
            except:
                try:
                    initial_state = utility.ensemble_mean(filter_obj.filter_configs['forecast_ensemble'])
                except:
                    raise ValueError("Couldn't find either forecast state or forecast ensemble while analysis should be done first!")
            finally:
                if initial_state is None:
                    try:
                        initial_state = utility.ensemble_mean(filter_obj.filter_configs['forecast_ensemble'])
                    except:
                        raise ValueError("Couldn't find either forecast state or forecast ensemble while analysis should be done first!")

        # Retrieve the reference state and evaluate initial root-mean-squared error
        reference_state = filter_obj.filter_configs['reference_state'].copy()
        initial_rmse = utility.calculate_rmse(initial_state, reference_state)

        #
        # Start the filtering process: preprocessing -> filtering(forecast->+<-anslsysis) -> postprocessing
        if apply_preprocessing:
            filter_obj.cycle_preprocessing()

        
        if not forecast_first and analysis_time != min(forecast_first, analysis_time, observation_time):
            # this is a double check!
            raise ValueError("While ANALYSIS should be done first, confusion occurred with times given!\n"
                             "\tCycle timespan:%s"
                             "\tObservation time: %f\n"
                             "\tForecast time: %f\n"
                             "\tAnalysis time: %f" % (repr(timespan), observation_time, forecast_time, analysis_time))
        elif not forecast_first:
            raise NotImplementedError("This is not implemented in the adaptive version for simplicity")
            #
                
        else:
            # forecast should be done first
            # print("\n\n\n\n\n\n FORECAST FIRTS \n\n\n\n\n")
            state_size = filter_obj.filter_configs['model'].state_size()
            try:
                reference_time = filter_obj.filter_configs['reference_time']
            except:
                raise ValueError("Couldn't find reference time in the configurations dictionary")
            else:
                if reference_time != timespan[0]:
                    raise ValueError("Reference time does not match the initial time of the reference time-span!")
            local_checkpoints = [reference_time, timespan[-1]]
            
            # Forecast step:
            filter_obj.forecast()
            # 
            try:
                forecast_state = filter_obj.filter_configs['forecast_state']
            except (NameError, AttributeError):
                raise NameError("forecast_state must be updated by the filter "
                                "and added to 'filter_obj.filter_configs'!")
                        
            tmp_trajectory = model.integrate_state(initial_state=reference_state, checkpoints=timespan)
            if isinstance(tmp_trajectory, list):
                up_reference_state = tmp_trajectory[-1].copy()
            else:
                up_reference_state = tmp_trajectory.copy()
            reference_time = local_checkpoints[-1]
            
            # Update the reference state
            if update_reference:
                filter_obj.filter_configs['reference_state'] = up_reference_state.copy()
                filter_obj.filter_configs['reference_time'] = reference_time
                
            # observation = filter_obj.filter_configs['observation']
            #
            forecast_rmse = utility.calculate_rmse(up_reference_state, forecast_state, state_size)
            
            
            # Machine Learning update:
            # Append a ML features dictionary in the last entry in the machine learning features list:
            ML_features_dict = self.initialize_ML_features_dict()
            self.ML_features.append(ML_features_dict)  # This will be accessed and updated as appropriate:
            
           
            # Update ML features list with analysis results:
            try:
                
                state_size = filter_obj.filter_configs['model'].state_size()
                self.ML_features[-1]["assim_cycle_number"] = len(self.ML_features)
                #
                self.ML_features[-1]["min_ensemble_forecast_variance"] = filter_obj.prior_variances.min()
                self.ML_features[-1]["max_ensemble_forecast_variance"] = filter_obj.prior_variances.max()
                if isinstance(filter_obj.prior_variances, (StateVectorBase, np.ndarray)):
                    forecast_variance_sum = filter_obj.prior_variances.sum()              
                else:
                    forecast_variance_sum = np.sm(filter_obj.prior_variances[:])
                self.ML_features[-1]["sum_ensemble_forecast_variance"] = forecast_variance_sum
                self.ML_features[-1]["mean_ensemble_forecast_variance"] = forecast_variance_sum/float(state_size)
                #
                self.ML_features[-1]["min_ensemble_forecast_mean"] = forecast_state.min()
                self.ML_features[-1]["max_ensemble_forecast_mean"] = forecast_state.max()
                #
                if isinstance(forecast_state, (StateVectorBase, np.ndarray)):
                    forecast_mean_sum = forecast_state.sum()
                else:
                    forecast_mean_sum = np.sum(forecast_state[:])
                self.ML_features[-1]["sum_ensemble_forecast_mean"] = forecast_mean_sum
                self.ML_features[-1]["mean_ensemble_forecast_mean"] = forecast_mean_sum / float(state_size)
                #
                self.ML_features[-1]["state_features_indexes"] = self.featured_state_indexes
                if self.featured_state_indexes is not None:
                    self.ML_features[-1]["state_features_values"] = forecast_state[np.squeeze(self.featured_state_indexes[:])]
                    
                self.ML_features[-1]["state_indexes_correlation"] = self.featured_correlation_indexes
                if self.featured_correlation_indexes is not None:
                    forecast_ensemble = self.filter.filter_configs['forecast_ensemble']
                    ensemble_size = len(forecast_ensemble)
                    num_indexes = len(self.featured_correlation_indexes)
                    local_ensemble = np.empty((num_indexes, ensemble_size )) # Each row of `m` represents a variable
                    for ens_ind in xrange(ensemble_size):
						local_ensemble[:, ens_ind] = forecast_ensemble[ens_ind][self.featured_correlation_indexes]
                    
                    #
                    # -------------------------------------------------------
                    # Ahmed Remark: Time Stamp [10:00 pm, July 15, 2017]
                    # -------------------------------------------------------
                    # Here is the main issue that caused Nans: 
                    # np_correlations = np.corrcoef(local_ensemble)  # DON'T USE THIS ANYMORE!
                    # I've created a more advanced sample covariance matrix constructor capable of handling cases where the covariance matrix is singular!
                    # You can check this function in '_utility_stat.py' module.
                    # We can discuss futher in details during our next meeting ...
                    # -------------------------------------------------------
                    correlations = utility.ensemble_covariance_matrix(local_ensemble, corr=True, zero_nans=True)
                    # -------------------------------------------------------
                    '''
                    corr0=correlations[0][1]
                    corr1=correlations[1][2]
                    corr2=correlations[2][3]
                    corr3=correlations[3][4]
                    corr4=correlations[4][5]
                    #self.ML_features[-1]["states_correlation"]=[corr0, corr1, corr2, corr3, corr4]
                    '''
                    
                    corrs = np.empty((num_indexes-1))
                    for i in xrange (num_indexes-1):
     				   corrs[i] = correlations[i][i+1]
     				   
                    self.ML_features[-1]["states_correlation"]=corrs
                    
                ##forecast_RMSE
                self.ML_features[-1]["forecast_RMSE"] = forecast_rmse
                
                # Forecasted Observation RMSE:
                observation = filter_obj.filter_configs['observation']  # this is y
                theoretical_observation = model.evaluate_theoretical_observation(forecast_state)  # this is H(x)
                try:
                    observation_size = model.observation_vector_size()
                except (NameError, AttributeError):
                    observation_size = None
                if observation_size is None:
                    forecast_observ_RMSE = utility.calculate_rmse(observation, theoretical_observation)
                else:
                    forecast_observ_RMSE = utility.calculate_rmse(observation, theoretical_observation, observation_size)
                # print("Forecasted Observation RMSE %f" % forecast_observ_RMSE)
                self.ML_features[-1]["forecast_observ_RMSE"] = forecast_observ_RMSE
                
                #
                #self.ML_features[-1]["localization_radius"] = filter_obj.filter_configs['localization_radius']
                
                
            except (KeyError):
                print("Couldn't find the appropriate Key. Please check the initialization of the machine learning dict!")
                raise
            
            # Analysis step:      
            # Leargning Phase:
            if len(self.ML_features) < self.test_phase_initial_index:
                # Now, we try analysis with different radii, and save analysis mean, and analysis mean, and rmse for each:
                # You can change that if you want to change that list every cycle, I recomment to updated to be a range around the winner from previeous cycle
                loc_radii_pool = list(self.loc_radii_pool)  
                winner_radius = np.infty
                winner_analysis_rmse = np.infty
                winner_observ=np.infty
                winner_objective=np.infty
                analysis_rmse_dict = {'radii':[], 'analysis_rmse':[], 'analysis_mean':[]}
                winner_distance=np.infty
                distance_dict = {'radii':[], 'distance':[], 'analysis_rmse':[],'analysis_mean':[] }
                objective_dict= {'radii':[], 'distance':[], 'analysis_rmse':[],'analysis_mean':[],'combination_objective':[], 'observ_rmse':[]}
                W=np.zeros((3));
                W[0]=1
                W[1]=0
                W[2]=0
                #
                if self._same_loc_radius:
                    for loc_radius_ind in xrange(len(loc_radii_pool)):
                        #
                        loc_radius = loc_radii_pool[loc_radius_ind]
                        
                        #
                        # Retrieve the state of the random number generator:
                        if self._random_seed is not None:
                            local_random_state = np.random.get_state()
                            np.random.seed(self._random_seed)
                        
                        if self._same_loc_radius:
                            print("LEARNING PHASE: Trying analysis with localization radius = %4.2f >> " % loc_radius),
                        else:
                            print("LEARNING PHASE: Trying analysis with random localization radii from: %s >> " % str(loc_radius)),
                            
                        filter_obj.filter_configs['localization_radius'] = loc_radius
                        filter_obj.analysis()
                        #
                        try:
                            analysis_state = filter_obj.filter_configs['analysis_state']
                        except (NameError, AttributeError):
                            raise NameError("analysis_state must be updated by the filter "
                                            "and added to 'filter_obj.filter_configs'!")
                        tmp_anal_rmse = utility.calculate_rmse(up_reference_state, analysis_state, state_size)
                        print("Analysis RMSE = %f" % tmp_anal_rmse)
                        #
                        ################# From here is modified for combination measure ##################################
                        
                        analysis_ensemble=copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        ensemble_size = len(forecast_ensemble)
                        analysis_ensemble_np = np.empty(shape=(state_size, ensemble_size))
                        for col_ind in xrange (ensemble_size):
                        	analysis_ensemble_np[:, col_ind] = analysis_ensemble[col_ind].get_numpy_array()
                    	
                    	
                    	#
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [08:24 pm, July 16, 2017]
                        # -------------------------------------------------------
                        # Some models, such as the current QG-1.5 implementation in DATeS, force Dirichlet boundary conditions with value 0;
                        # In such case, the truth, and the analysis ensemble have the same value. In this case, 
                        # we are at the mercy of how Numpy (or alternative packages) rank equal elements, and also will be dependent on how rank_hist function is implemented!
                        # We can modify first_var, var_skp, and las_var to take care of it, but
                        #
                        # A better approach, is as follows: These models (mentioned above) should provide the indexes (as required by DATeS), in the model_configs dictionary.
                        # I have modified the rank_hist function to handle such cases. Now it works fine.
                        # -------------------------------------------------------
                        ranks_freq, ranks_rel_freq, bins_bounds, fig_hist = utility.rank_hist(analysis_ensemble_np, 
                                                                                              up_reference_state.get_numpy_array(), 
                                                                                              first_var=0, 
                                                                                              last_var=None, 
                                                                                              var_skp=1,
                                                                                              #draw_hist=True,  # you can turn this off, after checking the plots, and the fitted pdf plots
                                                                                              hist_type='relfreq', 
                                                                                              first_time_ind=0, 
                                                                                              last_time_ind=None,
                                                                                              time_ind_skp=1,
                                                                                              ignore_indexes=model_boundary_indexes) 
                        
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [9:23 pm, July 16, 2017]
                        # -------------------------------------------------------
                        # Fitting a distribution is done incorrectly!
                        # params = dist.fit(ranks_rel_freq)
                        #
                        # Here is a serious issue; when you fit a distribution, you need to pass data as the first argument x.
                        # The first argument, i.e. the data, is the values of the variable sampled from the distribution from which data are assumed to be sampled.
                        # See 'http://glowingpython.blogspot.de/2012/07/distribution-fitting-with-scipy.html' for example
                        # This can be done as follows:
                        data = []
                        for fr, bn in zip(ranks_freq, bins_bounds):
                            data += [float(bn)]*fr
                        data = np.asarray(data)
                        #
                        dist = scipy.stats.beta
                        params = dist.fit(data)
                        # -------------------------------------------------------
                        # Fit a beta distribution:
                        betas = np.array([params[0], params[1]])
                        B0 = betas.sum()                        
                        #
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [9:23 pm, July 16, 2017]
                        # -------------------------------------------------------
                        # The fitted Beta distribution is assumed to have a domain/support [0, 1];
                        # We are expected to construct a pdf curve, and stretch it to the domain [0, Nens] to mach the rank histogram bounds
                        #
                        # pdf_fitted_beta = dist.pdf(bins_bounds, *params[:-2], loc=params[-2], scale=params[-1])
                        # -------------------------------------------------------
                        pdf_x = np.linspace(dist.ppf(0.01, betas[0], betas[1]), dist.ppf(0.99, params[0], params[1]), 50)
                        pdf_y = dist.pdf(pdf_x, *params[:-2], loc=params[-2], scale=params[-1])
                        if fig_hist is not None:
                            # Add the fitted pdf
                            # Here, I am assuming 'ranks_rel_freq' is used to plot the rank_hist; you can make it more general, but no need
                            if False:
                                if pdf_y.min() < ranks_rel_freq.min():
                                    pdf_y += abs(pdf_y.min()-ranks_rel_freq.min())
                                else:
                                    pdf_y -= abs(pdf_y.min()-ranks_rel_freq.min())
                            
                            # shift X values to 0 to ensemble_size
                            a, b = pdf_x.min(), pdf_x.max()
                            c, d = 0, ensemble_size
                            pdf_x = c + ((d-c)/(b-a)) * (pdf_x-a)
                            
                            ax = fig_hist.gca()
                            ax.plot(pdf_x, pdf_y, 'r-', linewidth=5, label='fitted beta', zorder=1)
                            
                            # Update y limits; just in-case!
                            ylim = ax.get_ylim()
                            ax.set_ylim([ylim[0], max(ylim[-1], max(pdf_y.max(), ranks_rel_freq.max()))]) 
                            #
                            #plt.show()
                            #
                        
                        # -------------------------------------------------------
                        
                        alphas = np.array([1, 1])
                        A0 = alphas.sum()
                        KL_distance = sp.gammaln(A0) - np.sum(sp.gammaln(alphas)) \
                                      - sp.gammaln(B0) + np.sum(sp.gammaln(betas)) \
                                      + np.sum((alphas-betas) * (sp.digamma(alphas)-sp.digamma(A0)))
                        
                        #
                        # TODO: Remove this section after you are done debugging everything!!!
                        if False:
                            print("params", params)
                            print("betas", betas)
                            print("alphas", alphas)
                            print("KL_distance", KL_distance)
                            print("ranks_rel_freq", ranks_rel_freq)
                            print("pdf_x, pdf_y", pdf_x, pdf_y)                        
                        #tmp_x = raw_input("Press Something to continue... << ")
                        #
                        
                        #
                        tmp_distance = KL_distance
                        
                        mu = np.mean(ranks_rel_freq)
                        var = np.var(ranks_rel_freq) 
                        print('the kl distance is ',KL_distance )
                        #print('the hellinger distance is ', HL_distance)
                        #print('the entropy distance is ',Ent_distance )
                        mu=np.sum(ranks_rel_freq*bins_bounds)/bins_bounds[-1]
                        var=np.float((np.sum(ranks_rel_freq*np.square(bins_bounds-mu)))/(bins_bounds[-1]-1))
                        #print ("the variance is",var )
                        observation = filter_obj.filter_configs['observation']  # this is y
                        #### I changed the following statement to the analysis state instead of forecast state
                        theoretical_observation = model.evaluate_theoretical_observation(analysis_state)  # this is H(x), 
                        tmp_observ_rmse=utility.calculate_rmse(observation,theoretical_observation)
                        print('tmp_observ_rmse is', tmp_observ_rmse)
                        tmp_objective=	W[0]*tmp_anal_rmse+W[1]*tmp_distance+W[2]*tmp_observ_rmse
                        print "{{{{{{{{{{{{{{{ tmp_objective", tmp_objective, " }}}}}}}}}}}}}}}}}"
                        if tmp_objective < winner_objective:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_observ=tmp_observ_rmse
                        	winner_objective=tmp_objective
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        objective_dict['radii'].append(loc_radius)
                        objective_dict['distance'].append(KL_distance)
                        objective_dict['analysis_rmse'].append(tmp_anal_rmse)
                        objective_dict['observ_rmse'].append(winner_observ)
                        objective_dict['analysis_mean'].append(analysis_state)
                        objective_dict['combination_objective'].append(tmp_objective)
                       	'''
                       
                        if tmp_anal_rmse < winner_analysis_rmse:
                            winner_analysis_rmse = tmp_anal_rmse
                            winner_radius = loc_radius
                            winner_state = analysis_state.copy()
                            winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        #
                        analysis_rmse_dict['radii'].append(loc_radius)
                        analysis_rmse_dict['analysis_rmse'].append(tmp_anal_rmse)
                        analysis_rmse_dict['analysis_mean'].append(analysis_state)
                        
                        if tmp_distance < winner_distance:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        distance_dict['radii'].append(loc_radius)
                        distance_dict['distance'].append(KL_distance)
                        distance_dict['analysis_rmse'].append(tmp_anal_rmse)
                        distance_dict['analysis_mean'].append(analysis_state)
                        '''
                    	################# up to here is modified for combination measure ##################################
                        
                        
                        # Reset the state of the random number generator:
                        if self._random_seed is not None:
                            np.random.set_state(local_random_state)
                            
                else:
                    num_training_trials = self._num_training_trials
                    for trial_ind in xrange(num_training_trials):
                        #
                        #
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [10:15 pm, July 15, 2017]
                        # -------------------------------------------------------
                        # loc_radius is a vector of the same size as the model grid that is used to localize covariances of state-vector entris.
                        # observation_size = model.observation_vector_size()
                        # loc_radius = [loc_radii_pool[i] for i in [np.random.randint(len(loc_radii_pool)) for d in xrange(observation_size)]]
                        # -------------------------------------------------------
                        loc_radius = [loc_radii_pool[i] for i in [np.random.randint(len(loc_radii_pool)) for d in xrange(state_size)]]
                        loc_radius = np.asarray(loc_radius)
                        # -------------------------------------------------------
                         
                         # REMOVE >> loc_radius = loc_radii_pool  # Choose random radius for each grid point in the observation space

                        #
                        # Retrieve the state of the random number generator:
                        if self._random_seed is not None:
                            local_random_state = np.random.get_state()
                            np.random.seed(self._random_seed)
                        
                        if self._same_loc_radius:
                            print("LEARNING PHASE: Trying analysis with localization radius = %4.2f: " % loc_radius)
                        else:
                            print("LEARNING PHASE: Trying analysis with random localization radii from: %s: " % str(loc_radius))
                            
                        filter_obj.filter_configs['localization_radius'] = loc_radius
                        filter_obj.analysis()
                        #
                        try:
                            analysis_state = filter_obj.filter_configs['analysis_state']
                        except (NameError, AttributeError):
                            raise NameError("analysis_state must be updated by the filter "
                                            "and added to 'filter_obj.filter_configs'!")
                        tmp_anal_rmse = utility.calculate_rmse(up_reference_state, analysis_state, state_size)
                        # print("Analysis RMSE = %f" % tmp_anal_rmse)
                        if np.isnan(tmp_anal_rmse):
                            print "WARNING: Trial", trial_ind, "out of ", num_training_trials-1, "DIVERGED. Continuing next trial..."
                            continue
                        #
                        ################# From here is modified for combination measure ##################################
                        
                        analysis_ensemble=copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        ensemble_size = len(forecast_ensemble)
                        analysis_ensemble_np = np.empty(shape=(state_size, ensemble_size))
                        for col_ind in xrange (ensemble_size):
                        	analysis_ensemble_np[:, col_ind] = analysis_ensemble[col_ind].get_numpy_array()
                        	
                        #
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [08:24 pm, July 16, 2017]
                        # -------------------------------------------------------
                        # Some models, such as the current QG-1.5 implementation in DATeS, force Dirichlet boundary conditions with value 0;
                        # In such case, the truth, and the analysis ensemble have the same value. In this case, 
                        # we are at the mercy of how Numpy (or alternative packages) rank equal elements, and also will be dependent on how rank_hist function is implemented!
                        # We can modify first_var, var_skp, and las_var to take care of it, but
                        #
                        # A better approach, is as follows: These models (mentioned above) should provide the indexes (as required by DATeS), in the model_configs dictionary.
                        # I have modified the rank_hist function to handle such cases. Now it works fine.
                        # -------------------------------------------------------
                        ranks_freq, ranks_rel_freq, bins_bounds, fig_hist = utility.rank_hist(analysis_ensemble_np, 
                                                                                              up_reference_state.get_numpy_array(), 
                                                                                              first_var=0, 
                                                                                              last_var=None, 
                                                                                              var_skp=1,
                                                                                              #draw_hist=True,  # you can turn this off, after checking the plots, and the fitted pdf plots
                                                                                              hist_type='relfreq', 
                                                                                              first_time_ind=0, 
                                                                                              last_time_ind=None,
                                                                                              time_ind_skp=1,
                                                                                              ignore_indexes=model_boundary_indexes) 
                        
                        # -------------------------------------------------------
                        data = []
                        for fr, bn in zip(ranks_freq, bins_bounds):
                            data += [float(bn)]*fr
                        data = np.asarray(data)
                        #
                        dist = scipy.stats.beta
                        params = dist.fit(data)
                        # -------------------------------------------------------
                        # Fit a beta distribution:
                        betas = np.array([params[0], params[1]])
                        B0 = betas.sum()                        
                        pdf_x = np.linspace(dist.ppf(0.01, betas[0], betas[1]), dist.ppf(0.99, params[0], params[1]), 50)
                        pdf_y = dist.pdf(pdf_x, *params[:-2], loc=params[-2], scale=params[-1])
                        if fig_hist is not None:
                            if False:
                                if pdf_y.min() < ranks_rel_freq.min():
                                    pdf_y += abs(pdf_y.min()-ranks_rel_freq.min())
                                else:
                                    pdf_y -= abs(pdf_y.min()-ranks_rel_freq.min())
                            
                            a, b = pdf_x.min(), pdf_x.max()
                            c, d = 0, ensemble_size
                            pdf_x = c + ((d-c)/(b-a)) * (pdf_x-a)
                            ax = fig_hist.gca()
                            ax.plot(pdf_x, pdf_y, 'r-', linewidth=5, label='fitted beta', zorder=1)
                            ylim = ax.get_ylim()
                            ax.set_ylim([ylim[0], max(ylim[-1], max(pdf_y.max(), ranks_rel_freq.max()))]) 
                            #
                            #plt.show()
                            #
                        alphas = np.array([1,1])
                        A0 = alphas.sum()
                        KL_distance = sp.gammaln(A0) - np.sum(sp.gammaln(alphas)) \
                                      - sp.gammaln(B0) + np.sum(sp.gammaln(betas)) \
                                      + np.sum((alphas-betas) * (sp.digamma(alphas)-sp.digamma(A0)))
                        
                        tmp_distance=KL_distance            
                        if np.isnan(KL_distance):
                            #plt.show()
						mu = np.mean(ranks_rel_freq)
                        var = np.var(ranks_rel_freq) 
                        print('the kl distance is ',KL_distance )
                        #print('the hellinger distance is ', HL_distance)
                        #print('the entropy distance is ',Ent_distance )
                        mu=np.sum(ranks_rel_freq*bins_bounds)/bins_bounds[-1]
                        var=np.float((np.sum(ranks_rel_freq*np.square(bins_bounds-mu)))/(bins_bounds[-1]-1))
                        #print ("the variance is",var )
                        observation = filter_obj.filter_configs['observation']  # this is y
                        #### I changed the following statement to the analysis state instead of forecast state
                        theoretical_observation = model.evaluate_theoretical_observation(analysis_state)  # this is H(x), 
                        tmp_observ_rmse=utility.calculate_rmse(observation,theoretical_observation)
                        print('tmp_observ_rmse is', tmp_observ_rmse)
                        tmp_objective=	W[0]*tmp_anal_rmse+W[1]*tmp_distance+W[2]*tmp_observ_rmse
                        print "{{{{{{{{{{{{{{{ tmp_objective", tmp_objective, " }}}}}}}}}}}}}}}}}"
                        if tmp_objective < winner_objective:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_observ=tmp_observ_rmse
                        	winner_objective=tmp_objective
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        objective_dict['radii'].append(loc_radius)
                        objective_dict['distance'].append(KL_distance)
                        objective_dict['analysis_rmse'].append(tmp_anal_rmse)
                        objective_dict['observ_rmse'].append(winner_observ)
                        objective_dict['analysis_mean'].append(analysis_state)
                        objective_dict['combination_objective'].append(tmp_objective)
                       	'''
                       
                       
                        if tmp_anal_rmse < winner_analysis_rmse:
                            winner_analysis_rmse = tmp_anal_rmse
                            winner_radius = loc_radius
                            winner_state = analysis_state.copy()
                            winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        #
                        analysis_rmse_dict['radii'].append(loc_radius)
                        analysis_rmse_dict['analysis_rmse'].append(tmp_anal_rmse)
                        analysis_rmse_dict['analysis_mean'].append(analysis_state)
                        
                        if tmp_distance < winner_distance:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        distance_dict['radii'].append(loc_radius)
                        distance_dict['distance'].append(KL_distance)
                        distance_dict['analysis_rmse'].append(tmp_anal_rmse)
                        distance_dict['analysis_mean'].append(analysis_state)
                        '''
                    	################# up to here is modified for combination measure ##################################
                        
                        
                        # Reset the state of the random number generator:
                        if self._random_seed is not None:
                            np.random.set_state(local_random_state)
                            
                try:
                    winner_state
                except(NameError, UnboundLocalError):
                    print "All trials have diverged..."
                    print "Quitting Adaptive localization..."
                    raise ValueError
                
                
                self.ML_features[-1]['analysis_RMSE_dict'] = analysis_rmse_dict
                
                # assign WINNING radius (of smaller RMSE) and analysis to the filter object     
                analysis_rmse = winner_analysis_rmse
                filter_obj.filter_configs['localization_radius'] = winner_radius
                filter_obj.filter_configs['analysis_state'] = winner_state
                filter_obj.filter_configs['analysis_ensemble'] = winner_analysis_ensemble  # a shallow copy is sufficient
                #print ">>>>>>>>>>>>>>>>>>>>> winner_state >>>>>>>>>>>>>>>>>>>>", winner_state
                print ">>>>>>>>>>>>>>>>>>>>> winner_radius >>>>>>>>>>>>>>>>>>>>", winner_radius
                self.ML_features[-1]['localization_radius']=winner_radius
                print ("this is the last row in training phase :", self.ML_features[-1]['localization_radius'])
                print ("ML feature rows are", len(self.ML_features))
                #self.count_iter+=1
                #print ('ML features ', self.ML_features[-1]['localization_radius'], self.count_iter )
                #
            else:
                # before testing, Creating the final dataset and training the machine
                if len(self.ML_features) == self.test_phase_initial_index:
                    if (self._same_loc_radius==False):
                        self.nTarget += len(self.ML_features[0]['localization_radius'])-1   
                    ds_final = np.zeros ((len(self.ML_features), self.nFeatures+self.nTarget))
                    for i in xrange (len(self.ML_features)):
                        ds_final[i, 0]=self.ML_features[i]['mean_ensemble_forecast_mean']
                        ds_final[i, 1]=self.ML_features[i]['max_ensemble_forecast_mean']
                        ds_final[i, 2]=self.ML_features[i]['min_ensemble_forecast_mean']
                        ds_final[i, 3]=self.ML_features[i]['sum_ensemble_forecast_mean']
                        ds_final[i, 4]=self.ML_features[i]['forecast_observ_RMSE']
                        ds_final[i, 5]=self.ML_features[i]['forecast_RMSE']
                        count = 6
                        for t in xrange (len(self.featured_state_indexes)):
                            ds_final[i, count]=self.ML_features[i]['state_features_values'][t]
                            count += 1
                        for t in xrange (num_indexes-1):
                           ds_final[i, count]=self.ML_features[i]['states_correlation'][t]
                           count += 1
                        _loc_radii = self.ML_features[i]['localization_radius']
                        if np.isscalar(_loc_radii):
                            ds_final[i, count] = _loc_radii
                        else:
                            #print "count:count+len(_loc_radii)", count, ":", count+len(_loc_radii)
                            #print "also _loc_radii", _loc_radii
                            #print "also ds_final[i, count:count+len(_loc_radii)].shape", ds_final[i, count:count+len(_loc_radii)].shape
                            #ds_final[i, count:count+len(_loc_radii)] = np.asarray(_loc_radii).squeeze()
                            #ds_final[i, count:count+len(self.ML_features[0]['localization_radius'])] = _loc_radii
                            ds_final[i, count:count+state_size] = _loc_radii
                            
                        #print ("X_train values:", ds_final[i, 0:self.nFeatures])
                        #print ("Y_train values:", ds_final[i, self.nFeatures:self.nFeatures+self.nTarget])
                        #print ("for i , _loc_radii is :", i , self.ML_features[i]['localization_radius'])
                        #print ("loc radi is", self.ML_features[i]['localization_radius'])
                           
                            
					
                    # Train the model given the features X_train, and the prediction parameters (localization radii) Y_train
                    X_train = ds_final[0:self.test_phase_initial_index-1, 0:self.nFeatures]
                    Y_train = ds_final[0:self.test_phase_initial_index-1, self.nFeatures:self.nFeatures+self.nTarget]
                    print ("X_train shape is :", np.shape(X_train))	
                    print ("Y_train shape is :", np.shape(Y_train))
                    
    
                    #### Trying the learning process with different models
                    #self.logistic_regression_model.fit(X_train, Y_train)
                    #self.linear_regression_model.fit(X_train, Y_train)
                    #self.NN_regression_model.fit(X_train, Y_train)
                    #self.GNB_regression_model.fit(X_train, Y_train)
                    #self.GPR_regression_model.fit(X_train, Y_train)
                    #self.SVM_regression_model.fit(X_train, Y_train)
                    #self.SGD_regression_model.partial_fit(X_train, Y_train)
                    #self.PassiveAgressive_regression_model.partial_fit(X_train, Y_train)
                    #self.GradientBoosting_regression_model.fit(X_train, Y_train)
                    ##### These are with for multi output:
                    self.NearestNeighbors_regression_model.fit(X_train, Y_train)
                    # self.forest_regression_model.fit(X_train, np.ravel(Y_train))
                    #self.lasso_regression_model.fit(X_train, Y_train)
                    #self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=4,random_state=2))
                    #self.regr_multirf = MultiOutputRegressor(self.PassiveAgressive_regression_model)
                    #self.regr_multirf = MultiOutputRegressor(self.NearestNeighbors_regression_model)
                    #self.regr_multirf = MultiOutputRegressor(self.lasso_regression_model)
                    #self.regr_multirf.fit(X_train, Y_train)
                    #print(Y_train)
    
                #### Testing phase
                ### Reset the count_tune every tune_value assimilation cycles and re-train the model
                print  (">>>>>>>>>>>>>>>>>>>>>TESTING PHASE STARTED>>>>>>>>>>>>>>>>>>>>>")
         
                if self.count_tune == self.tune_value:
                    self.count_tune=0
                    print ("the shape of ds_toadd is ", self.ds_toAdd)
                    self.PassiveAgressive_regression_model.partial_fit(self.ds_toAdd[:,0:self.nFeatures], self.ds_toAdd[:,self.nFeatures:self.nFeatures+self.nTarget])
                    self.ds_toAdd = np.empty((self.tune_value, self.nFeatures+self.nTarget)) 
               
                
                # Retrieve prediction features:
            
                features_vec = np.empty((1, self.nFeatures))  
                features_vec[0, 0]=self.ML_features[-1]['mean_ensemble_forecast_mean']
                features_vec[0, 1]=self.ML_features[-1]['max_ensemble_forecast_mean']
                features_vec[0, 2]=self.ML_features[-1]['min_ensemble_forecast_mean']
                features_vec[0, 3]=self.ML_features[-1]['sum_ensemble_forecast_mean']
                features_vec[0, 4]=self.ML_features[-1]['forecast_observ_RMSE']
                features_vec[0, 5]=self.ML_features[-1]['forecast_RMSE']
                count = 6
                for t in xrange (len(self.featured_state_indexes)):
                    features_vec[0, count]=self.ML_features[-1]['state_features_values'][t]
                    count += 1
                for t in xrange (num_indexes-1):
                    features_vec[0, count]=self.ML_features[-1]['states_correlation'][t]
                    count += 1
               
                
                #### estimate the local radii
                #learned_radius= self.logistic_regression_model.predict(features_vec)  
                #learned_radius= np.round(self.linear_regression_model.predict(features_vec))
                #learned_radius= np.round(self.NN_regression_model.predict(features_vec))
                #learned_radius= self.GNB_regression_model.predict(features_vec)
                #learned_radius= np.round(self.GPR_regression_model.predict(features_vec))
                #learned_radius= np.round(self.SVM_regression_model.predict(features_vec))
                #learned_radius= np.round(self.SGD_regression_model.predict(features_vec))
                #learned_radius = np.round(self.GradientBoosting_regression_model.predict(features_vec))
                #while learned_radius not in self.loc_radii_pool:
                #learned_radius= np.round(self.PassiveAgressive_regression_model.predict(features_vec))
                ##### These are with for multi output:
                learned_radius=np.round(self.NearestNeighbors_regression_model.predict(features_vec))
                # learned_radius= np.round(self.forest_regression_model.predict(features_vec))
                #learned_radius= np.round(self.regr_multirf.predict (features_vec))
                #learned_radius= np.round(self.lasso_regression_model.predict(features_vec)) 
                self.estimatedRadii.append(learned_radius) 
                
                ############################DATASET TO ADD ############################################
                #### dataset to add is used during the online learning for incremental learning
                #####
                self.ds_toAdd[self.count_tune][0]=self.ML_features[-1]['mean_ensemble_forecast_mean']
                self.ds_toAdd[self.count_tune][1]=self.ML_features[-1]['max_ensemble_forecast_mean']
                self.ds_toAdd[self.count_tune][2]=self.ML_features[-1]['min_ensemble_forecast_mean']
                self.ds_toAdd[self.count_tune][3]=self.ML_features[-1]['sum_ensemble_forecast_mean']
                self.ds_toAdd[self.count_tune][4]=self.ML_features[-1]['forecast_observ_RMSE']
                self.ds_toAdd[self.count_tune][5]=self.ML_features[-1]['forecast_RMSE']
                count = 6
                for t in xrange (len(self.featured_state_indexes)):
                    self.ds_toAdd[self.count_tune][count]=self.ML_features[-1]['state_features_values'][t]
                    count += 1
                for t in xrange (num_indexes-1):
                    self.ds_toAdd[self.count_tune][count]=self.ML_features[-1]['states_correlation'][t]
                    count +=1
                '''
                if np.isscalar(learned_radius):
                    self.ds_toAdd[self.count_tune][count] = learned_radius
                else:
                    self.ds_toAdd[self.count_tune, count:count+state_size] = learned_radius
                '''
                ####
                if self.count_tune <= self.tune_value-1 :
                    self.count_tune=self.count_tune+1  
                ############################DATASET TO ADD ############################################    
                    
                ########### First obtain the best in the testing phase
                
                loc_radii_pool = list(self.loc_radii_pool)  
                winner_radius = np.infty
                winner_analysis_rmse = np.infty
                winner_observ=np.infty
                winner_objective=np.infty
                analysis_rmse_dict = {'radii':[], 'analysis_rmse':[], 'analysis_mean':[]}
                winner_distance=np.infty
                distance_dict = {'radii':[], 'distance':[], 'analysis_rmse':[],'analysis_mean':[] }
                objective_dict= {'radii':[], 'distance':[], 'analysis_rmse':[],'analysis_mean':[],'combination_objective':[], 'observ_rmse':[]}
                W=np.zeros((3));
                W[0]=1
                W[1]=0
                W[2]=0
                
                #
                
                if self._same_loc_radius:
                    for loc_radius_ind in xrange(len(loc_radii_pool)):
                        #
                        loc_radius = loc_radii_pool[loc_radius_ind]
                        
                        #
                        # Retrieve the state of the random number generator:
                        if self._random_seed is not None:
                            local_random_state = np.random.get_state()
                            np.random.seed(self._random_seed)
                        
                        if self._same_loc_radius:
                            print("TESTING PHASE: Trying analysis with localization radius = %4.2f >> " % loc_radius),
                        else:
                            print("TESTING PHASE: Trying analysis with random localization radii from: %s >> " % str(loc_radius)),
                            
                        filter_obj.filter_configs['localization_radius'] = loc_radius
                        filter_obj.analysis()
                        #
                        try:
                            analysis_state = filter_obj.filter_configs['analysis_state']
                        except (NameError, AttributeError):
                            raise NameError("analysis_state must be updated by the filter "
                                            "and added to 'filter_obj.filter_configs'!")
                        tmp_anal_rmse = utility.calculate_rmse(up_reference_state, analysis_state, state_size)
                        print("Analysis RMSE = %f" % tmp_anal_rmse)
                        #
                        ################# From here is modified for combination measure ##################################
                        
                        analysis_ensemble=copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        ensemble_size = len(forecast_ensemble)
                        analysis_ensemble_np = np.empty(shape=(state_size, ensemble_size))
                        for col_ind in xrange (ensemble_size):
                        	analysis_ensemble_np[:, col_ind] = analysis_ensemble[col_ind].get_numpy_array()
                        	
                        #
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [08:24 pm, July 16, 2017]
                        # -------------------------------------------------------
                        # Some models, such as the current QG-1.5 implementation in DATeS, force Dirichlet boundary conditions with value 0;
                        # In such case, the truth, and the analysis ensemble have the same value. In this case, 
                        # we are at the mercy of how Numpy (or alternative packages) rank equal elements, and also will be dependent on how rank_hist function is implemented!
                        # We can modify first_var, var_skp, and las_var to take care of it, but
                        #
                        # A better approach, is as follows: These models (mentioned above) should provide the indexes (as required by DATeS), in the model_configs dictionary.
                        # I have modified the rank_hist function to handle such cases. Now it works fine.
                        # -------------------------------------------------------
                        ranks_freq, ranks_rel_freq, bins_bounds, fig_hist = utility.rank_hist(analysis_ensemble_np, 
                                                                                              up_reference_state.get_numpy_array(), 
                                                                                              first_var=0, 
                                                                                              last_var=None, 
                                                                                              var_skp=1,
                                                                                              #draw_hist=True,  # you can turn this off, after checking the plots, and the fitted pdf plots
                                                                                              hist_type='relfreq', 
                                                                                              first_time_ind=0, 
                                                                                              last_time_ind=None,
                                                                                              time_ind_skp=1,
                                                                                              ignore_indexes=model_boundary_indexes) 
                        
                        # -------------------------------------------------------
                        data = []
                        for fr, bn in zip(ranks_freq, bins_bounds):
                            data += [float(bn)]*fr
                        data = np.asarray(data)
                        #
                        dist = scipy.stats.beta
                        params = dist.fit(data)
                        # -------------------------------------------------------
                        # Fit a beta distribution:
                        betas = np.array([params[0], params[1]])
                        B0 = betas.sum()                        
                        pdf_x = np.linspace(dist.ppf(0.01, betas[0], betas[1]), dist.ppf(0.99, params[0], params[1]), 50)
                        pdf_y = dist.pdf(pdf_x, *params[:-2], loc=params[-2], scale=params[-1])
                        if fig_hist is not None:
                            if False:
                                if pdf_y.min() < ranks_rel_freq.min():
                                    pdf_y += abs(pdf_y.min()-ranks_rel_freq.min())
                                else:
                                    pdf_y -= abs(pdf_y.min()-ranks_rel_freq.min())
                            
                            a, b = pdf_x.min(), pdf_x.max()
                            c, d = 0, ensemble_size
                            pdf_x = c + ((d-c)/(b-a)) * (pdf_x-a)
                            ax = fig_hist.gca()
                            ax.plot(pdf_x, pdf_y, 'r-', linewidth=5, label='fitted beta', zorder=1)
                            ylim = ax.get_ylim()
                            ax.set_ylim([ylim[0], max(ylim[-1], max(pdf_y.max(), ranks_rel_freq.max()))]) 
                            #
                            #plt.show()
                            #
                        
                        alphas = np.array([1,1])
                        A0 = alphas.sum()
                        KL_distance = sp.gammaln(A0) - np.sum(sp.gammaln(alphas)) \
                                      - sp.gammaln(B0) + np.sum(sp.gammaln(betas)) \
                                      + np.sum((alphas-betas) * (sp.digamma(alphas)-sp.digamma(A0)))
                        
                        tmp_distance=KL_distance
                        mu = np.mean(ranks_rel_freq)
                        var = np.var(ranks_rel_freq) 
                        print('the kl distance is ',KL_distance )
                        #print('the hellinger distance is ', HL_distance)
                        #print('the entropy distance is ',Ent_distance )
                        mu=np.sum(ranks_rel_freq*bins_bounds)/bins_bounds[-1]
                        var=np.float((np.sum(ranks_rel_freq*np.square(bins_bounds-mu)))/(bins_bounds[-1]-1))
                        #print ("the variance is",var )
                        observation = filter_obj.filter_configs['observation']  # this is y
                        #### I changed the following statement to the analysis state instead of forecast state
                        theoretical_observation = model.evaluate_theoretical_observation(analysis_state)  # this is H(x), 
                        tmp_observ_rmse=utility.calculate_rmse(observation,theoretical_observation)
                        print('tmp_observ_rmse is', tmp_observ_rmse)
                        print('tmp_anal_rmse is', tmp_anal_rmse)
                        tmp_objective=	W[0]*tmp_anal_rmse+W[1]*tmp_distance+W[2]*tmp_observ_rmse
                        print "{{{{{{{{{{{{{{{ tmp_objective", tmp_objective, " }}}}}}}}}}}}}}}}}"
                        if tmp_objective < winner_objective:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_observ=tmp_observ_rmse
                        	winner_objective=tmp_objective
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        objective_dict['radii'].append(loc_radius)
                        objective_dict['distance'].append(KL_distance)
                        objective_dict['analysis_rmse'].append(tmp_anal_rmse)
                        objective_dict['observ_rmse'].append(winner_observ)
                        objective_dict['analysis_mean'].append(analysis_state)
                        objective_dict['combination_objective'].append(tmp_objective)
                       	'''       
                        if tmp_anal_rmse < winner_analysis_rmse:
                            winner_analysis_rmse = tmp_anal_rmse
                            winner_radius = loc_radius
                            winner_state = analysis_state.copy()
                            winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        #
                        analysis_rmse_dict['radii'].append(loc_radius)
                        analysis_rmse_dict['analysis_rmse'].append(tmp_anal_rmse)
                        analysis_rmse_dict['analysis_mean'].append(analysis_state)
                        
                        if tmp_distance < winner_distance:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        distance_dict['radii'].append(loc_radius)
                        distance_dict['distance'].append(KL_distance)
                        distance_dict['analysis_rmse'].append(tmp_anal_rmse)
                        distance_dict['analysis_mean'].append(analysis_state)
                        '''
                    	################# up to here is modified for combination measure ##################################
                        
                        
                        # Reset the state of the random number generator:
                        if self._random_seed is not None:
                            np.random.set_state(local_random_state)
                            
                else:
                    num_training_trials = self._num_training_trials
                    for trial_ind in xrange(num_training_trials):
                        #
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [10:16 pm, July 15, 2017]
                        # -------------------------------------------------------
                        # loc_radius is a vector of the same size as the model grid that is used to localize covariances of state-vector entris.
                        # observation_size = model.observation_vector_size()
                        # loc_radius = [loc_radii_pool[i] for i in [np.random.randint(len(loc_radii_pool)) for d in xrange(observation_size)]]
                        # -------------------------------------------------------
                        loc_radius = [loc_radii_pool[i] for i in [np.random.randint(len(loc_radii_pool)) for d in xrange(state_size)]]
                        loc_radius = np.asarray(loc_radius)
                        # -------------------------------------------------------
                         
                         # REMOVE >> loc_radius = loc_radii_pool  # Choose random radius for each grid point in the observation space

                        #
                        # Retrieve the state of the random number generator:
                        if self._random_seed is not None:
                            local_random_state = np.random.get_state()
                            np.random.seed(self._random_seed)
                        
                        if self._same_loc_radius:
                            print("TESTING PHASE: Trying analysis with localization radius = %4.2f: " % loc_radius)
                        else:
                            print("TESTING PHASE: Trying analysis with random localization radii from: %s: " % str(loc_radius))
                            
                        filter_obj.filter_configs['localization_radius'] = loc_radius
                        filter_obj.analysis()
                        #
                        try:
                            analysis_state = filter_obj.filter_configs['analysis_state']
                        except (NameError, AttributeError):
                            raise NameError("analysis_state must be updated by the filter "
                                            "and added to 'filter_obj.filter_configs'!")
                        tmp_anal_rmse = utility.calculate_rmse(up_reference_state, analysis_state, state_size)
                        # print("Analysis RMSE = %f" % tmp_anal_rmse)
                        if np.isnan(tmp_anal_rmse):
                            print "WARNING: Trial", trial_ind, "out of ", num_training_trials-1, "DIVERGED. Continuing next trial..."
                            continue
                        #
                        ################# From here is modified for combination measure ##################################
                        
                        analysis_ensemble=copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        ensemble_size = len(forecast_ensemble)
                        analysis_ensemble_np = np.empty(shape=(state_size, ensemble_size))
                        for col_ind in xrange (ensemble_size):
                        	analysis_ensemble_np[:, col_ind] = analysis_ensemble[col_ind].get_numpy_array()
                        	
                        #
                        # -------------------------------------------------------
                        # Ahmed Remark: Time Stamp [08:24 pm, July 16, 2017]
                        # -------------------------------------------------------
                        # Some models, such as the current QG-1.5 implementation in DATeS, force Dirichlet boundary conditions with value 0;
                        # In such case, the truth, and the analysis ensemble have the same value. In this case, 
                        # we are at the mercy of how Numpy (or alternative packages) rank equal elements, and also will be dependent on how rank_hist function is implemented!
                        # We can modify first_var, var_skp, and las_var to take care of it, but
                        #
                        # A better approach, is as follows: These models (mentioned above) should provide the indexes (as required by DATeS), in the model_configs dictionary.
                        # I have modified the rank_hist function to handle such cases. Now it works fine.
                        # -------------------------------------------------------
                        ranks_freq, ranks_rel_freq, bins_bounds, fig_hist = utility.rank_hist(analysis_ensemble_np, 
                                                                                              up_reference_state.get_numpy_array(), 
                                                                                              first_var=0, 
                                                                                              last_var=None, 
                                                                                              var_skp=1,
                                                                                              #draw_hist=True,  # you can turn this off, after checking the plots, and the fitted pdf plots
                                                                                              hist_type='relfreq', 
                                                                                              first_time_ind=0, 
                                                                                              last_time_ind=None,
                                                                                              time_ind_skp=1,
                                                                                              ignore_indexes=model_boundary_indexes) 
                        
                        # -------------------------------------------------------
                         
                        data = []
                        for fr, bn in zip(ranks_freq, bins_bounds):
                            data += [float(bn)]*fr
                        data = np.asarray(data)
                        #
                        dist = scipy.stats.beta
                        params = dist.fit(data)
                        # -------------------------------------------------------
                        # Fit a beta distribution:
                        betas = np.array([params[0], params[1]])
                        B0 = betas.sum()                        
                        pdf_x = np.linspace(dist.ppf(0.01, betas[0], betas[1]), dist.ppf(0.99, params[0], params[1]), 50)
                        pdf_y = dist.pdf(pdf_x, *params[:-2], loc=params[-2], scale=params[-1])
                        if fig_hist is not None:
                            if False:
                                if pdf_y.min() < ranks_rel_freq.min():
                                    pdf_y += abs(pdf_y.min()-ranks_rel_freq.min())
                                else:
                                    pdf_y -= abs(pdf_y.min()-ranks_rel_freq.min())
                            
                            a, b = pdf_x.min(), pdf_x.max()
                            c, d = 0, ensemble_size
                            pdf_x = c + ((d-c)/(b-a)) * (pdf_x-a)
                            ax = fig_hist.gca()
                            ax.plot(pdf_x, pdf_y, 'r-', linewidth=5, label='fitted beta', zorder=1)
                            ylim = ax.get_ylim()
                            ax.set_ylim([ylim[0], max(ylim[-1], max(pdf_y.max(), ranks_rel_freq.max()))]) 
                            #
                            #plt.show()
                           
                        
                        alphas = np.array([1,1])
                        A0 = alphas.sum()
                        KL_distance = sp.gammaln(A0) - np.sum(sp.gammaln(alphas)) \
                                      - sp.gammaln(B0) + np.sum(sp.gammaln(betas)) \
                                      + np.sum((alphas-betas) * (sp.digamma(alphas)-sp.digamma(A0)))
                        
                        tmp_distance=KL_distance 
                        mu = np.mean(ranks_rel_freq)
                        var = np.var(ranks_rel_freq) 
                        print('the kl distance is ',KL_distance )
                        #print('the hellinger distance is ', HL_distance)
                        #print('the entropy distance is ',Ent_distance )
                        mu=np.sum(ranks_rel_freq*bins_bounds)/bins_bounds[-1]
                        var=np.float((np.sum(ranks_rel_freq*np.square(bins_bounds-mu)))/(bins_bounds[-1]-1))
                        #print ("the variance is",var )
                        observation = filter_obj.filter_configs['observation']  # this is y
                        #### I changed the following statement to the analysis state instead of forecast state
                        theoretical_observation = model.evaluate_theoretical_observation(analysis_state)  # this is H(x), 
                        tmp_observ_rmse=utility.calculate_rmse(observation,theoretical_observation)
                        print('tmp_observ_rmse is', tmp_observ_rmse)
                        print('tmp_anal_rmse is', tmp_anal_rmse)
                        tmp_objective=	W[0]*tmp_anal_rmse+W[1]*tmp_distance+W[2]*tmp_observ_rmse
                        print "{{{{{{{{{{{{{{{ tmp_objective", tmp_objective, " }}}}}}}}}}}}}}}}}"
                        if tmp_objective < winner_objective:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_observ=tmp_observ_rmse
                        	winner_objective=tmp_objective
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        objective_dict['radii'].append(loc_radius)
                        objective_dict['distance'].append(KL_distance)
                        objective_dict['analysis_rmse'].append(tmp_anal_rmse)
                        objective_dict['observ_rmse'].append(winner_observ)
                        objective_dict['analysis_mean'].append(analysis_state)
                        objective_dict['combination_objective'].append(tmp_objective)
                        estimated_analysis_rmse = tmp_anal_rmse
                        
                       	'''
                        if tmp_anal_rmse < winner_analysis_rmse:
                            winner_analysis_rmse = tmp_anal_rmse
                            winner_radius = loc_radius
                            winner_state = analysis_state.copy()
                            winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                        #
                        analysis_rmse_dict['radii'].append(loc_radius)
                        analysis_rmse_dict['analysis_rmse'].append(tmp_anal_rmse)
                        analysis_rmse_dict['analysis_mean'].append(analysis_state)
                        
                        if tmp_distance < winner_distance:
                        	winner_analysis_rmse = tmp_anal_rmse
                        	winner_distance = tmp_distance
                        	winner_radius = loc_radius
                        	winner_state = analysis_state.copy()
                        	winner_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                            
                        #
                        distance_dict['radii'].append(loc_radius)
                        distance_dict['distance'].append(KL_distance)
                        distance_dict['analysis_rmse'].append(tmp_anal_rmse)
                        distance_dict['analysis_mean'].append(analysis_state)
                        '''
                        
                    	################# up to here is modified for combination measure ##################################
                        
                        
                        # Reset the state of the random number generator:
                        if self._random_seed is not None:
                            np.random.set_state(local_random_state)
                            
                try:
                    winner_state
                except(NameError, UnboundLocalError):
                    print "All trials have diverged..."
                    print "Quitting Adaptive localization..."
                    raise ValueError
                
                
                
                
                
                #  Use the learned radius on the filter config
                print("****TESTING PHASE: Analysis with learned localization radius  is ****>> " , learned_radius)
                #filter_obj.filter_configs['localization_radius'] = learned_radius.reshape(len(self.ML_features[0]['localization_radius']))
                filter_obj.analysis()
                

                #
                try:
                    analysis_state = filter_obj.filter_configs['analysis_state']
                except (NameError, AttributeError):
                    raise NameError("analysis_state must be updated by the filter "
                                    "and added to 'filter_obj.filter_configs'!")
                analysis_rmse = utility.calculate_rmse(up_reference_state, analysis_state, state_size)
                estimated_analysis_state = analysis_state
                estimated_state = analysis_state.copy()
                estimated_analysis_ensemble = copy.copy(filter_obj.filter_configs['analysis_ensemble'])
                # assign estimated radius and analysis to the filter object     
                #filter_obj.filter_configs['localization_radius'] = learned_radius.reshape(len(self.ML_features[0]['localization_radius']))
                filter_obj.filter_configs['localization_radius'] = learned_radius
                filter_obj.filter_configs['analysis_state'] = estimated_state
                filter_obj.filter_configs['analysis_ensemble'] = estimated_analysis_ensemble
                #print("estimated_analysis_rmse", estimated_analysis_rmse)
                #print("winner_analysis_rmse", winner_analysis_rmse)
                #self.err_estimationRMSE.append(np.abs(winner_analysis_rmse-estimated_analysis_rmse))
                self.testRadii.append(winner_radius)
                self.err_estimationRadii.append(winner_radius-learned_radius)  # what is th
                
                
        # Apply post-processing if required
        if apply_postprocessing:
            filter_obj.cycle_postprocessing()

        # Update filter statistics (including RMSE)
        if 'filter_statistics' not in filter_obj.output_configs:
            filter_obj.output_configs.update(dict(filter_statistics=dict(initial_rmse=None,forecast_rmse=None, analysis_rmse=None)))
        else:
            if 'analysis_rmse' not in filter_obj.output_configs['filter_statistics']:
                filter_obj.output_configs['filter_statistics'].update(dict(analysis_rmse=None))
            if 'forecast_rmse' not in filter_obj.output_configs['filter_statistics']:
                filter_obj.output_configs['filter_statistics'].update(dict(forecast_rmse=None))
            if 'initial_rmse' not in filter_obj.output_configs['filter_statistics']:
                filter_obj.output_configs['filter_statistics'].update(dict(initial_rmse=None))
		
        # now update the RMSE's
        filter_obj.output_configs['filter_statistics']['initial_rmse'] = initial_rmse
        filter_obj.output_configs['filter_statistics']['forecast_rmse'] = forecast_rmse
        filter_obj.output_configs['filter_statistics']['analysis_rmse'] = analysis_rmse

        # output and save results if requested
        if filter_obj.output_configs['scr_output']:
            filter_obj.print_cycle_results()
        if filter_obj.output_configs['file_output']:
            filter_obj.save_cycle_results()
            

    def printResults(self):
               
        print ("estimatd radii are:")        
        print (self.estimatedRadii)
        print ("the best radii are:")
        print (self.testRadii)
        #print ("the estimation error in radii is")
        #print (self.err_estimationRadii) 
        #print ("The sum of the abs error in radii is ")
        #print (np.sum (np.abs(self.err_estimationRadii)))
        #print ("the estimation error in RMSE is")
        #print (self.err_estimationRMSE)
        #print ("The sum of the error in RMSE is ")
        #print (np.sum (self.err_estimationRMSE))
        #print ("the KL measure is")
        #print (self.klMeasure)
        #print ("the KL measure1 is")
        #print (self.klMeasure1)
        

        
    
    @staticmethod
    
    def initialize_ML_features_dict():
        """
        Construct and return template for features for machine learning algorithm that will be used for each choice of the localization radius
        """
        # The next dictionary is a template for features for machine learning algorithm that will be used for each choice of the localization radius
        ML_Features_Template = {"assim_cycle_number":None,  # 1) the assimilation cycle number
                                "min_ensemble_forecast_variance":None,     # 2) the min of ensemble forecast variance 
                                "max_ensemble_forecast_variance":None,     # 3) the max of ensemble forecast variace
                                "mean_ensemble_forecast_variance":None,    # 4) the mean of ensemble forecast variace
                                "mean_ensemble_forecast_mean":None,        # 5) the mean of ensemble forecast mean
                                "min_ensemble_forecast_mean":None,         # 6) the min of ensemble forecast mean
                                "max_ensemble_forecast_mean":None,         # 7) the max of ensemble forecast mean
                                "sum_ensemble_forecast_mean":None,         # 8) the sum of ensemble forecast mean
                                "sum_ensemble_forecast_mean":None,         # 9) the sum of ensemble forecast mean
                                "state_features_indexes":None,             # 10) indexes of the state taken as features
                                "state_features_values":None,              # 11) selected values of the state taken as features
                                "forecast_RMSE":None,                      # 12) RMSE of the forecast mean
                                "analysis_RMSE_dict":None,                 # 13) RMSE of the analysis mean with different radii
                                "forecast_observ_RMSE":None,               # 14) RMSE between observation (y), and model forecasted observation H(xf)
                                "state_indexes_correlation":None,          # 15) indexes of the state for correlation purpose
                                "states_correlation":None,                 # 16) the correlation between states of the system
                                "forecast_ensemble":None,                  # 17) the whole forecast ensemble
                                "localization_radius":None                 # i ) This is the localization radius winning in previous cycle
                                }
        return ML_Features_Template
       


