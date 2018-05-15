
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


from time_integration_base import TimeIntegratorBase


class ExplicitRungeKutta(TimeIntegratorBase):

    def __init__(self, options_dict=None):
        TimeIntegratorBase.__init__(self, options_dict)

    
    def sub_iterate(self, local_t0, local_tf):
        """
        Integrates across a single subinterval of the time integration window.

        Input:
            local_t0: beginning time of local integration window.
            local_tf: end time of local integration window.

        Output:
            soln_state: solution at local_tf
        """
        # retrieve coefficients:
        _a = self._coeffs['a']
        _b = self._coeffs['b']
        _c = self._coeffs['c']
        _s = self._coeffs['s']
        
        current_state = self._initial_state.copy()
        step_size = self._step_size
        current_time = local_t0
        # Initialize k variable.
        k = []
        for stage in xrange(0, _s):
            k.append(self._initial_state.copy())

        # Run main time integration loop.
        while abs(current_time - local_tf) > 1e-8:

            # Verify that current step_size does not lead to a time > than local_tf
            if abs(current_time + step_size - local_tf) >  1e-8:
                step_size = local_tf - current_time

            # Construct stage values.
            for stage in xrange(0, _s):
                insum = current_state.copy()
                if stage != 0:
                    for j in xrange(0, stage):
                        if _a[stage][j] != 0:
                            insum = insum.axpy(step_size*_a[stage][j], k[j])
                            # argument.
                k[stage] = self._model.step_forward_function(current_time + self._step_size*_c[stage], insum)

            # Compute new solution
            for stage in xrange(0, _s):
                if _b[stage] != 0:
                    current_state = current_state.axpy(step_size*_b[stage], k[stage])

            # Update current time.
            current_time += step_size
        return current_state

    def set_coeffs(self):
        """
        set integration method coefficients
        """
        if self._int_method == 0 or self._method == 'RK4':
            self._coeffs = {'a': [[],[1.0/2.0], [0, 1.0/2.0], [0, 0, 1.0]],  # why the first one is empty?
                            'b': [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0],
                            'c': [0, 1.0/2.0, 1.0/2.0, 1.0],
                            's': 4
                            }
        else:
            raise Exception('Invalid method specified')

