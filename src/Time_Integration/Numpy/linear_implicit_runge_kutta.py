
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


class LIRK(TimeIntegratorBase):
    """
    Class for Linearly Implcit Runge Kutta time stepping methods. currently Implemented :
    Second Order L-stable Rosenbrock method with fixed time steeping ref:http://dx.doi.org/10.1137/S1064827597326651
    """
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
        _gamma = self._coeffs['gamma']

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

            jacobian = self._model.step_forward_function_Jacobian(current_time, current_state)
            scaled_jacobian = jacobian.scale(-step_size * _gamma[0],in_place=False)
            lhs=scaled_jacobian.addAlphaI(1,in_place=False)  # LHS = (I- hJ*gamma)

            for stage in xrange(0, _s):
                    insum = current_state.copy()
                    outsum = (current_state.copy()).scale(0.0)
                    for j in xrange(0, stage):
                          insum = insum.axpy(_a[stage][j], k[j])
                          outsum = outsum.axpy(_gamma[stage][j], k[j])
                    Fi = self._model.step_forward_function(current_time+_c[stage]*step_size, insum)
                    rhs = ((jacobian.vector_product(outsum)).add(Fi,in_place=False)).scale(step_size)
                    k[stage] = lhs.solve(rhs)

            for stage in xrange(0, _s):
                current_state = current_state.axpy(_b[stage], k[stage])
            current_time = current_time + step_size   # whats the point of this?
        return current_state

    def set_coeffs(self):
        if self._int_method is 0 or self._method is 'ROS2':
            self._coeffs={ 'a': [[], [1.0]],
                          'b': [0.5, 0.5],
                          'c': [0.0 ,1.0], # fix for autonomous case
                          'gamma' : [1.707106781186, [-2.414213562373]],
                          's': 2
                         }
        else:
            raise Exception('Invalid method specified')
