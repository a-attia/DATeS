
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


"""
    This python script generates the python wrapper for the QG step function based on the model grid size.
    This script is just a learning-step script and will be optimized soon!
"""


# 
# 
import os
import shutil

wrapper_signature_suffix = 'QG_wrapper'
wrapper_signature_filename = wrapper_signature_suffix+'.pyf'
extension_module_filename = wrapper_signature_suffix+'.so'

FC = 'gfortran'
ICDIRs = ['/usr/local/include/', '/usr/local/bin/']
SRCFILES = ['utils.f90',
            'parameters.f90',
            'helmholtz.f90',
            'calc.f90',
            'qgflux.f90',
            'qgstep.f90',
            'qgstep_wrap.f90']

tmp_outfile = 'f2py_outfile.tmp'
_parameters_src_filename = 'parameters.f90'
_MREFIN_filename = 'MREFIN_value.dat'

def create_wrapper(in_MREFIN=7):
    """
    Create wrapper function of the QG-model code given the model grid size.
    For now (following Sakov's code) we assume the grid is square.
    """
    nx = 2 * pow(2, (in_MREFIN-1)) + 1
    ny = nx
    print('%s\n\t>>>>>> GQ-Model 1.5. <<<<<<\n%s\n\t+ Model grid-size: %d x %d' % ('-'*50, '-'*50, nx, ny))
    print('\t+ Started creating wrapper function...')

    # This dirctory retrieval step should be optimized. I am thinking of a general way to organize models and path
    # their paths and information as a dictionary!
    cwd = os.getcwd()
    try:
        dates_root_path = os.environ['DATES_ROOT_PATH']
    except:
        dates_root_path = '/home/attia/dates/'  # of course this is temporary!
    src_dir = os.path.join(dates_root_path, 'src/Models_Forest/2D/QG_1p5/f90/')
    os.chdir(src_dir)
    if not os.path.isfile(os.path.join(src_dir, wrapper_signature_filename)):
        raise IOError('signature file is missing!')
    else:
        # backup existing extension module if exists:
        old_filename = os.path.join(src_dir, extension_module_filename)
        if os.path.isfile(old_filename):
            backup_filename = os.path.join(src_dir, extension_module_filename+'.backup')
            shutil.move(old_filename, backup_filename)
            #
        # update the parameters file in the f90 code. This is dangerous: we may consider adding one module to read the
        # grid size from a temporary file!
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~_MREFIN_filename
        with open(os.path.join(src_dir, _parameters_src_filename), 'r') as parameters_file:
            parameters_src = parameters_file.readlines()

        for ln_ind in xrange(len(parameters_src)):
            ln = parameters_src[ln_ind]
            if ln.strip().startswith('integer, parameter, public :: MREFIN ='):
                split_ln = (ln.strip('\n')).split('=')
                if split_ln[-1] != str(in_MREFIN):
                    ln = split_ln[0] + '= ' + str(in_MREFIN) + ' \n'
                    parameters_src[ln_ind] = ln
                    update_parameter_file = True
                else:
                    update_parameter_file = False
                break

        if update_parameter_file:
            with open(os.path.join(src_dir, _parameters_src_filename), 'w') as parameters_file:
                parameters_file.writelines(parameters_src)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # The following should be ignored. Fortran modules can't read from files in the declaration part, in general
        # it cannot contain executable statements...
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # with open(os.path.join(src_dir, _parameters_src_filename), 'r') as parameters_file:
        #     parameters_src = parameters_file.readlines()
        #
        # for ln_ind in xrange(len(parameters_src)):
        #     ln = parameters_src[ln_ind]
        #     if ln.strip().startswith('character(STRLEN) :: MREFIN_filename = '):
        #         split_ln = (ln.strip('\n')).split("=")
        #         tmp_filename = str(split_ln[-1])
        #         if tmp_filename != _MREFIN_filename:
        #             ln = split_ln[0] + '= ' + _MREFIN_filename + ' \n'
        #         parameters_src[ln_ind] = ln
        #         # update the MREF_filename in the parameters.f90 file if necessary. We shall remove this check once
        #         # we decide that this approach is good enough!
        #         with open(os.path.join(src_dir, _parameters_src_filename), 'w') as parameters_file:
        #             parameters_file.writelines(parameters_src)
        #
        # # Now we are good, updte the value of MREFIN if necessary
        # with open(os.path.join(src_dir, _MREFIN_filename), 'r+') as parameter_val_file:
        #     parameter_val = str.strip(parameter_val_file.read(), "\n ")
        #     if parameter_val != str(in_MREFIN):
        #         parameter_val_file.write(str)
        # with open(os.path.join(src_dir, _parameters_src_filename), 'w') as parameters_file:
        #     parameters_file.writelines(parameters_src)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Parameters are set. Run the f2py wrapping command
        run_f2py_compiler_cmd()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # add a check to make sure the extension module is created...
        if not os.path.isfile(os.path.join(src_dir, extension_module_filename)):
            raise IOError("Failed to wrap the model!")
        else:
            print('%s\n\tWrapper created successfully!\n%s' % (''.join(['-']*50), ''.join(['-']*50)))
        if os.path.isfile(tmp_outfile):
            os.remove(tmp_outfile)
        os.chdir(cwd)


def run_f2py_compiler_cmd():
    """
    Run the f2py compiler command. create the extension module to be imported by python model driver
    """
    os.system("f2py -c --fcompiler=%s --quiet %s %s %s > %s" % (FC, wrapper_signature_filename, ' '.join(SRCFILES),
                                                          ' '.join(['-I' + inc_path for inc_path in ICDIRs]),
                                                                tmp_outfile))
