
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
    This python script generates the python wrapper for the FATODE-ERK functions.
    like qg_wrapper generator, this script is a learning-step script and will be optimized!
"""


#
import os
import shutil
import dates_utility as utility

wrapper_signature_suffix = 'erkfwd'
wrapper_signature_filename = wrapper_signature_suffix+'.pyf'
extension_module_filename = wrapper_signature_suffix+'.so'

FC = 'gfortran'  # we can use 'dpkg --list | grep compiler' to get a list of compilers later [Ubuntu/Debian]. For Yum-based, we can use 'yum search all compiler'.

FWD_MODULE = 'erk_f90_integrator'
ICDIRS = ['/usr/local/include/', '/usr/local/bin/']
SRCFILES = ['ERK_f90_Integrator.F90']
FLIBS = ['lblas', 'llapack']  # Fortran librarires to be linked


tmp_outfile = 'f2py_outfile.tmp'

def create_wrapper(verbose=False):
    """
    Create wrapper function of the FATODE ERK (forward and adjoint) code.
    """
    if verbose:
        print("%s\n\t>>>>>> installing FATODE-ERK. <<<<<<\n%s\n" % ('-'*50, '-'*50))
    # This dirctory retrieval step should be optimized. I am thinking of a general way to organize models and path
    # their paths and information as a dictionary!
    cwd = os.getcwd()
    try:
        dates_root_path = os.environ['DATES_ROOT_PATH']
    except:
        print("DATeS is not properly initialized;")
        print("failed to retrieve the environment variable 'DATES_ROOT_PATH' ")
        print("You need to call dates_setup.initialize() at the beginning of your driver!")
        raise ValueError

    src_dir = os.path.join(dates_root_path, 'src/Time_Integration/FATODE/FATODE/FWD/ERK/')
    if not os.path.isdir(src_dir):
        print("Failed to access the FATODE ERK directory:")
        print(src_dir)
        raise IOError

    os.chdir(src_dir)

    # check if the wrapper exists, and is valid:
    create_module = True
    if os.path.isfile(os.path.join(src_dir, extension_module_filename)):
        create_module = False
        # extension module exists, test if valid.
        try:
            exec("from %s import %s" % (wrapper_signature_suffix, FWD_MODULE))
            exec("del"+ FWD_MODULE)
        except(NameError, ImportError, SyntaxError, AttributeError):
            create_module = True
        else:
            print("Unexpected Error occured. Failed to import the extension module; the wrapper will be recreated.")
            raise

    # Re/Create the extension module if necessary
    if create_module:
        if not os.path.isfile(os.path.join(src_dir, wrapper_signature_filename)):
            print("signature file is missing!")
            raise IOError
            #
        else:
            # backup existing extension module if exists:
            old_filename = os.path.join(src_dir, extension_module_filename)
            if os.path.isfile(old_filename):
                backup_filename = os.path.join(src_dir, extension_module_filename+'.backup')
                shutil.move(old_filename, backup_filename)
                #

            # Parameters are set. Run the f2py wrapping command
            if verbose:
                print("\t+ Started creating wrapper function...")
            run_f2py_compiler_cmd(verbose=verbose)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

            # add a check to make sure the extension module is created...
            if not os.path.isfile(os.path.join(src_dir, extension_module_filename)):
                print("Couldn't load: %s" %( os.path.join(src_dir, extension_module_filename)))
                print("Failed to wrap the FWD integrator! For info Check the file %s" %os.path.isfile(os.path.join(src_dir, tmp_outfile)))
                raise IOError
            else:
                if verbose:
                    print('%s\n\tWrapper created successfully!\n%s' % (''.join(['-']*50), ''.join(['-']*50)))
                #
                if os.path.isfile(tmp_outfile):
                    os.remove(tmp_outfile)
                os.chdir(cwd)
    else:
        pass


def run_f2py_compiler_cmd(verbose=False):
    """
    Run the f2py compiler command. create the extension module to be imported by python model driver.
    We assume gfortran exists. Also, we assume -lblas -lgfortran -llapack are available.
    If these libraries are not installed, TRY sudo [apt-get/yum] install libblas-dev liblapack-dev
    we can use 'dpkg --list | grep compiler' to get a list of compilers later [Ubuntu/Debian]. For Yum-based, we can use 'yum search all compiler'.
    """
    wrapping_cmd = "f2py --fcompiler=%s -c %s %s %s %s > %s" % (FC,
                                                                wrapper_signature_filename,
                                                                ' '.join(['-I' + inc_path for inc_path in ICDIRS]),
                                                                ' '.join(['-' + lib for lib in FLIBS]),
                                                                ' '.join(SRCFILES),
                                                                tmp_outfile)

    if verbose:
        print("executing the command:\n %s" % wrapping_cmd)

    res = os.system(wrapping_cmd)

    if res != 0:
        print "Executing wrapper command failed with return code: ", res
    else:
        pass

    # cleanup dSYM directory (for Mac OS) if generated:
    cwd = os.getcwd()
    dirs = utility.get_list_of_subdirectories(cwd)
    for d in dirs:
        if 'dSYM' in d:
            shutil.rmtree(d, ignore_errors=True)
