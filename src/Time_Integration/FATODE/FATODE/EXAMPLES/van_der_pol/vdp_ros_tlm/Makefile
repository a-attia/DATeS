############################################################################
#
#  Module:          Makefile
#
#  Purpose:         Top-level Makefile
#
#  Creation date:   Feb 27, 2011
#
#  Modified:
#
#  Send bug reports, comments or suggestions to zhang@vt.edu
#
############################################################################


export ARCH=ar -cr
export RANLIB=ranlib

# Configuration flags for linear solvers. Must choose one from the following three options.
# #  -DFULL_ALGEBRA            use BLAS and LAPACK library (full algebra)
# #  -DSPARSE_UMF              use UMFpack (sparse format) 
# #  -DSPARSE_LU               use SuperLU (sparse format)
#

LS_CONFIG = -DFULL_ALGEBRA

#~~~> gfortran (GNU FORTRAN Compiler)
export FC = gfortran
export FFLAGS = -cpp -O3 -pg -ffree-line-length-none $(LS_CONFIG)
BLAS= ../../../GFORTRAN_LIBS/libblas.a
LAPACK= ../../../GFORTRAN_LIBS/liblapack.a


FATDIR     = ../../..
LIBDIR     = $(FATDIR)/LSS_LIBS/x86_64_Linux
MODEL      = TLM
FAMILY     = ROS_TLM

APP        = van_der_pol
PAR        = 
LSSOLVER   = LS_Solver.o
INTEGRATOR = $(FAMILY)_f90_Integrator.o

default: driver

# swe2D_upwind.o: swe2D_upwind.F90
#	$(FC) $(FFLAGS) -c $<

LS_Solver.o: $(FATDIR)/$(MODEL)/$(FAMILY)/LS_Solver.F90
	$(FC) $(FFLAGS) -c $<

$(FAMILY)_f90_Integrator.o: $(FATDIR)/$(MODEL)/$(FAMILY)/$(FAMILY)_f90_Integrator.F90
	$(FC) $(FFLAGS) -c $<

vdp_ros_tlm_dr.o: vdp_ros_tlm_dr.f90 $(PAR) $(LSSOLVER) $(INTEGRATOR)
	$(FC) $(FFLAGS) -c $<



LIB = -lm

LUPACK = $(LIBDIR)/SUPERLU/libsuperlu_4.2.a
LUWRAP = $(LIBDIR)/SUPERLU/c_fortran_dgssv.o $(LIBDIR)/SUPERLU/c_fortran_zgssv.o

UMFPACK = $(LIBDIR)/UMFPACK/libumfpack.a $(LIBDIR)/UMFPACK/libamd.a
UMFWRAP = $(LIBDIR)/UMFPACK/umf4_f77wrapper.o $(LIBDIR)/UMFPACK/umf4_f77zwrapper.o

default: driver

driver: vdp_ros_tlm_dr.o $(PAR) $(LSSOLVER) $(INTEGRATOR)
	$(FC) $(FFLAGS) -o $(APP) $< $(LUWRAP) $(LUPACK) $(UMFWRAP) $(UMFPACK) $(INTEGRATOR) $(LSSOLVER) $(PAR) $(LAPACK) $(BLAS) $(XERBLA) $(LIB)

purge:  clean

clean:
	rm -f *~ *.mod *.o

help:
	@$(ECHO) "usage: make ?"

