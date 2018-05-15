###########################################################################
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
LS_CONFIG  =  -DFULL_ALGEBRA

#~~~> NAG fortran
#export FC=nagfor
#export FFLAGS = -O3 $(LS_CONFIG)
#BLAS = ~/nagfor_libs/libblas.a
#LAPACK = ~/nagfor_libs/liblapack.a

#~~~> absoft fortran
#export FC=f90
#export FFLAGS = -O3 $(LS_CONFIG) 
#BLAS = /Applications/Absoft10.1/lib/libblas.a
#LAPACK = /Applications/Absoft10.1/lib/liblapack.a

#~~~> intel fortran
#export FC=ifort
#export FFLAGS= -cpp -O3 -nogen-interface $(LS_CONFIG) -warn all
#BLAS = /opt/ifort_libs/libblas.a
#LAPACK = /opt/ifort_libs/liblapack.a


#~~~> sunf90
#export FC = /opt/oracle/solstudio12.2/bin/sunf90 
#export FFLAGS = -fpp -O3 -free $(LS_CONFIG)
#BLAS =  -xlic_lib=sunperf
#LAPACK =


#~~~> gfortran (GNU FORTRAN Compiler)
export FC = gfortran
export FFLAGS = -cpp -O3 -pg -ffree-line-length-none $(LS_CONFIG)
BLAS = ../../../GFORTRAN_LIBS/libblas.a
LAPACK = ../../../GFORTRAN_LIBS/liblapack.a


FATDIR     = ../../..
LIBDIR     = $(FATDIR)/LSS_LIBS/x86_64_Linux
MODEL      = FWD
FAMILY     = ROS

APP        = van_der_pol
PAR        = 
LSSOLVER   = LS_Solver.o
INTEGRATOR = $(FAMILY)_f90_Integrator.o

default: driver

LS_Solver.o: $(FATDIR)/$(MODEL)/$(FAMILY)/LS_Solver.F90
	$(FC) $(FFLAGS) -c $<

$(FAMILY)_f90_Integrator.o: $(FATDIR)/$(MODEL)/$(FAMILY)/$(FAMILY)_f90_Integrator.F90
	$(FC) $(FFLAGS) -c $<

vdp_ros_dr.o: vdp_ros_dr.F90 $(PAR) $(LSSOLVER) $(INTEGRATOR)
	$(FC) $(FFLAGS) -c $<


LIB = -lm

LUPACK = $(LIBDIR)/SUPERLU/libsuperlu_4.2.a
LUWRAP = $(LIBDIR)/SUPERLU/c_fortran_dgssv.o $(LIBDIR)/SUPERLU/c_fortran_zgssv.o

UMFPACK = $(LIBDIR)/UMFPACK/libumfpack.a $(LIBDIR)/UMFPACK/libamd.a
UMFWRAP = $(LIBDIR)/UMFPACK/umf4_f77wrapper.o $(LIBDIR)/UMFPACK/umf4_f77zwrapper.o


default: driver

driver: vdp_ros_dr.o $(PAR) $(LSSOLVER) $(INTEGRATOR)
	$(FC) $(FFLAGS) -o $(APP) $< $(LUWRAP) $(LUPACK) $(UMFWRAP) $(UMFPACK) $(LSSOLVER) $(INTEGRATOR) $(PAR) $(LAPACK) $(BLAS) $(XERBLA) $(LIB)

purge:  clean

clean:
	rm -f *~ *.mod *.o  

help:
	@$(ECHO) "usage: make ?"

