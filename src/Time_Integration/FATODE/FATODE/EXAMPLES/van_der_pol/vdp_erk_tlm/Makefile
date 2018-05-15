############################################################################
#
##  Module:          Makefile
#
##  Purpose:         Top-level Makefile
#
##  Creation date:   Nov 27, 2010
#
##  Modified:
#
##  Send bug reports, comments or suggestions to zhang@vt.edu
#
#############################################################################
export ARCH=ar -cr
export RANLIB=ranlib
#~~~> PGF90 (Portland Group Compiler)
#~~~> Optimize and generate code for the Pentium4 architecture
export FC=pgf90
export FFLAGS=-g -O2 -fastsse -Mcache_align -tp=penryn-64 -Mflushz -Minform=warn -Mpreprocess -DFULL_ALGEBRA
FATDIR = ../../../
MODEL   = TLM
FAMILY  = ERK_TLM

APP = van_der_pol

INTEGRATOR=$(FAMILY)_f90_Integrator.o

default: driver

$(FAMILY)_f90_Integrator.o: $(FAMILY)_f90_Integrator.f90
	$(FC) $(FFLAGS) -c $<

van_der_pol_dr.o: van_der_pol_dr.f90 $(INTEGRATOR) 
	$(FC) $(FFLAGS) -c $<
#other libraries

XERBLA = -lpgftnrtl -pgf90libs

LIB = -lm

BLAS=/opt/pgi/linux86-64/7.2-5/lib/libblas.a
LAPACK=/opt/pgi/linux86-64/7.2-5/lib/liblapack.a

default: driver

driver: van_der_pol_dr.o $(INTEGRATOR)
	$(FC) $(FFLAGS) -o $(APP) $< $(INTEGRATOR) $(LAPACK) $(BLAS) $(XERBLA) $(LIB)

purge:  clean

clean:
	rm -f *~ *.o *.mod
	rm -f driver
help:
	@$(ECHO) "usage: make ?"

