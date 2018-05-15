gcc -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -DMAJOR_VERSION=1 -DMINOR_VERSION=0 -I/usr/include/python2.7/ -c fatode_erk_fwd_c_layer.c 
gcc -fPIC -g -c ERK_f90_Integrator.F90 
ld -shared --start-group ERK_f90_Integrator.o fatode_erk_fwd_c_layer.o --end-group -lblas -llapack -o fatode_erk_fwd.so

