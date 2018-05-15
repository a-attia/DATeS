# Set here the desired C compiler and its optimization options
CC   = gcc
CXX  = g++
#/opt/oracle/solstudio12.2/bin/cc
CFLAGS = -Wall -g -fPIC $(OPTFLAGS) -fopenmp
debug: CFLAGS += -DDEBUG
CXXFLAGS = -std=c++11 $(CFLAGS)

#OPTFLAGS = -m64 -O3 -march=native -ftree-vectorize
#OPTFLAGS = -m64 -Ofast -march=barcelona -msse4a -ftree-vectorize
OPTFLAGS = -m64 -O3 -march=native -mavx -ftree-vectorize

#INCLUDES = 
#CLIBS = -lm

INCLUDES = -I/home/rossg/software/atlas/gcc/include -I/usr/include/python2.7
CLIBS = -lm -L/home/rossg/software/atlas/gcc/lib -llapack -lcblas -latlas

INTEGRATORS   = rok_integrator.c
INTEGRATOROBJ = rok_integrator.o

HEADERS = swe_parameters.h rok_linearalgebra.h rok_instrumentation.h swe_initial_cond.h cart_swe_model.h random.h

SRC =   swe_driver.c swe_scale_driver.c swe_fun.c swe_jac.c swe_times.c swe_fun_d.c swe_fun_b.c rok_integrator.c rok_linearalgebra_serial.c rok_linearalgebra_openmp.c rok_instrumentation.c swe_initial_cond.c cart_swe_model.c random.cpp

SWIG = swig
SWIGFLAGS = -python -O -Wall
SWIGSRC = cart_swe_model_wrap.c
SWIGOBJ = cart_swe_model.o cart_swe_model_wrap.o
SWIGWRAP = cart_swe_model.i

OBJ =   swe_fun.o swe_jac.o swe_times.o swe_fun_d.o swe_fun_b.o rok_instrumentation.o swe_initial_cond.o random.o

DRIVEROBJ = swe_driver.o

SERIALOBJ = rok_linearalgebra_serial.o
OPENMPOBJ = rok_linearalgebra_openmp.o

SCALEOBJ = swe_scale_driver.o

default: serial

serial: build
	$(CXX) $(CFLAGS) $(DRIVEROBJ) $(OBJ) $(INTEGRATOROBJ) $(SERIALOBJ) $(CLIBS) -o swe_serial.exe   


openmp: build
	$(CXX) $(CFLAGS) $(DRIVEROBJ) $(OBJ) $(INTEGRATOROBJ) $(OPENMPOBJ) $(CLIBS) -o swe_omp.exe   

build: $(HEADERS) $(DRIVEROBJ) $(OBJ) $(INTEGRATOROBJ) $(SERIALOBJ) $(OPENMPOBJ) $(SCALEOBJ) $(SWIGOBJ)

debug: build
	$(CXX) $(CFLAGS) $(DRIVEROBJ) $(OBJ) $(INTEGRATOROBJ) $(SERIALOBJ) $(CLIBS) -o swe_debug.exe   

scale: build
	$(CXX) $(CFLAGS) $(SCALEOBJ) $(OBJ) $(CLIBS) -o swe_scale.exe

lib: build
	$(CXX) $(CFLAGS) -shared $(OBJ) $(OPENMPOBJ) $(SWIGOBJ) $(CLIBS) -o _cart_swe.so

clean:
	rm -f *.o swe_*.exe rok_debug_output.txt
	rm -f *.so *.a *_wrap.c _cart_swe.py *.pyc
#swe*.dat

zip: $(HEADERS) $(SRC)
	mkdir swe_project
	cp *.cpp *.c *.h *.sh *_header.txt Makefile ./swe_project
	tar -czf swe_project.tar.gz ./swe_project
	rm -rf swe_project

$(SWIGSRC): $(SWIGWRAP) $(HEADERS)
	$(SWIG) $(SWIGFLAGS) $<

$(DRIVEROBJ): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

$(SCALEOBJ): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
