default:fortran64

export UMFPATH=/Users/hong/UMF_5.5.0

include $(UMFPATH)/UFconfig/UFconfig.mk

C = $(CC) $(CFLAGS) $(UMFPACK_CONFIG) -I$(UMFPATH)/UMFPACK/Include -I$(UMFPATH)/AMD/Include -I$(UMFPATH)/UFconfig

fortran: umf4_f77wrapper.o umf4_f77zwrapper.o

fortran64: umf4_f77wrapper64.o umf4_f77zwrapper64.o

UMFPACK = libumfpack.a libamd.a

umf4_f77wrapper.o:umf4_f77wrapper.c $(INC)
	$(C) -c umf4_f77wrapper.c -o umf4_f77wrapper.o
 
umf4_f77zwrapper.o: umf4_f77zwrapper.c $(INC)
	$(C) -c umf4_f77zwrapper.c -o umf4_f77zwrapper.o

umf4_f77wrapper64.o:umf4_f77wrapper.c $(INC)
	$(C) -DDLONG -c umf4_f77wrapper.c -o umf4_f77wrapper64.o

umf4_f77zwrapper64.o: umf4_f77zwrapper.c $(INC)
	$(C) -DDLONG -c umf4_f77zwrapper.c -o umf4_f77zwrapper64.o

clean:
	- rm umf4*.o
