#!/bin/bash

# Standard speedup experiment.
#TOLS="3 4 5 6 7 8 9 10"
#SIZES="32 64 128 256 512 1024"
#KRYLOV="4"
#THREADS="1 2 4 8"

# Thread block size experiment.
#TOLS="6"
#SIZES="64 128 256 512 1024"
#KRYLOV="10"
#THREADS="128 256 384 512 640 768 996 1024"

# Arnoldi scalability experiment.
#TOLS="2"
#SIZES="32 64 128 256"
#KRYLOV="10 40 70 100"
#THREADS="256"

# Convergence test
TOLS="4"
SIZES="128"
KRYLOV=4
THREADS=8
JVMODE=2
SAVESTATE=0
FIXEDSTEP=1
FSTEPSIZE="0.0500 0.0250 0.0125 0.00625 0.003125 0.0015625 0.00078125 0.000390625 0.0001953125 0.00009765625"
ICSELECT="0"

cp ./swe_time_header.txt ./swe_out_time.dat
for i in $SIZES; do
   make clean
   ./set_input.sh $i
   make openmp
   for j in $ICSELECT; do
      for l in $FSTEPSIZE; do
         ./swe_omp.exe $l $KRYLOV $THREADS $JVMODE $SAVESTATE $FIXEDSTEP $l $j
      done
   done
done
