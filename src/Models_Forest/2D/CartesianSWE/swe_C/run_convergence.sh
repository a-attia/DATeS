#!/bin/bash

# Convergence test
SIZES="128"
KRYLOV=4
THREADS=8
JVMODE=1
SAVESTATE=0
FIXEDSTEP=1
#FSTEPSIZE="0.0500 0.0250 0.0125 0.00625 0.003125 0.0015625 0.00078125 0.000390625 0.0001953125 0.00009765625 0.000048828125 0.0000244140625 0.00001220703125 0.000006103515625 0.0000030517578125"
#FSTEPSIZE="1e-4 1e-5 1e-6 1e-7"
FSTEPSIZE="0.00000152587890625 0.000000762939453125"
ICSELECT="9"

cp ./swe_time_header.txt ./swe_out_time.dat
for i in $SIZES; do
   make clean
   ./set_input.sh $i
   make openmp
   for j in $ICSELECT; do
      for l in $FSTEPSIZE; do
         ./swe_omp.exe $l $KRYLOV $THREADS $JVMODE $SAVESTATE $FIXEDSTEP $l $j
      done
      mkdir convergence_"$i"_jv_c"$j"
      mv swe_out_s"$i"_*.dat ./convergence_"$i"_jv_c"$j"
      cp swe_out_time.dat ./convergence_"$i"_jv_c"$j"
   done
done

JVMODE=2

for i in $SIZES; do
   make clean
   ./set_input.sh $i
   make openmp
   for j in $ICSELECT; do
      for l in $FSTEPSIZE; do
         ./swe_omp.exe $l $KRYLOV $THREADS $JVMODE $SAVESTATE $FIXEDSTEP $l $j
      done
      mkdir convergence_"$i"_fd_c"$j"
      mv swe_out_s"$i"_*.dat ./convergence_"$i"_fd_c"$j"
      cp swe_out_time.dat ./convergence_"$i"_fd_c"$j"
   done
done
