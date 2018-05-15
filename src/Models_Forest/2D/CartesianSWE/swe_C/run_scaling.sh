#!/bin/bash
set -x

# RHS and JV scalability test
SIZES="128 256 512 1024"
THREADS="1 2 4 8 16 20 32 40"
COUNT=10000
ICSELECT="0 1 2"

#SIZES="128"
#THREADS="1 2 4 8"
#COUNT=10000
#ICSELECT="0"


for i in $SIZES; do
   make clean
   ./set_input.sh $i
   make scale
   for j in $ICSELECT; do
      cp ./swe_scale_header.txt ./swe_scale_times_s"$i"_c"$j".dat
      for l in $THREADS; do
         ./swe_scale.exe $l $COUNT $j
      done
   done
done

