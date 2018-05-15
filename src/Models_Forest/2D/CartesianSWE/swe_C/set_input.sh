#!/bin/bash

case "$1" in
32 | 64 | 128 | 256 | 512 | 1024)
	rm -f swe_input.c swe_parameters.h
	ln -s ./swe_input$1.c ./swe_input.c
	ln -s ./swe_parameters$1.h ./swe_parameters.h
	;;
*)
	echo "Bad input size."
	;;
esac

