
%module cart_swe
%{
#include "cart_swe_model.h"
%}

%include "cpointer.i"
%pointer_functions(double, doublep);
%pointer_functions(int, intp);

%include "carrays.i"
%array_functions(double, darray);

%include "cart_swe_model.h"

