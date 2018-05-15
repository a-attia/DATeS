#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 1024000

double vector1[N];
double vector2[N];

int main () {
   int i, len = N;
   double a = 0.5;
   double ompdot = 0.0;
   double serdot = 0.0;

   srand(0);

   for (i = 0; i < len; i++) {
      vector1[i] = a * (double)(rand() % 1000) / 1000.0;
      vector2[i] = a * a * a * (double)(rand() % 10000) / 10000.0;
   }

   // Serial
   for (i = 0; i < len; i++) {
      serdot += vector1[i] * vector2[i];
   }
   printf("serial dot-product: %24.20e\n", serdot);

   // OpenMP
   #pragma omp parallel for private(i) shared(len, vector1, vector2) reduction(+:ompdot) schedule(auto) num_threads(8)
   for (i = 0; i < len; i++) {
      ompdot = ompdot + vector1[i] * vector2[i];
   }
   printf("parallel dot-product: %24.20e\n", ompdot);

   printf("difference: %24.20e\n", serdot - ompdot);
   
   return 0;
}

