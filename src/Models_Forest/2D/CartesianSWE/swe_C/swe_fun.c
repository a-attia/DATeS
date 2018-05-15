
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include "swe_parameters.h"
#include "rok_instrumentation.h"

void swe_fun(double t, double X[], double Y[]);
void swe_fun_init();
void swe_fun_cleanup();
void swe_fun_main(int M, int N, double X[], double dx, double dy, double Y[]);

// timing data
double fun_time;
double fun_count;

#define Xdim (NX+2)
#define Ydim (NY+2)

//double tx2d[Xdim][Ydim], tu2d[Xdim][Ydim], tv2d[Xdim][Ydim];
double Hx[Xdim-1][Ydim-1],Ux[Xdim-1][Ydim-1],Vx[Xdim-1][Ydim-1];
double Hy[Xdim-1][Ydim-1], Uy[Xdim-1][Ydim-1], Vy[Xdim-1][Ydim-1];

void swe_fun_init() {
   fun_count = 0;
   fun_time = 0.0;
}

void swe_fun_cleanup() {
   rok_record_time(fun_time/fun_count, TIMER_SWE_FUN);
}

void swe_fun(double t, double X[], double Y[]) {
   struct timeval t1, t2;
   gettimeofday(&t1, NULL);
   swe_fun_main(NX+2, NY+2, X, DX, DY, Y);
   gettimeofday(&t2, NULL);
   fun_time += 1000.0 * (t2.tv_sec - t1.tv_sec) + (1.0/1000.0) * (t2.tv_usec - t1.tv_usec);
   fun_count++;
}

void swe_fun_main(int M, int N, double X[], double dx, double dy, double Y[])
{
   int i, j;
   
//   int l;
//   l = M * N;
   
   // Initialize derivatives
#pragma omp parallel private(i, j)
{
   #pragma omp for schedule(static)
   for(i = 0; i < M-1; i++) {
      for(j = 0; j < N-1; j++) {
         Hx[i][j] = 0.0;
      }
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M-1; i++) {
      for(j = 0; j < N-1; j++) {
         Hy[i][j] = 0.0;
      }
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M-1; i++) {
      for(j = 0; j < N-1; j++) {
         Ux[i][j] = 0.0;
      }
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M-1; i++) {
      for(j = 0; j < N-1; j++) {
         Uy[i][j] = 0.0;
      }
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M-1; i++) {
      for(j = 0; j < N-1; j++) {
         Vx[i][j] = 0.0;
      }
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M-1; i++) {
      for(j = 0; j < N-1; j++) {
         Vy[i][j] = 0.0;
      }
   }
}
   // Unpack current state
/*     for (i = 0; i < M; i++) {
      for(j = 0; j < N; j++) {
         int p = i * M + j;
         tx2d[i][j] = X[p];
         tu2d[i][j] = X[l+p];
         tv2d[i][j] = X[2*l+p];
      }
   } */
   
#define x2d(V, I, J) ( (V)[(I) * M + (J)] )
#define u2d(V, I, J) ( (V)[(M * N) + (I) * M + (J)] )
#define v2d(V, I, J) ( (V)[(2 * M * N) + (I) * M + (J)] )
//#define x2d(V, I, J) (tx2d[(I)][(J)])
//#define u2d(V, I, J) (tu2d[(I)][(J)])
//#define v2d(V, I, J) (tv2d[(I)][(J)])
   
   // Compute x-direction finite differences
#pragma omp parallel private(i, j)
{
   #pragma omp for schedule(static)
   for (i = 0; i < M-1; i++) {
      for(j = 0; j < N-2; j++) {
         Hx[i][j] = (x2d(X,i+1,j+1) + x2d(X,i,j+1)) / 2.0;
      }
   }
   #pragma omp for schedule(static)
   for (i = 0; i < M-1; i++) {
      for(j = 0; j < N-2; j++) {
         Ux[i][j] = (u2d(X,i+1,j+1) + u2d(X,i,j+1)) / 2.0;
      }
   }
   #pragma omp for schedule(static)
   for (i = 0; i < M-1; i++) {
      for(j = 0; j < N-2; j++) {
         Vx[i][j] = (v2d(X,i+1,j+1) + v2d(X,i,j+1)) / 2.0;
      }
   }
   
   // Compute y-direction finite differences
   #pragma omp for schedule(static)
   for (i = 0; i < M-2; i++) {
      for(j = 0; j < N-1; j++) {
         Hy[i][j] = (x2d(X,i+1,j+1) + x2d(X,i+1,j)) / 2.0;
      }
   }
   #pragma omp for schedule(static)
   for (i = 0; i < M-2; i++) {
      for(j = 0; j < N-1; j++) {
         Uy[i][j] = (u2d(X,i+1,j+1) + u2d(X,i+1,j)) / 2.0;
      }
   }
   #pragma omp for schedule(static)
   for (i = 0; i < M-2; i++) {
      for(j = 0; j < N-1; j++) {
         Vy[i][j] = (v2d(X,i+1,j+1) + v2d(X,i+1,j)) / 2.0;
      }
   }
}
  
   // Evaluate shallow water equations in 2D
#pragma omp parallel private(i, j)
{
   #pragma omp for schedule(static)
   for (i = 1; i < M-1; i++) {
      for(j = 1; j < N-1; j++) {
         x2d(Y,i,j) = -(1.0/dx) * (Ux[i][j-1] - Ux[i-1][j-1]) - 
                       (1.0/dy) * (Vy[i-1][j] - Vy[i-1][j-1]);
      }
   }
   #pragma omp for schedule(static)
   for (i = 1; i < M-1; i++) {
      for(j = 1; j < N-1; j++) {
         u2d(Y,i,j) = -(1.0/dx) * ((Ux[i][j-1] * Ux[i][j-1] / Hx[i][j-1] + (g/2.0) * Hx[i][j-1] * Hx[i][j-1]) -
                                   (Ux[i-1][j-1] * Ux[i-1][j-1] / Hx[i-1][j-1] + (g/2.0) * Hx[i-1][j-1] * Hx[i-1][j-1])) -
                       (1.0/dy) * ((Vy[i-1][j] * Uy[i-1][j] / Hy[i-1][j]) - 
                                   (Vy[i-1][j-1] * Uy[i-1][j-1] / Hy[i-1][j-1]));
      }
   }
   #pragma omp for schedule(static)
   for (i = 1; i < M-1; i++) {
      for(j = 1; j < N-1; j++) {
         v2d(Y,i,j) = -(1.0/dx) * ((Ux[i][j-1] * Vx[i][j-1] / Hx[i][j-1]) - 
                                   (Ux[i-1][j-1] * Vx[i-1][j-1] / Hx[i-1][j-1])) -
                       (1.0/dy) * ((Vy[i-1][j] * Vy[i-1][j] / Hy[i-1][j] + (g/2.0) * Hy[i-1][j] * Hy[i-1][j]) - 
                                   (Vy[i-1][j-1] * Vy[i-1][j-1] / Hy[i-1][j-1] + (g/2.0) * Hy[i-1][j-1] * Hy[i-1][j-1]));
      }
   }
}  

   // Fill in boundary conditions
#pragma omp parallel private(i, j)
{
   #pragma omp for schedule(static)
   for(i = 0; i < M; i++) {
      x2d(Y,i,0) = x2d(Y,i,1);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M; i++) {
      u2d(Y,i,0) = u2d(Y,i,1);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M; i++) {
      v2d(Y,i,0) = -v2d(Y,i,1);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M; i++) {
      x2d(Y,i,N-1) = x2d(Y,i,N-2);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M; i++) {
      u2d(Y,i,N-1) = u2d(Y,i,N-2);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < M; i++) {
      v2d(Y,i,N-1) = -v2d(Y,i,N-2);
   }
   
   #pragma omp for schedule(static)
   for(i = 0; i < N; i++) {
      x2d(Y,0,i) = x2d(Y,1,i);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < N; i++) {
      u2d(Y,0,i) = -u2d(Y,1,i);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < N; i++) {
      v2d(Y,0,i) = v2d(Y,1,i);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < N; i++) {
      x2d(Y,M-1,i) = x2d(Y,M-2,i);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < N; i++) {
      u2d(Y,M-1,i) = -u2d(Y,M-2,i);
   }
   #pragma omp for schedule(static)
   for(i = 0; i < N; i++) {
      v2d(Y,M-1,i) = v2d(Y,M-2,i);
   }
}
   
   // Repack
/*    for(i=0;i<M;i++)
   {
         for(j=0;j<N;j++)
         {
               int p = i*M+j;
               Y[p]=tx2d[i][j];
               Y[l+p]=tu2d[i][j];
               Y[2*l+p]=tv2d[i][j];
         }
   } */
   return;
}
