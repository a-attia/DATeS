
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "swe_parameters.h"
#include "swe_initial_cond.h"
#define NUM_FILES
#define NUM_TIMERS
#define TIMER_INDICES
#include "rok_instrumentation.h"

void swe_fun(double t, double X[], double Y[]);
void swe_fun_init();
void swe_fun_cleanup();
void swe_fun_d(double t, double X[], double Xd[], double Yd[]);
void swe_fun_d_init();
void swe_fun_d_cleanup();

extern double X[];
double Y[NVAR];
double V[NVAR];

int main(int argc, char * argv[]) {
   char filename[128];
   char number[32];

   // Setup for scaling test.
   printf("SIZE = (%d, %d)\n", NX, NY);

   // Setup number of threads.
   int THREADS = omp_get_max_threads();
   if (argc > 1) THREADS = atoi(argv[1]);
   printf("THREADS = %d\n", THREADS);
   omp_set_num_threads(THREADS);

   // Select iteration count.
   int COUNT = 10;
   if (argc > 2) COUNT = atoi(argv[2]);
   printf("COUNT = %d\n", COUNT);

   // Select initial condition.
   int ICSELECT = 0;
   if (argc > 3) ICSELECT = atoi(argv[3]);
   printf("ICSELECT = %d\n", ICSELECT);
   swe_ic_set(ICSELECT);

   // Initialize instrumentation.
   rok_instrumentation_init();

   // Save initial condition.
   int ICSAVE = 0;
   if (argc > 4) ICSAVE = atoi(argv[4]);
   printf("ICSAVE = %d\n", ICSAVE);
   if (ICSAVE) {
      strcpy(filename, "swe_ic_s");
      strcat(filename, N_STRING);
      strcat(filename, "_c");
      sprintf(number, "%d", ICSELECT);
      strcat(filename, number);
      strcat(filename, ".dat");
      rok_save_init(filename, "w", 0);
      rok_save_vector(NVAR, X, 0);
   }

   // Save timing data for RHS and JV.
   strcpy(filename, "swe_scale_times_s");
   strcat(filename, N_STRING);
   strcat(filename, "_c");
   sprintf(number, "%d", ICSELECT);
   strcat(filename, number);
   strcat(filename, ".dat");
   rok_save_init(filename, "a", 1);

   // Initialize RHS and JV.
   swe_fun_init();
   swe_fun_d_init();

   // Total times.
   struct timeval t1, t2;
   double total_time;

   // Run both RHS and JV some number of times.
   int i;
   double *x_pointer, *y_pointer, *v_pointer;

   // Copy IC into Y.
   for (i = 0; i < NVAR; i++) {
      Y[i] = X[i];
   }

   // Time COUNT iterations of the RHS.
   y_pointer = Y;
   v_pointer = V;
   gettimeofday(&t1, NULL);
   for (i = 0; i < COUNT; i++) {
      swe_fun(tspan*((double)i/(double)COUNT), y_pointer, v_pointer);
      double *ptemp = y_pointer;
      y_pointer = v_pointer;
      v_pointer = ptemp;
   }
   gettimeofday(&t2, NULL);
   
   total_time = 1000.0*(t2.tv_sec - t1.tv_sec) + (1.0/1000.0)*(t2.tv_usec - t1.tv_usec);
   rok_record_time(total_time, TIMER_RHS);

   // Time COUNT iterations of the JV.
   x_pointer = X;
   y_pointer = Y;
   v_pointer = V;
   gettimeofday(&t1, NULL);
   for (i = 0; i < COUNT; i++) {
      swe_fun_d(tspan*((double)i/(double)COUNT), x_pointer, y_pointer, v_pointer);
      double *ptemp = y_pointer;
      y_pointer = v_pointer;
      v_pointer = ptemp;
   }
   gettimeofday(&t2, NULL);
   
   total_time = 1000.0*(t2.tv_sec - t1.tv_sec) + (1.0/1000.0)*(t2.tv_usec - t1.tv_usec);
   rok_record_time(total_time, TIMER_JV);

   // Cleanup after RHS and JV (and save times).
   swe_fun_cleanup();
   swe_fun_d_cleanup();

   // Output times.
   printf("Totals:\n RHS: %fms   JV: %fms\n", rok_get_time(TIMER_RHS), rok_get_time(TIMER_JV));
   printf("Averages:\n RHS: %fms   JV: %fms\n", rok_get_time(TIMER_SWE_FUN), rok_get_time(TIMER_SWE_FUN_D));

   double timings[4];
   timings[0] = rok_get_time(TIMER_RHS);
   timings[1] = rok_get_time(TIMER_JV);
   timings[2] = rok_get_time(TIMER_SWE_FUN);
   timings[3] = rok_get_time(TIMER_SWE_FUN_D);

   rok_save_vector(4, timings, 1);

   // Cleanup
   rok_save_cleanup();
   
   return 0;
}

