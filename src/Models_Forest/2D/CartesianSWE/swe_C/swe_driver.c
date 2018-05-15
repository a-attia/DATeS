
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "swe_parameters.h"
#include "swe_initial_cond.h"
#include "rok_instrumentation.h"

#define QUOTE(arg) #arg
#define NZ 32
#define SIZENAME QUOTE(NZ)

#ifdef __cplusplus
extern "C" {
#endif

void INTEGRATE( double TIN, double TOUT );

#ifdef __cplusplus
}
#endif

extern double X[];
extern double Times[];
extern int    Times_len;
double ATOL[NVAR];                       /* Absolute tolerance */
double RTOL[NVAR];                       /* Relative tolerance */
double STEPMIN;                          /* Lower bound for integration step */
double STEPMAX;                          /* Upper bound for integration step */
int    KRYLOV;
int    THREADS;
int    JVMODE;
int    SAVESTATE;
int    FIXEDSTEP;
double FSTEPSIZE;

extern int Nstp, Nacc, Nrej, Nfun, Njv;

int main(int argc, char *argv[]) {
   int i;
   
   printf("SIZE = (%d, %d)\n", NX, NY);
   
   // Time steps.
   double tstart = 0.0;
   double tend   = tspan;
   double dt     = (tend - tstart);
   double t      = tstart;
   printf("TSPAN = (%.1f - %.1f)\n", tstart, tend);
   
   // Setup tolerences.
   double TOL = 1e-6;
   if (argc > 1) TOL = strtod(argv[1], NULL);
   printf("TOL = %e\n", TOL);
   for( i = 0; i < NVAR; i++ ) {
      RTOL[i] = TOL;
      ATOL[i] = TOL;
   }
   STEPMIN = 0.01;
   STEPMAX = 900;
   
   // Setup Krylov basis size.
   KRYLOV = 4;
   if (argc > 2) KRYLOV = atoi(argv[2]);
   printf("KRYLOV = %d\n", KRYLOV);
   
   // Print number of threads.
   THREADS = omp_get_max_threads();
   if (argc > 3) THREADS = atoi(argv[3]);
   printf("THREADS = %d\n", THREADS);
   omp_set_num_threads(THREADS);
   
   // Print Jacobian-vector product mode.
   JVMODE = 1;
   if (argc > 4) JVMODE = atoi(argv[4]);
   printf("JVMODE = %d\n", JVMODE);

   // Set whether state is saved.
   SAVESTATE = 0;
   if (argc > 5) SAVESTATE = atoi(argv[5]);
   printf("SAVESTATE = %d\n", SAVESTATE);

   // Fixed time step setup.
   FIXEDSTEP = 0;
   FSTEPSIZE = 1e-2;
   if (argc > 6) FIXEDSTEP = atoi(argv[6]);
   if (argc > 7) FSTEPSIZE = strtod(argv[7], NULL);
   printf("FIXEDSTEP = %d\nFSTEPSIZE = %e\n", FIXEDSTEP, FSTEPSIZE);
   
   // Setup output file.
   char filename[128] = "swe_out_s";
   strcat(filename, N_STRING);
   strcat(filename, "_tol");
   if (argc > 1) {
      strcat(filename, argv[1]);
   } else {
      strcat(filename, "1e-6");
   }
   strcat(filename, ".dat");
   FILE *output = fopen(filename, "w");
   
   // Timing info.
   struct timeval t1, t2;
   FILE *toutput = fopen((char*)("swe_out_time.dat"), (char*)("a"));
  
   // Initialize instrumentation.
   rok_instrumentation_init();
   rok_save_init((char*)("swe_allsteps.dat"), (char*)("w"), 0);
   rok_save_init((char*)("swe_alltimes.dat"), (char*)("w"), 1);

   // Set initial conditions.
   int ICSELECT = 0;
   if (argc > 8) ICSELECT = atoi(argv[8]);
   printf("ICSELECT = %d\n", ICSELECT);
   swe_ic_set(ICSELECT);
   printf("Initial condition generated.\n");
//   swe_ic_base();
//   swe_ic_gaussian(0, 1.0, 16, 16, 2.0, 2.0);
//   swe_ic_rectangle(1, 1.0, 16, 16, 2*midgrid, 2*midgrid);
//   swe_ic_rectangle(2, 1.0, 16, 16, 2*midgrid, 2*midgrid);
//   swe_ic_gaussian(0, -0.5, 24, 8, 0.5, 0.5);
//   swe_ic_rectangle(0, 1.0, 0, 16, 4, 28);
//   swe_ic_rectangle(1, 1.0, 0, 16, 4, 32);
//   swe_ic_rectangle(2, 0.5, 16, 16, 32, 32);
 
   // Time loop;
   while (t < tend) {
      // Run integrator.
      printf("tstart=%f   tstop=%f\n", t, t+dt);
      gettimeofday(&t1, NULL);
      INTEGRATE(t, t+dt);
      gettimeofday(&t2, NULL);
      t += dt;
   }
   
   // Save data.
    for (i = 0; i < NVAR; i++) {
       fprintf(output, "%24.16e ", X[i]);
    }
    fprintf(output, "\n");
   
   // Close output file.
   fclose(output);
   
   // Save timing data.
   double total_time = 1000.0*(t2.tv_sec - t1.tv_sec) + (1.0/1000.0)*(t2.tv_usec - t1.tv_usec);
   rok_record_time(total_time, TIMER_TOTAL);
   fprintf(toutput, "%5s %10.2e %5d %7d %7d %4d %5d %9.3e %2d ", N_STRING, TOL, KRYLOV, THREADS, JVMODE, SAVESTATE, FIXEDSTEP, FSTEPSIZE, ICSELECT);
   for (i = 0; i < rok_get_num_timers(); i++) {
      fprintf(toutput, "%24.20e ", rok_get_time(i));
   }
   fprintf(toutput, "%5d %5d %5d %7d %7d ", Nstp, Nacc, Nrej, Nfun, Njv);
   for (i = 0; i < rok_get_num_err_records(); i++) {
      fprintf(toutput, "%24.20e ", rok_get_err(i));
   }
   fprintf(toutput, "\n");
   fclose(toutput);
   
   printf("times: ");
   for (i = 0; i < rok_get_num_timers(); i++) {
      printf("%fms ", rok_get_time(i));
   }
   printf("\n");

   printf("errors: ");
   for (i = 0; i < rok_get_num_err_records(); i++) {
      printf("%e ", rok_get_err(i));
   }
   printf("\n");
   
   rok_save_cleanup();

   return 0;
}

