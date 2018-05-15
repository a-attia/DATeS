#include <stdio.h>
#include "rok_instrumentation.h"

// Timing.
#ifndef NUM_TIMERS
#define NUM_TIMERS 6
#endif
double rok_timedata[NUM_TIMERS];

void rok_record_time(double time, int index) {
   if (index >= 0 && index < NUM_TIMERS)
      rok_timedata[index] = time;
}

double rok_get_time(int index) {
   if (index >= 0 && index < NUM_TIMERS)
      return rok_timedata[index];
   else
      return -1.0;
}

int rok_get_num_timers() {
   return (int)NUM_TIMERS;
}

// Error norms.
#ifndef NUM_ERR_RECORDS
#define NUM_ERR_RECORDS 7
#endif
double rok_errdata[NUM_ERR_RECORDS];

void rok_record_err(double err, int index) {
   if (index >= 0 && index < NUM_ERR_RECORDS)
      rok_errdata[index] = err;
}

double rok_get_err(int index) {
   if (index >= 0 && index < NUM_ERR_RECORDS)
      return rok_errdata[index];
   else
      return -1.0;
}

int rok_get_num_err_records() {
   return (int)NUM_ERR_RECORDS;
}


// Initialization.

void rok_instrumentation_init() {
   int i;
   for (i = 0; i < NUM_TIMERS; i++) {
      rok_timedata[i] = 0.0;
   }
   for (i = 0; i < NUM_ERR_RECORDS; i++) {
      rok_errdata[i] = 0.0;
   }
}


// Save vectors to file.
#ifndef NUM_FILES
#define NUM_FILES 3
#endif
FILE * rok_files[NUM_FILES];

void rok_save_init(char * filename, char * mode, int index) {
   if (index >= 0 && index < NUM_FILES) {
      rok_files[index] = fopen(filename, mode);
   }
}

void rok_save_cleanup() {
   int i;
   for (i = 0; i < NUM_FILES; i++) {
      if (rok_files[i])
         fclose(rok_files[i]);
   }
}

void rok_save_vector(int N, double vector[], int index) {
   int i;
   if (index >= 0 && index < NUM_FILES && rok_files[index]) {
      for (i = 0; i < N; i++) {
         fprintf(rok_files[index], "%24.20e  ", vector[i]);
      }
      fprintf(rok_files[index], "\n");
   }
}

