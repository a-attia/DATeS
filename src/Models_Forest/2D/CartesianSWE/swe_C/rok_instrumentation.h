
#ifdef __cplusplus
extern "C" {
#endif

// Initialization.
void rok_instrumentation_init();

// Timing measurement functions.
int    rok_get_num_timers();
void   rok_record_time(double time, int index);
double rok_get_time(int index);

// Error norm measurement functions.
int    rok_get_num_err_records();
void   rok_record_err(double err, int index);
double rok_get_err(int index);

// Save vectors to file.
void rok_save_init(char * filename, char * mode, int index);
void rok_save_cleanup();
void rok_save_vector(int N, double vector[], int index);

#define TIMER_TOTAL     0
#define TIMER_ARNOLDI   2
#define TIMER_SOLVER    3
#define TIMER_SWE_FUN   1
#define TIMER_SWE_FUN_D 4
#define TIMER_SWE_FUN_B 5
#define TIMER_RHS       2
#define TIMER_JV        3

#ifdef __cplusplus
}
#endif

