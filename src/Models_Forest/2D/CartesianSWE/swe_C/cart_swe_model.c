
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "cart_swe_model.h"
#include "swe_parameters.h"
#include "rok_linearalgebra.h"
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
void swe_fun_b(double t, double X[], double Xb[], double Yb[]);
void swe_fun_b_init();
void swe_fun_b_cleanup();

extern double X[];

// Functions to be called directly by the Python class.

// Model class functions
int model_init(int mesh_size, int thread_count, double *initial_cond) {
    printf("model init\n");
    omp_set_num_threads(thread_count);
    swe_ic_set(10);
    rok_instrumentation_init();
    swe_fun_init();
    swe_fun_d_init();
    swe_fun_b_init();
    // Copy IC
    rok_dcopy(NVAR, X, 1, initial_cond, 1);
    return 0;
}

int model_del() {
    printf("model del\n");
    swe_fun_cleanup();
    swe_fun_d_cleanup();
    swe_fun_b_cleanup();
    return 0;
}

int model_rhs(double t, double *in_vector, double *out_vector) {
    printf("rhs\n");
    swe_fun(t, in_vector, out_vector);
    return 0;
}

// State matrix functions_
int model_jac_vec(double t, double *in_state, double *in_vector, double *out_vector) {
    printf("jac-vec\n");
    swe_fun_d(t, in_state, in_vector, out_vector);
    return 0;
}

int model_jac_t_vec(double t, double *in_state, double *in_vector, double *out_vector) {
    printf("jac-t-vec\n");
    swe_fun_b(t, in_state, out_vector, in_vector);
    return 0;
}

// State vector functions
double * vec_init() {
    printf("vec init\n");
    return (double *)rok_vector(NVAR, sizeof(double));
}

int vec_del(double *in_vector) {
    printf("vec del\n");
    rok_freevector((void *)in_vector);
    return 0;
}

int vec_get_size(int *out_integer) {
    printf("vec get size\n");
    *out_integer = NVAR;
    return 0;
}

int vec_scale(double in_alpha, double *io_vector) {
    printf("vec scale\n");
    rok_dscal(NVAR, in_alpha, io_vector, 1);
    return 0;
}

int vec_copy(double *in_vector, double *out_vector) {
    printf("vec copy\n");
    rok_dcopy(NVAR, in_vector, 1, out_vector, 1);
    return 0;
}

int vec_dot(double *in_vector_x, double *in_vector_y, double *out_scalar) {
    printf("vec dot\n");
    *out_scalar = rok_ddot(NVAR, in_vector_x, 1, in_vector_y, 1);
    return 0;
}

int vec_axpy(double in_alpha, double *in_vector_x, double *io_vector_y) {
    printf("vec axpy\n");
    rok_daxpy(NVAR, in_alpha, in_vector_x, 1, io_vector_y, 1);
    return 0;
}

int vec_add(double *in_vector_x, double *io_vector_y) {
    printf("vec add\n");
    rok_daxpy(NVAR, 1.0, in_vector_x, 1, io_vector_y, 1);
    return 0;
}

int vec_norm2(double *in_vector, double *out_scalar) {
    printf("vec norm\n");
    *out_scalar = sqrt(rok_ddot(NVAR, in_vector, 1, in_vector, 1));
    return 0;
}

