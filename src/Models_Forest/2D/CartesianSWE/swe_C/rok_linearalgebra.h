
#ifdef __cplusplus
extern "C" {
#endif

// Threading setup/cleanup
void rok_threadsetup();
void rok_threadcleanup();

// Vector and Matrix allocation
void * rok_vector(int N, size_t typesize);
void * rok_matrix(int M, int N, size_t typesize);
void rok_freevector(void * vector);
void rok_freematrix(void * matrix);

// BLAS like functions
void   rok_dcopy(int N, double X[], int incX, double Y[], int incY);
void   rok_daxpy(int N, double Alpha, double X[], int incX, double Y[], int incY);
void   rok_dscal(int N, double Alpha, double X[], int incX);
void   rok_dadd(int N, double X[], double Y[], double Z[]);
double rok_ddot(int N, double X[], int incX, double Y[], int incY);
void   rok_dgemv(char trans, int M, int N, double alpha, double A[], int lda, double X[], int incX, double beta, double Y[], int incY);

// Lapack like functions.
int rok_dgetrf(int M, int N, double *A, int lda, int *pivot);
int rok_dgetrs(char trans, int N, int NRHS, double *A, int lda, int *pivot, double *B, int ldb);

// Helpers.
double rok_dlamch( char C );
void   rok_set2zero(int N, double Y[]);

#ifdef __cplusplus
}
#endif

