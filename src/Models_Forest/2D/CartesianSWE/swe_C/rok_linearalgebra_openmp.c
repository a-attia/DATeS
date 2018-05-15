
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cblas.h>
#include <clapack.h>


#define TMULT 4
#define ZERO  (double)0.0
#define ONE   (double)1.0
#define HALF  (double)0.5
#define TWO   (double)2.0
#define MOD(A,B) (int)((A)%(B))

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_threadsetup() {}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   OpenMP threading.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_threadcleanup() {}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   OpenMP threading.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void * rok_vector(int N, size_t typesize)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   if (N * typesize > 0)
      return malloc(N * typesize);
   else
      return NULL;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void * rok_matrix(int M, int N, size_t typesize)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   if (M * N * typesize > 0)
      return malloc(M * N * typesize);
   else
      return NULL;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_freevector(void * vector)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   if (vector != NULL)
      free(vector);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_freematrix(void * matrix)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   if (matrix != NULL)
      free(matrix);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_dcopy(int N, double X[], int incX, double Y[], int incY)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    copies a vector, x, to a vector, y:  y <- x
    only for incX=incY=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   int i, tcount;
   tcount = TMULT*omp_get_max_threads();
   if (N <= 0) return;

#pragma omp parallel for private(i) shared(X, Y, N) if(N > tcount)
   for (i = 0; i < N; i++) {
      Y[i] = X[i];
   }
} /* end function WCOPY */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_daxpy(int N, double Alpha, double X[], int incX, double Y[], int incY)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    constant times a vector plus a vector: y <- y + Alpha*x
    only for incX=incY=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   int i, tcount;
   tcount = TMULT*omp_get_max_threads();

   if (Alpha == ZERO) return;
   if (N  <=  0) return;

#pragma omp parallel for private(i) shared(Alpha, X, Y, N) if(N > tcount)
   for (i = 0; i < N; i++) {
      Y[i] = Y[i] + Alpha*X[i];
   }
} /* end function  WAXPY */



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_dscal(int N, double Alpha, double X[], int incX)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    constant times a vector: x(1:N) <- Alpha*x(1:N) 
    only for incX=incY=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   int i, tcount;
   tcount = TMULT*omp_get_max_threads();

   if (Alpha == ONE) return;
   if (N  <=  0) return;

   if (Alpha == (-ONE)) {
#pragma omp parallel for private(i) shared(X, N) if(N > tcount)
      for (i = 0; i < N; i++) {
         X[i] = -X[i];
      }
   } else if (Alpha == ZERO) {
#pragma omp parallel for private(i) shared(X, N) if(N > tcount)
      for (i = 0; i < N; i ++) {
         X[i] = ZERO;
      }
   } else {
#pragma omp parallel for private(i) shared(Alpha, X, N) if(N > tcount)
      for (i = 0; i < N; i ++) {
         X[i] = Alpha * X[i];
      }
   }
} /* end function WSCAL */

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
double rok_ddot(int N, double X[], int incX, double Y[], int incY)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dot product of two vectors: p <- x(1:N)^T * y(1:N) 
    only for incX=incY=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   double temp;
   int i, tcount;
   tcount = TMULT*omp_get_max_threads();
   
   temp = 0.0;
   
   if (N <= 0) return 0.0;
   
#pragma omp parallel for private(i) shared(X, Y, N) reduction(+:temp) if(N > tcount)
   for (i = 0; i < N; i++) {
      temp = temp + X[i] * Y[i];
   }
   return temp;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_dgemv(char trans, int M, int N, double alpha, double A[], int lda, double X[], int incX, double beta, double Y[], int incY)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Matrix vector product: y <- alpha*A*x + beta*y 
    only for incX=incY=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   double temp;
   int i, j, lenx, leny, tcount;
   tcount = TMULT*omp_get_max_threads();
  
   if (M == 0 || N == 0 || (alpha == ZERO && beta == ZERO)) return;
   
   if (trans == 'n' || trans == 'N') {
      lenx = N;
      leny = M;
   } else {
      lenx = M;
      leny = N;
   }

   // First, perform y <- beta*y
   if (beta != ONE) {
      if (beta == ZERO) {
#pragma omp parallel for private(i) shared(leny, Y) if(leny > tcount)
         for (i = 0; i < leny; i++) {
            Y[i] = ZERO;
         }
      } else {
#pragma omp parallel for private(i) shared(leny, beta, Y) if(leny > tcount)
         for (i = 0; i < leny; i++) {
            Y[i] = beta*Y[i];
         }
      }
   }
   if (alpha == ZERO) return;
   
   if (trans == 'n' || trans == 'N') {
      // Next, y <- alpha*A*x + y
#pragma omp parallel for private(i, j, temp) shared(leny, lenx, lda, alpha, A, X, Y) if(leny >= lenx && leny > tcount)
      for (i = 0; i < leny; i++) {
         temp = ZERO;
#pragma omp parallel for private(j) shared(i, lenx, lda, A, X) reduction(+:temp) if(lenx > leny && lenx > tcount)
         for (j = 0; j < lenx; j++) {
            temp = temp + A[i*lda + j]*X[j];
         }
         Y[i] = Y[i] + alpha*temp;
      }
   } else {
      // y <- alpha*A^T*x + y
      for (i = 0; i < lenx; i++) {
         if (X[i] != ZERO) {
            temp = alpha*X[i];
#pragma omp parallel for private(j) shared(i, lda, leny, temp, A, Y) if(leny > tcount)
            for (j = 0; j < leny; j++) {
               Y[j] = Y[j] + temp*A[i*lda + j];
            }
         }
      }
   }
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int rok_dgetrf(int M, int N, double *A, int lda, int *pivot)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 * 
 * 
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
//   printf("ERROR: rok_dgetrf() is unimplemented.\n");
//   exit(-1);
//   return -1;
   return clapack_dgetrf(CblasRowMajor, M, N, A, lda, pivot);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int rok_dgetrs(char trans, int N, int NRHS, double *A, int lda, int *pivot, double *B, int ldb)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 * 
 * 
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
//   printf("ERROR: rok_dgetrs() is unimplemented.\n");
//   exit(-1);
//   return -1;
   if (trans == 'y' || trans == 'Y')
      return clapack_dgetrs(CblasRowMajor, CblasTrans, N, NRHS, A, lda, pivot, B, ldb);
   else
      return clapack_dgetrs(CblasRowMajor, CblasNoTrans, N, NRHS, A, lda, pivot, B, ldb);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
double rok_dlamch_add( double  A, double  B )
{
   return (A + B);
} /* end function  WLAMCH_ADD */

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
double rok_dlamch( char C )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    returns epsilon machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   int i;
   double Suma;
   static double Eps;
   static char First = 1;
   
   if (First) {
      First = 0;
      Eps = pow(HALF,16);
      for ( i = 17; i <= 80; i++ ) {
         Eps = Eps*HALF;
   Suma = rok_dlamch_add(ONE,Eps);
   if (Suma <= ONE) break;
      } /* end for */
      if (i==80) {
   printf("\nERROR IN WLAMCH. Very small EPS = %g\n",Eps);
         return (double)2.2e-16;
}
      Eps *= TWO; i--;      
   } /* end if First */

   return Eps;

} /* end function WLAMCH */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   copies zeros into the vector y:  y <- 0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_set2zero(int N, double Y[])
{
   int i, tcount;
   tcount = TMULT*omp_get_max_threads();
   
   if (N <= 0) return;

#pragma omp parallel for private(i) shared(N, Y) if(N > tcount)
   for (i = 0; i < N; i++) {
      Y[i] = ZERO;
   }
} /* end function Set2Zero */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   adds two vectors: z <- x + y     BLAS - like
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_dadd(int N, double X[], double Y[], double Z[])
{
   int i, tcount;
   tcount = TMULT*omp_get_max_threads();

   if ( N <= 0 ) return;
   
#pragma omp parallel for private(i) shared(N, X, Y, Z) if(N > tcount)
   for (i = 0; i < N; i++) {
      Z[i] = X[i] + Y[i];
   }
} /* end function WADD */

/* End of BLAS_UTIL function                                        */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

