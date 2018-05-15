/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                                                  */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "rok_linearalgebra.h"
#include "swe_parameters.h"
#include "rok_instrumentation.h"


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                                                  */
/* INTEGRATE - Integrator routine                                   */
/*   Arguments :                                                    */
/*      TIN       - Start Time for Integration                      */
/*      TOUT      - End Time for Integration                        */
/*                                                                  */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


 #define MAX(a,b) ( ((a) >= (b)) ?(a):(b)  )
 #define MIN(b,c) ( ((b) <  (c)) ?(b):(c)  )	
 #define ABS(x)   ( ((x) >=  0 ) ?(x):(-x) ) 
 #define SQRT(d)  ( pow((d),0.5)  )
 #define SIGN(x)  ( ((x) >=  0 ) ?[0]:(-1) )

/*~~> Numerical constants */
 #define  ZERO     (double)0.0
 #define  ONE      (double)1.0
 #define  TWO      (double)2.0
 #define  HALF     (double)0.5
 #define  DeltaMin (double)1.0e-6    
   
/*~~> Debug print stuff. */
FILE * debug_file;

#ifdef DEBUG
   #define DEBUG_SETUP(S)     debug_file = fopen((S), "w")
   #define DEBUG_PRINT(...)   fprintf(debug_file, __VA_ARGS__)
   #define DEBUG_CLEAN        fclose(debug_file)
#else
   #define DEBUG_SETUP(S)        
   #define DEBUG_PRINT(...)   
   #define DEBUG_CLEAN        
#endif


/*~~~> External variables. */
 extern double X[];
 extern double ATOL[];
 extern double RTOL[];
 extern double STEPMIN;
 extern double STEPMAX;
 extern int    KRYLOV;
 extern int    JVMODE;
 extern int    SAVESTATE;
 extern int    FIXEDSTEP;
 extern double FSTEPSIZE;
 
/*~~~> Collect statistics: global variables */   
 int Nfun,Njac,Njv,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng;
 double jv_err;
 double jv_maxerr;
 int jv_count;
 double y_err;
 double y_maxerr;
 int y_count;
 double eps_ave;
 double eps_max;
 double eps_min;
 int eps_count;

void jv_init() {
   jv_err =  0.0;
   jv_count = 0;
   jv_maxerr = INT_MIN;
   y_err = 0.0;
   y_maxerr = INT_MIN;
   y_count = 0;
   eps_ave = 0.0;
   eps_max = INT_MIN;
   eps_min = INT_MAX;
   eps_count = 0;
}

/*~~~> Function headers */   
 void FunTemplate(double, double [], double []); 
 void JacTemplate(double, double [], double []);
 int RosenbrockKrylov(double Y[], double Tstart, double Tend,
     double AbsTol[], double RelTol[],
     void (*ode_Fun)(double, double [], double []), 
     void (*ode_Jac)(double, double [], double []),
     double RPAR[], int IPAR[]);
 int RosenbrockKrylovIntegrator(
     double Y[], double Tstart, double Tend ,     
     double  AbsTol[], double  RelTol[],
     void (*ode_Fun)(double, double [], double []), 
     void (*ode_Jac)(double, double [], double []),
     int rok_S,
     double rok_M[], double rok_E[], 
     double rok_A[], double rok_C[],
     double rok_Alpha[],double  rok_Gamma[],
     double rok_ELO, char rok_NewF[],
     char Autonomous, char FixedStep, int FullJacobian, char VectorTol,
     int Max_no_steps, int Krylov_size, 
     double Roundoff, double Hmin, double Hmax, double Hstart, double Hfixed,
     double FacMin, double FacMax, double FacRej, double FacSafe, 
     double *Texit, double *Hexit );
 double rok_VectorNorm(
       int length, double vector[] );
 void rok_JacVectorProd(
     int FullJacobian, double T, double norm_Y,
     double Y[], double Fcn0[], double Jac[], double vector[], 
     void (*ode_Fun)(double, double [], double []),
     double result[] );
 void rok_ArnoldiMethod (
      int Krylov_size, double Fcn0[],
     int FullJacobian, char Autonomous, double T,
     double Y[], double Jac0[], double dFdT[], 
     void (*ode_Fun)(double, double [], double []),
     double Hes[], double Vtrans[] );
 char rok_PrepareMatrix (
     double* H, 
     int Direction, int Krylov_size, double gam, double Hes[], 
     double Imhgh[], int Pivot[] );
 double rok_ErrorNorm ( 
     double Y[], double Ynew[], double Yerr[], 
     double AbsTol[], double RelTol[], 
     char VectorTol );
 int  rok_ErrorMsg(int Code, double T, double H);
 void rok_FunTimeDerivative ( 
     double T, double Roundoff, 
     double Y[], double Fcn0[], 
     void ode_Fun(double, double [], double []), 
     double dFdT[] );
 void FunTemplate( double T, double Y[], double Ydot[] );
 void JacTemplate( double T, double Y[], double Ydot[] );
 void DecompTemplate( int N, double A[], int Pivot[], int* ising );
 void SolveTemplate( int N, double A[], int Pivot[], double b[] );
 void Rok4a ( int *rok_S, double rok_A[], double rok_C[], 
             double rok_M[], double rok_E[], 
       double rok_Alpha[], double rok_Gamma[], 
       char rok_NewF[], double *rok_ELO, char* rok_Name );
 void Rok4b ( int *rok_S, double rok_A[], double rok_C[], 
             double rok_M[], double rok_E[], 
       double rok_Alpha[], double rok_Gamma[], 
       char rok_NewF[], double *rok_ELO, char* rok_Name );
 void Rok4p ( int *rok_S, double rok_A[], double rok_C[], 
             double rok_M[], double rok_E[], 
       double rok_Alpha[], double rok_Gamma[], 
       char rok_NewF[], double *rok_ELO, char* rok_Name );

 void swe_fun(double t, double X[], double Y[]);
 void swe_fun_init();
 void swe_fun_cleanup();
 void swe_fun_d(double t, double X[], double Xd[], double Yd[]);
 void swe_fun_d_init();
 void swe_fun_d_cleanup();
 void swe_jac(double t, double Y[], double J[]);
 void swe_jac_vec(double J[], double vector[], double product[]);
 void swe_jac_print(double J[]);
 
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void INTEGRATE( double TIN, double TOUT )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   static double  RPAR[20];
   static int  i, IERR, IPAR[20];
   static int Ns=0, Na=0, Nr=0, Ng=0;
   
   DEBUG_SETUP("rok_debug_output.txt");
   rok_threadsetup();

   for ( i = 0; i < 20; i++ ) {
     IPAR[i] = 0;
     RPAR[i] = ZERO;
   } /* for */
   
   
   IPAR[0] = 1;    /* non-autonomous */
   IPAR[1] = 1;    /* vector tolerances */
   RPAR[2] = STEPMIN; /* starting step */
   IPAR[3] = 1;    /* choice of the method */
   IPAR[4] = KRYLOV; /* Krylov space dimension */
   IPAR[5] = JVMODE; /* 0 for full Jacobian,
                        1 for Jacobian-vector product,
                        2 for finite-difference approx. */
   IPAR[6] = FIXEDSTEP; /* Toggles fixed stepsize. */
   RPAR[7] = FSTEPSIZE;

   IERR = RosenbrockKrylov(X, TIN, TOUT,
           ATOL, RTOL,
           &FunTemplate, &JacTemplate,
           RPAR, IPAR);

       
   Ns=Ns+IPAR[12];
   Na=Na+IPAR[13];
   Nr=Nr+IPAR[14];
   Ng=Ng+IPAR[17];
   printf("\n Step=%d  Acc=%d  Rej=%d  Singular=%d\n",
         Ns,Na,Nr,Ng);
   printf(" Njac=%d  Nfun=%d  Njv=%d\n", Njac, Nfun, Njv);


   if (IERR < 0)
     printf("\n Rosenbrock-Krylov: Unsucessful step at T=%g: IERR=%d\n",
         TIN,IERR);
   
   TIN = RPAR[10];      /* Exit time */
   STEPMIN = RPAR[11];  /* Last step */
   
   rok_threadcleanup();
   DEBUG_CLEAN;
   
} /* INTEGRATE */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int RosenbrockKrylov(double Y[], double Tstart, double Tend,
        double AbsTol[], double RelTol[],
        void (*ode_Fun)(double, double [], double []), 
  void (*ode_Jac)(double, double [], double []),
        double RPAR[], int IPAR[])
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
    Solves the system y'=F(t,y) using a Rosenbrock-Krylov method defined by:

     G = 1/(H*gamma[0]) - ode_Jac(t0,Y0)
     T_i = t0 + Alpha(i)*H
     Y_i = Y0 + \sum_{j=1}^{i-1} A(i,j)*K_j
     G * K_i = ode_Fun( T_i, Y_i ) + \sum_{j=1}^S C(i,j)/H * K_j +
         gamma(i)*dF/dT(t0, Y0)
     Y1 = Y0 + \sum_{j=1}^S M(j)*K_j 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
  *~~~>   INPUT ARGUMENTS: 
    
-     Y(NVAR)    = vector of initial conditions (at T=Tstart)
-    [Tstart,Tend]  = time range of integration
     (if Tstart>Tend the integration is performed backwards in time)  
-    RelTol, AbsTol = user precribed accuracy
-    void ode_Fun( T, Y, Ydot ) = ODE function, 
                       returns Ydot = Y' = F(T,Y) 
-    void ode_Fun( T, Y, Ydot ) = Jacobian of the ODE function,
                       returns Jcb = dF/dY 
-    IPAR(1:10)    = int inputs parameters
-    RPAR(1:10)    = real inputs parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  *~~~>     OUTPUT ARGUMENTS:  
     
-    Y(NVAR)    -> vector of final states (at T->Tend)
-    IPAR(11:20)   -> int output parameters
-    RPAR(11:20)   -> real output parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  *~~~>    RETURN VALUE (int):  

-    IERR       -> job status upon return
       - succes (positive value) or failure (negative value) -
           =  1 : Success
           = -1 : Improper value for maximal no of steps
           = -2 : Selected Rosenbrock method not implemented
           = -3 : Hmin/Hmax/Hstart must be positive
           = -4 : FacMin/FacMax/FacRej must be positive
           = -5 : Improper tolerance values
           = -6 : No of steps exceeds maximum bound
           = -7 : Step size too small
           = -8 : Matrix is repeatedly singular
           = -9 : Krylov subspace dimension must be positive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
  *~~~>     INPUT PARAMETERS:

    Note: For input parameters equal to zero the default values of the
       corresponding variables are used.

    IPAR[0]   = 1: F = F(y)   Independent of T (AUTONOMOUS)
        = 0: F = F(t,y) Depends on T (NON-AUTONOMOUS)
    IPAR[1]   = 0: AbsTol, RelTol are NVAR-dimensional vectors
        = 1:  AbsTol, RelTol are scalars
    IPAR[2]  -> maximum number of integration steps
        For IPAR[2]=0) the default value of 100000 is used

    IPAR[3]  -> selection of a particular Rosenbrock method
        = 0 :  default method is Rodas3
        = 1 :  method is  Ros2
        = 2 :  method is  Ros3 
        = 3 :  method is  Ros4 
        = 4 :  method is  Rodas3
        = 5:   method is  Rodas4
        
    IPAR[4]  -> dimension of the Krylov space used
        = 0:  Default value of 5 is used
        
    IPAR[5]  -> select how Jacobian-vector products are calculated
        = 0:   defaults to calculating the full Jacobian
        = 1:   uses a Jacobian-vector product function
        = 2:   uses a finite difference calculation

    IPAR[6]  -> toggle for fixed step size integration
        = 0:   uses variable step size controller
        = 1:   uses a fixed step size given by RPAR[7]

    RPAR[0]  -> Hmin, lower bound for the integration step size
          It is strongly recommended to keep Hmin = ZERO 
    RPAR[1]  -> Hmax, upper bound for the integration step size
    RPAR[2]  -> Hstart, starting value for the integration step size
          
    RPAR[3]  -> FacMin, lower bound on step decrease factor (default=0.2)
    RPAR[4]  -> FacMin,upper bound on step increase factor (default=6)
    RPAR[5]  -> FacRej, step decrease factor after multiple rejections
            (default=0.1)
    RPAR[6]  -> FacSafe, by which the new step is slightly smaller 
         than the predicted value  (default=0.9)
    RPAR[7]  -> Hfixed, step size for fixed step integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
  *~~~>     OUTPUT PARAMETERS:

    Note: each call to Rosenbrock adds the corrent no. of fcn calls
      to previous value of IPAR[10], and similar for the other params.
      Set IPAR(11:20) = 0 before call to avoid this accumulation.

    IPAR[10] = No. of function calls
    IPAR[11] = No. of jacobian calls
    IPAR[12] = No. of steps
    IPAR[13] = No. of accepted steps
    IPAR[14] = No. of rejected steps (except at the beginning)
    IPAR[15] = No. of LU decompositions
    IPAR[16] = No. of forward/backward substitutions
    IPAR[17] = No. of singular matrix decompositions
    IPAR[18] = No. of jacobian vector product calls

    RPAR[10]  -> Texit, the time corresponding to the 
            computed Y upon return
    RPAR[11]  -> Hexit, last accepted step before exit
    For multiple restarts, use Hexit as Hstart in the following run 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
{   

  /*~~~>  The method parameters    */   
   #define Smax 6
   int  Method, rok_S, Krylov_size;
   double rok_M[Smax], rok_E[Smax];
   double rok_A[Smax*(Smax-1)/2], rok_C[Smax*(Smax-1)/2];
   double rok_Alpha[Smax], rok_Gamma[Smax], rok_ELO;
   char rok_NewF[Smax], rok_Name[12];
  /*~~~>  Local variables    */  
   int Max_no_steps, IERR, i, UplimTol, FullJacobian;
   char Autonomous, VectorTol, FixedStep;
   double Roundoff,FacMin,FacMax,FacRej,FacSafe;
   double Hmin, Hmax, Hstart, Hfixed, Hexit, Texit;
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  /*~~~>  Initialize statistics */
   Nfun = IPAR[10];
   Njac = IPAR[11];
   Nstp = IPAR[12];
   Nacc = IPAR[13];
   Nrej = IPAR[14];
   Ndec = IPAR[15];
   Nsol = IPAR[16];
   Nsng = IPAR[17];
   Njv  = IPAR[18];
   
  /*~~~>  Autonomous or time dependent ODE. Default is time dependent. */
   Autonomous = !(IPAR[0] == 0);

  /*~~~>  For Scalar tolerances (IPAR[1] != 0)  the code uses AbsTol[0] and RelTol[0]
!   For Vector tolerances (IPAR[1] == 0) the code uses AbsTol(1:NVAR) and RelTol(1:NVAR) */
   if (IPAR[1] == 0) {
      VectorTol = 1; UplimTol  = NVAR;
   } else { 
      VectorTol = 0; UplimTol  = 1;
   } /* end if */
   
  /*~~~>   The maximum number of steps admitted */
   if (IPAR[2] == 0)  
      Max_no_steps = 100000;
   else                
      Max_no_steps=IPAR[2];
   if (Max_no_steps < 0) { 
      printf("\n User-selected max no. of steps: IPAR[2]=%d\n",IPAR[2]);
      return rok_ErrorMsg(-1,Tstart,ZERO);
   } /* end if */

  /*~~~>  The particular Rosenbrock method chosen */
   if (IPAR[3] == 0)  
       Method = 3;
   else                
       Method = IPAR[3];
   if ( (IPAR[3] < 1) || (IPAR[3] > 5) ){  
      printf("\n User-selected Rosenbrock-Krylov method: IPAR[3]=%d\n",IPAR[3]);
      return rok_ErrorMsg(-2,Tstart,ZERO);
   } /* end if */
   
  /*~~~>  The dimension of the Krylov space to use */
   if (IPAR[4] == 0)
      Krylov_size = 4;
   else
      Krylov_size = IPAR[4];
   if ( Krylov_size < 0 || Krylov_size > NVAR) {
      printf("\n User-selected Krylov subspace dimensions: IPAR[4]=%d\n",IPAR[4]);
      return rok_ErrorMsg(-9,Tstart,ZERO);
   } /* end if */
   
  /*~~~>  Full Jacobian or finite difference approximation */
   FullJacobian = IPAR[5];

  /*~~~>  Toggle fixed step size */
   FixedStep = !(IPAR[6] == 0);
   
  /*~~~>  Unit Roundoff (1+Roundoff>1)   */
   Roundoff = rok_dlamch('E');

  /*~~~>  Lower bound on the step size: (positive value) */
   Hmin = RPAR[0];
   if (RPAR[0] < ZERO) {	 
      printf("\n User-selected Hmin: RPAR[0]=%e\n", RPAR[0]);
      return rok_ErrorMsg(-3,Tstart,ZERO);
   } /* end if */
  /*~~~>  Upper bound on the step size: (positive value) */
   if (RPAR[1] == ZERO)  
      Hmax = ABS(Tend-Tstart);
   else   
      Hmax = MIN(ABS(RPAR[1]),ABS(Tend-Tstart));
   if (RPAR[1] < ZERO) {	 
      printf("\n User-selected Hmax: RPAR[1]=%e\n", RPAR[1]);
      return rok_ErrorMsg(-3,Tstart,ZERO);
   } /* end if */
  /*~~~>  Starting step size: (positive value) */
   if (RPAR[2] == ZERO) 
      Hstart = MAX(Hmin,DeltaMin);
   else
      Hstart = MIN(ABS(RPAR[2]),ABS(Tend-Tstart));
   if (RPAR[2] < ZERO) {	 
      printf("\n User-selected Hstart: RPAR[2]=%e\n", RPAR[2]);
      return rok_ErrorMsg(-3,Tstart,ZERO);
   } /* end if */
  /*~~~>  Step size can be changed s.t.  FacMin < Hnew/Hexit < FacMax  */
   if (RPAR[3] == ZERO)
      FacMin = (double)0.2;
   else
      FacMin = RPAR[3];
   if (RPAR[3] < ZERO) {	 
      printf("\n User-selected FacMin: RPAR[3]=%e\n", RPAR[3]);
      return rok_ErrorMsg(-4,Tstart,ZERO);
   } /* end if */
   if (RPAR[4] == ZERO) 
      FacMax = (double)6.0;
   else
      FacMax = RPAR[4];
   if (RPAR[4] < ZERO) {	 
      printf("\n User-selected FacMax: RPAR[4]=%e\n", RPAR[4]);
      return rok_ErrorMsg(-4,Tstart,ZERO);
   } /* end if */
  /*~~~>   FacRej: Factor to decrease step after 2 succesive rejections */
   if (RPAR[5] == ZERO) 
      FacRej = (double)0.1;
   else
      FacRej = RPAR[5];
   if (RPAR[5] < ZERO) {	 
      printf("\n User-selected FacRej: RPAR[5]=%e\n", RPAR[5]);
      return rok_ErrorMsg(-4,Tstart,ZERO);
   } /* end if */
  /*~~~>   FacSafe: Safety Factor in the computation of new step size */
   if (RPAR[6] == ZERO) 
      FacSafe = (double)0.9;
   else
      FacSafe = RPAR[6];
   if (RPAR[6] < ZERO) {	 
      printf("\n User-selected FacSafe: RPAR[6]=%e\n", RPAR[6]);
      return rok_ErrorMsg(-4,Tstart,ZERO);
   } /* end if */
  /*~~~>  Hfixed:  check the fixed time step */
   if (IPAR[6] && RPAR[7] == ZERO) {
      Hfixed = Hstart;
   } else {
      Hfixed = RPAR[7];
   }
   if (IPAR[2] == 0 && IPAR[6])
      Max_no_steps = 1 + (int)(ABS(Tend-Tstart)/Hfixed);
  /*~~~>  Check if tolerances are reasonable */
    for (i = 0; i < UplimTol; i++) {
      if ( (AbsTol[i] <= ZERO)  ||  (RelTol[i] <= 10.0*Roundoff)
          ||  (RelTol[i] >= ONE) ) {
        printf("\n  AbsTol[%d] = %e\n",i,AbsTol[i]);
        printf("\n  RelTol[%d] = %e\n",i,RelTol[i]);
        return rok_ErrorMsg(-5,Tstart,ZERO);
      } /* end if */
    } /* for */
     
 
  /*~~~>   Initialize the particular Rosenbrock method */
   switch (Method) {
     case 1:
       Rok4a(&rok_S, rok_A, rok_C, rok_M, rok_E, 
         rok_Alpha, rok_Gamma, rok_NewF, &rok_ELO, rok_Name);
       break;
     case 2:
        Rok4b(&rok_S, rok_A, rok_C, rok_M, rok_E, 
         rok_Alpha, rok_Gamma, rok_NewF, &rok_ELO, rok_Name);
        break;
     case 3:
        Rok4p(&rok_S, rok_A, rok_C, rok_M, rok_E, 
         rok_Alpha, rok_Gamma, rok_NewF, &rok_ELO, rok_Name);
        break;
     default:
       printf("\n Unknown Rosenbrock-Krylov method: IPAR[3]= %d", Method);
       return rok_ErrorMsg(-2,Tstart,ZERO); 
   } /* end switch */

  /*~~~>  Rosenbrock method   */
   IERR = RosenbrockKrylovIntegrator( Y,Tstart,Tend,
        AbsTol, RelTol,
        ode_Fun,ode_Jac ,
      /*  Rosenbrock method coefficients  */     
        rok_S, rok_M, rok_E, rok_A, rok_C, 
        rok_Alpha, rok_Gamma, rok_ELO, rok_NewF,
      /*  Integration parameters */ 
        Autonomous, FixedStep, FullJacobian, VectorTol,
        Max_no_steps, Krylov_size,
        Roundoff, Hmin, Hmax, Hstart, Hfixed,
        FacMin, FacMax, FacRej, FacSafe, 
      /* Output parameters */ 
  &Texit, &Hexit );


  /*~~~>  Collect run statistics */
   IPAR[10] = Nfun;
   IPAR[11] = Njac;
   IPAR[12] = Nstp;
   IPAR[13] = Nacc;
   IPAR[14] = Nrej;
   IPAR[15] = Ndec;
   IPAR[16] = Nsol;
   IPAR[17] = Nsng;
   IPAR[18] = Njv;
  /*~~~> Last T and H */
   RPAR[10] = Texit;
   RPAR[11] = Hexit;    
   
   return IERR;
   
} /* Rosenbrock */

   
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int RosenbrockKrylovIntegrator(
  /*~~~> Input: the initial condition at Tstart; Output: the solution at T */  
     double Y[],
  /*~~~> Input: integration interval */   
     double Tstart, double Tend ,     
  /*~~~> Input: tolerances  */        
     double  AbsTol[], double  RelTol[],
  /*~~~> Input: ode function and its Jacobian */      
     void (*ode_Fun)(double, double [], double []), 
     void (*ode_Jac)(double, double [], double []) ,
  /*~~~> Input: The Rosenbrock method parameters */   
     int rok_S,
     double rok_M[], double rok_E[], 
     double rok_A[], double rok_C[],
     double rok_Alpha[],double  rok_Gamma[],
     double rok_ELO, char rok_NewF[],
  /*~~~> Input: integration parameters  */     
     char Autonomous, char FixedStep, int FullJacobian, char VectorTol,
     int Max_no_steps, int Krylov_size, 
     double Roundoff, double Hmin, double Hmax, double Hstart, double Hfixed,
     double FacMin, double FacMax, double FacRej, double FacSafe, 
  /*~~~> Output: time at which the solution is returned (T=Tend  if success)   
             and last accepted step  */     
     double *Texit, double *Hexit ) 
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      Template for the implementation of a generic Rosenbrock method 
      defined by rok_S (no of stages) and coefficients rok_{A,C,M,E,Alpha,Gamma}
      
      returned value: IERR, indicator of success (if positive) 
                                      or failure (if negative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{   
   double *Ynew, *Fcn0, *Fcn, *dFdT,
      *Jac0, *Vtrans, *Lambdatrans,
      *Imhgh, *Hes, *Phi,
      *tempvec;
   double *K;   
   double H, T, Hnew, HG, Fac, Tau; 
   double Err, *Yerr;
   int *Pivot, Direction, ioffset, koffset, j, istage;
   char RejectLastH, RejectMoreH;
   
   struct timeval t1, t2;
   double arnoldi_time = 0.0;
   int    arnoldi_count = 0;
   double solve_time = 0.0;
   int    solve_count = 0;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
   Ynew = (double *)rok_vector(NVAR, sizeof(double));
   Yerr = (double *)rok_vector(NVAR, sizeof(double));
   Fcn0 = (double *)rok_vector(NVAR+1, sizeof(double));
   Fcn  = (double *)rok_vector(NVAR+1, sizeof(double));
   dFdT = (double *)rok_vector(NVAR, sizeof(double));
   Vtrans = (double *)rok_matrix(NVAR+1, Krylov_size, sizeof(double));
   Lambdatrans = (double *)rok_matrix(Krylov_size, rok_S, sizeof(double));
   Imhgh = (double *)rok_matrix(Krylov_size, Krylov_size, sizeof(double));
   Hes = (double *)rok_matrix(Krylov_size, Krylov_size, sizeof(double));
   Phi = (double *)rok_vector(Krylov_size, sizeof(double));
   tempvec = (double *)rok_vector(Krylov_size, sizeof(double));
   K = (double *)rok_matrix(NVAR, rok_S, sizeof(double));
   Pivot = (int *)rok_vector(Krylov_size, sizeof(int));
   
   if (FullJacobian == 0)
      Jac0 = (double *)rok_vector(NSPARSE, sizeof(double));
   else
      Jac0 = NULL;
   
   swe_fun_init();
   swe_fun_d_init();
   jv_init();
   
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

   DEBUG_PRINT("\n\nStart Rok integration.\n");
   DEBUG_PRINT("FullJacobian = %d\n\n", FullJacobian);

#ifdef DEBUG
//    DEBUG_PRINT("\nY:\n");
//    for (i = 0; i < NVAR; i++) {
//       DEBUG_PRINT("  %12.5f ", Y[i]);
//    }
//    DEBUG_PRINT("\n\n");
#endif
   
  /*~~~>  INITIAL PREPARATIONS  */
   T = Tstart;
   *Hexit = 0.0;
   if (!FixedStep)
      H = MIN(Hstart,Hmax);
   else
      H = MIN(Hfixed,Hmax);
   if (ABS(H) <= 10.0*Roundoff) 
        H = DeltaMin;
   
   if (Tend >= Tstart) {
     Direction = +1;
   } else {
     Direction = -1;
   } /* end if */

   RejectLastH = 0; RejectMoreH = 0;
   
   DEBUG_PRINT("Start time loop\n");
   
  /*~~~> Time loop begins below  */ 
   while ( ( (Direction > 0) && ((T-Tend)+Roundoff <= ZERO) )
       || ( (Direction < 0) && ((Tend-T)+Roundoff <= ZERO) ) ) { 
          
      if ( Nstp > Max_no_steps ) { /* Too many steps */
         *Texit = T;
         return rok_ErrorMsg(-6,T,H);
      }
      if ( ((T+0.1*H) == T) || (H <= Roundoff) ) { /* Step size too small */
          *Texit = T;
          return rok_ErrorMsg(-7,T,H);
      }   
      
     /*~~~>  Limit H if necessary to avoid going beyond Tend   */  
      *Hexit = H;
      H = MIN(H,ABS(Tend-T));
      
      DEBUG_PRINT("H = %25.20e\n", H);

     /*~~~>  Save current solution   */
      if (SAVESTATE) {
         rok_save_vector(NVAR, Y, 0);
         rok_save_vector(1, &T, 1);
      }

     /*~~~>   Compute the function at current time  */
      (*ode_Fun)(T,Y,Fcn0);

     /*~~~>  Compute the function derivative with respect to T  */
      if (!Autonomous) 
          rok_FunTimeDerivative ( T, Roundoff, Y, Fcn0, ode_Fun, dFdT );
      
#ifdef DEBUG
      if (T == Tstart) {
         DEBUG_PRINT("\nFcn:\n");
         for (int i = 0; i < NVAR; i++) {
            DEBUG_PRINT("  %18.17e", Fcn0[i]);
         }
         DEBUG_PRINT("\n\n");
         
         DEBUG_PRINT("\ndFdT:\n");
         for (int i = 0; i < NVAR; i++) {
            DEBUG_PRINT("  %18.17e", dFdT[i]);
         }
         DEBUG_PRINT("\n\n");
      }
#endif
      
     /*~~~>   Compute the Jacobian at current time  */
      if (FullJacobian == 0)
         (*ode_Jac)(T,Y,Jac0);
      
#ifdef DEBUG
      if (T == Tstart) {
         swe_jac_print(Jac0);
      }
#endif
            
      DEBUG_PRINT("Start Arnoldi iteration.\n");
      DEBUG_PRINT("Krylov size = %d\n", Krylov_size);
      
      gettimeofday(&t1, NULL);
      
     /*~~~>  Compute Krylov basis and upper Hessenberg matrix */
      rok_ArnoldiMethod(Krylov_size, Fcn0, FullJacobian, Autonomous, T,
                        Y, Jac0, dFdT, ode_Fun, Hes, Vtrans);
      
      gettimeofday(&t2, NULL);
      arnoldi_time += 1000.0 * (t2.tv_sec - t1.tv_sec) + (1.0/1000.0) * (t2.tv_usec - t1.tv_usec);
      arnoldi_count++;
      
      DEBUG_PRINT("Finish Arnoldi.\n");

#ifdef DEBUG
      if (T == Tstart) { 
         DEBUG_PRINT("\nHessenberg:\n");
         for (int i = 0; i < Krylov_size; i++) {
            for (int j = 0; j < Krylov_size; j++) {
               DEBUG_PRINT("  %18.17e", Hes[i*Krylov_size+j]);
            }
            DEBUG_PRINT("\n");
         }
         DEBUG_PRINT("\nV:\n");
         for (int j = 0; j < NVAR+1; j++) {
            for (int i = 0; i < Krylov_size; i++) {
               DEBUG_PRINT("  %18.17e", Vtrans[i*(NVAR+1)+j]);
            }
            DEBUG_PRINT("\n");
         }
         DEBUG_PRINT("\n");
      }
#endif
      
     /*~~~>  Repeat step calculation until current step accepted  */
      while (1) { /* WHILE STEP NOT ACCEPTED */
         
         DEBUG_PRINT("Start LU decomposition.\n");
        /*~~~>  Construct and LU decompose the left-hand side matrix. */
         if( rok_PrepareMatrix( &H, Direction, Krylov_size, rok_Gamma[0],
               Hes, Imhgh, Pivot) ) { /* More than 5 consecutive failed decompositions */
            *Texit = T;
            return rok_ErrorMsg(-8,T,H);
         }

         DEBUG_PRINT("Start stage calculations.\n");
        /*~~~>   Compute the stages  */
         for (istage = 1; istage <= rok_S; istage++) {
            
            /* Current istage offset. Current istage vector is K[ioffset:ioffset+NVAR-1] */
            ioffset = NVAR*(istage-1);
            /* Current istage offset in Krylov subspace. Lambdatrans[koffset:koffset+Krylov_size-1] */
            koffset = Krylov_size*(istage-1);
         
           /*~~~>  Get a new function evaluation. */
            if ( istage == 1 ) {
               /* For the 1st istage the function has been computed previously */
               rok_dcopy(NVAR, Fcn0, 1, Fcn, 1);
            } else if ( rok_NewF[istage-1] ) { 
               /* istage>1 and a new function evaluation is needed at current istage */
               rok_dcopy(NVAR, Y, 1, Ynew, 1);
               for (j = 1; j <= istage-1; j++)
                  rok_daxpy(NVAR, rok_A[(istage-1)*(istage-2)/2 + j-1],
                     &K[NVAR*(j-1)], 1, Ynew, 1); 
               Tau = T + rok_Alpha[istage-1]*Direction*H;
               (*ode_Fun)(Tau,Ynew,Fcn);
            } /* end if istage */
         
            if (!Autonomous)
               Fcn[NVAR] = ONE;
            else
               Fcn[NVAR] = ZERO;
            
#ifdef DEBUG
            DEBUG_PRINT("\nNew Fcn:\n");
            for (int i = 0; i < NVAR+1; i++) {
               DEBUG_PRINT("  %25.19e", Fcn[i]);
            }
            DEBUG_PRINT("\n\n");
#endif
           
           /*~~~>  Compute Phi <- Vtrans*Fcn */
            rok_dgemv('n', Krylov_size, NVAR+1, ONE, Vtrans, NVAR+1, Fcn, 1, ZERO, Phi, 1);
            
#ifdef DEBUG
            DEBUG_PRINT("\nPhi:\n");
            for (int i = 0; i < Krylov_size; i++) {
               DEBUG_PRINT("  %25.19e", Phi[i]);
            }
            DEBUG_PRINT("\n\n");
#endif
            
            gettimeofday(&t1, NULL);
            
           /*~~~>  Compute right-hand side vector  */
            rok_dscal(Krylov_size, ZERO, tempvec, 1);
            for (j = 1; j <= istage-1; j++) {
               HG = rok_C[(istage-1)*(istage-2)/2 + j-1] * (Direction*H);
               rok_daxpy(Krylov_size, HG, &Lambdatrans[(j-1)*Krylov_size], 1, tempvec, 1);
            }
            rok_dgemv('n', Krylov_size, Krylov_size, ONE, Hes, Krylov_size, tempvec, 1, ZERO, &Lambdatrans[koffset], 1);
            rok_daxpy(Krylov_size, H, Phi, 1, &Lambdatrans[koffset], 1);
           
#ifdef DEBUG
            DEBUG_PRINT("\nRHS:\n");
            for (int i = 0; i < Krylov_size; i++) {
               DEBUG_PRINT("  %25.19e", *(&Lambdatrans[koffset]+i));
            }
            DEBUG_PRINT("\n\n");
#endif
 
           /*~~~>  Solve linear system for Lamda_i in the Krylov subspace. */
            SolveTemplate(Krylov_size, Imhgh, Pivot, &Lambdatrans[koffset]);

            gettimeofday(&t2, NULL);
            solve_time += 1000.0 * (t2.tv_sec - t1.tv_sec) + (1.0/1000.0) * (t2.tv_usec - t1.tv_usec);
            solve_count++;
            
#ifdef DEBUG
            DEBUG_PRINT("\nLambda(%d):\n", istage);
            for (int i = 0; i < Krylov_size; i++) {
               DEBUG_PRINT("  %25.19e", *(&Lambdatrans[koffset]+i));
            }
            DEBUG_PRINT("\n\n");
#endif
           
           /*~~~>  Recover K[ioffset] from the Krylov subspace. */
            /* K_i <- V*Lamda_i + h(F_i - V*Phi_i) */
            rok_dcopy(NVAR, Fcn, 1, &K[ioffset], 1);
            rok_dgemv('y', Krylov_size, NVAR, -ONE, Vtrans, NVAR+1, Phi, 1, ONE, &K[ioffset], 1);
            rok_dgemv('y', Krylov_size, NVAR, ONE, Vtrans, NVAR+1, &Lambdatrans[koffset], 1, H, &K[ioffset], 1);
            
            /* Refactored method of recovering K[ioffset]... cheaper, but numerically? */
            /* K_i <- -h*V((-1/h)Lamda_i + Phi_i) + h*F_i */
//             rok_dcopy(NVAR, Fcn, 1, &K[ioffset], 1);
//             rok_daxpy(Krylov_size, -ONE/H, &Lambdatrans[koffset], 1, Phi, 1);
//             rok_dgemv('y', Krylov_size, NVAR, -H, Vtrans, NVAR+1, Phi, 1, H, &K[ioffset], 1);
            
         } /* for istage */	    
            

        /*~~~>  Compute the new solution   */
         rok_dcopy(NVAR, Y, 1, Ynew, 1);
         for (j=1; j<=rok_S; j++)
            rok_daxpy(NVAR, rok_M[j-1], &K[NVAR*(j-1)], 1, Ynew, 1);

        /*~~~>  Compute the error estimation   */
         rok_dscal(NVAR, ZERO, Yerr, 1);
         for (j=1; j<=rok_S; j++)    
            rok_daxpy(NVAR, rok_E[j-1], &K[NVAR*(j-1)], 1, Yerr, 1);
         Err = rok_ErrorNorm ( Y, Ynew, Yerr, AbsTol, RelTol, VectorTol );
         y_err += Err;
         y_count++;
         y_maxerr = MAX(Err, y_maxerr);
         rok_record_err(y_err/y_count, 2);
         rok_record_err(y_maxerr, 3);
         
         DEBUG_PRINT("error norm: %25.20e\n", Err);

        /*~~~> New step size is bounded by FacMin <= Hnew/H <= FacMax  */
         Fac  = MIN(FacMax,MAX(FacMin,FacSafe/pow(Err,ONE/rok_ELO)));
         Hnew = H*Fac;  

        /*~~~>  Check the error magnitude and adjust step size  */
         Nstp++;
         if (FixedStep) {
            rok_dcopy(NVAR, Ynew, 1, Y, 1);
            T += Direction*H;
            H = Hfixed;
            //printf("*");
            //fflush(stdout);
            break; /* EXIT LOOP: WHILE STEP NOT ACCEPTED */
         } else if ( (Err <= ONE) || (H <= Hmin) ) {    /*~~~> Accept step  */
            Nacc++;
            rok_dcopy(NVAR, Ynew, 1, Y, 1);
            T += Direction*H;
            Hnew = MAX(Hmin,MIN(Hnew,Hmax));
            /* No step size increase after a rejected step  */
            if (RejectLastH) 
               Hnew = MIN(Hnew,H); 
            RejectLastH = 0; RejectMoreH = 0;
            H = Hnew;
            DEBUG_PRINT("Step accepted - exit loop.\n");
            //printf("*");
            //fflush(stdout);
            break; /* EXIT THE LOOP: WHILE STEP NOT ACCEPTED */
         } else {             /*~~~> Reject step  */
            if (Nacc >= 1) 
               Nrej++;    
            if (RejectMoreH) 
               Hnew=H*FacRej;   
            RejectMoreH = RejectLastH; RejectLastH = 1;
            H = Hnew;
            DEBUG_PRINT("Step rejected - repeat loop.\n");
         } /* end if Err <= 1 */

      } /* while LOOP: WHILE STEP NOT ACCEPTED */

   } /* while: time loop */   
  
  /*~~~> Save final solution. */
   if (SAVESTATE) {
      rok_save_vector(NVAR, Y, 0);
      rok_save_vector(1, &T, 1);
   }
 
  /*~~~> Cleanup memory use. */
   rok_freevector(Ynew);
   rok_freevector(Yerr);
   rok_freevector(Fcn0);
   rok_freevector(Fcn);
   rok_freevector(dFdT);
   rok_freematrix(Vtrans);
   rok_freematrix(Lambdatrans);
   rok_freematrix(Imhgh);
   rok_freematrix(Hes);
   rok_freevector(Phi);
   rok_freevector(tempvec);
   rok_freematrix(K);
//   free(Pivot);
   
   if (FullJacobian == 0)
      rok_freevector(Jac0);
   
   swe_fun_cleanup();
   swe_fun_d_cleanup();
   
   rok_record_time(arnoldi_time/arnoldi_count, TIMER_ARNOLDI);
   rok_record_time(solve_time/solve_count, TIMER_SOLVER);
   
   printf("\n\n");
   
  /*~~~> The integration was successful */
   *Texit = T;
   return 1;    

}  /* RosenbrockIntegrator */
 

   
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
double rok_ErrorNorm ( 
  /*~~~> Input arguments */  
     double Y[], double Ynew[], double Yerr[], 
     double AbsTol[], double RelTol[], 
     char VectorTol )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Computes and returns the "scaled norm" of the error vector Yerr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{   	 
  /*~~~> Local variables */     
   double Err, Scale, Ymax;   
   int i;
   
   Err = ZERO;
   for (i=0; i<NVAR; i++) {
     Ymax = MAX(ABS(Y[i]),ABS(Ynew[i]));
     if (VectorTol) {
       Scale = AbsTol[i]+RelTol[i]*Ymax;
     } else {
       Scale = AbsTol[0]+RelTol[0]*Ymax;
     } /* end if */
     Err = Err + (Yerr[i]*Yerr[i])/(Scale*Scale);
   } /* for i */
   Err  = SQRT(Err/(double)NVAR);

   return Err;
   
} /* rok_ErrorNorm */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_FunTimeDerivative ( 
    /*~~~> Input arguments: */ 
        double T, double Roundoff, 
        double Y[], double Fcn0[], 
  void (*ode_Fun)(double, double [], double []), 
    /*~~~> Output arguments: */ 
        double dFdT[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The time partial derivative of the function by finite differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{
  /*~~~> Local variables */     
   double Delta;    
   
   Delta = SQRT(Roundoff)*MAX(DeltaMin,ABS(T));
   (*ode_Fun)(T+Delta,Y,dFdT);
   rok_daxpy(NVAR, (-ONE), Fcn0, 1, dFdT, 1);
   rok_dscal(NVAR, (ONE/Delta), dFdT, 1);

}  /*  rok_FunTimeDerivative */

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
double rok_VectorNorm(
   /*~~~> Input arguments: */
       int length, double vector[] ) 
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Returns the simple 2-norm of a vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   return SQRT(rok_ddot(length, vector, 1, vector, 1));
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_JacVectorProd(
   /*~~~> Input arguments: */
     int FullJacobian, double T, double norm_Y,
     double Y[], double Fcn0[], double Jac[], double vector[], 
     void (*ode_Fun)(double, double [], double []),
   /*~~~> Output arguments: */
     double result[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Computes the product of the Jacobian matrix and a vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   if (FullJacobian == 0) {
      swe_jac_vec(Jac, vector, result);
   } else if (FullJacobian == 1) {
      swe_fun_d(T, Y, vector, result);
   } else {
     /*~~~> Approximate Jacobian-vector product. */
      double *tempvec, epsilon;
      
      tempvec = (double *)rok_vector(NVAR, sizeof(double));
      
      epsilon = 1e-7 * MAX(1e-5, norm_Y);
      //epsilon = 1e-4;
      DEBUG_PRINT("JV delta = %25.20e\n", delta);
      rok_dcopy(NVAR, Y, 1, tempvec, 1);
      rok_daxpy(NVAR, epsilon, vector, 1, tempvec, 1);
      (*ode_Fun)(T, tempvec, result);
      
#ifdef DEBUG
      {
         int i;
         DEBUG_PRINT("JV Y+epsilonV:\n");
         for (i = 0; i < NVAR; i++) {
            DEBUG_PRINT("  %18.17e", tempvec[i]);
         }
         DEBUG_PRINT("\n\n");
         
         DEBUG_PRINT("JV Fcn:\n");
         for (i = 0; i < NVAR+1; i++) {
            DEBUG_PRINT("  %18.17e", result[i]);
         }
         DEBUG_PRINT("\n\n");
      }
#endif

      rok_daxpy(NVAR, -ONE, Fcn0, 1, result, 1);
      rok_dscal(NVAR, ONE/epsilon, result, 1);
      
      rok_freevector(tempvec);

      // Record epsilon stats
      eps_max = MAX(eps_max, epsilon);
      eps_min = MIN(eps_min, epsilon);
      eps_ave += epsilon;
      eps_count++;
      rok_record_err(eps_min, 4);
      rok_record_err(eps_ave/eps_count, 5);
      rok_record_err(eps_max, 6);

   }
   // Measure the error against analytical Jacobian-vector product:
   double * analytical = (double *)rok_vector(NVAR, sizeof(double));
   double err = 0.0;
   swe_fun_d(T, Y, vector, analytical);
   rok_daxpy(NVAR, -ONE, result, 1, analytical, 1);
   err = rok_VectorNorm(NVAR, analytical);
   jv_err += err;
   jv_count++;
   jv_maxerr = MAX(err, jv_maxerr);
//   printf("(%f) %e, %e, %e\n", T, err, jv_err, jv_maxerr);
   rok_record_err(jv_err/jv_count, 0);
   rok_record_err(jv_maxerr, 1);
   rok_freevector(analytical);
   

   Njv++;

#ifdef DEBUG
   DEBUG_PRINT("Jac-vec product:\n");
   for (int i = 0; i < NVAR; i++) {
      DEBUG_PRINT("  %25.19e", result[i]);
   }
   DEBUG_PRINT("\n\n");
#endif
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void rok_ArnoldiMethod (
   /*~~~> Input arguments: */
      int Krylov_size, double Fcn0[], 
   /*~~~> Input arguments for computing Jacobian-vector products: */
     int FullJacobian, char Autonomous, double T,
     double Y[], double Jac0[], double dFdT[], 
     void (*ode_Fun)(double, double [], double []),
   /*~~~> Output arguments: */
     double Hes[], double Vtrans[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Finds the Krylov subspace and upper Hessenberg matrix using modified Arnoldi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
   int i, j;
   double beta, tau, rho, normY;
   double *zeta, xi;
   
   zeta = (double *)rok_vector(NVAR, sizeof(double));
   
#define V(I)   ( Vtrans[((I)-1)*(NVAR+1)] )
#define w(I)   ( Vtrans[((I)-1)*(NVAR+1) + NVAR] )
#define H(I,J) ( Hes[((I)-1) * Krylov_size + ((J)-1)] )
   
  /*~~~>  Initialize starting vectors. */
   normY = rok_VectorNorm(NVAR, Y);
   beta = SQRT(rok_ddot(NVAR, Fcn0, 1, Fcn0, 1) + ONE);
   rok_dcopy(NVAR, Fcn0, 1, &V(1), 1);
   rok_dscal(NVAR, ONE/beta, &V(1), 1);
   if (!Autonomous)
      w(1) = ONE/beta;
   else
      w(1) = ZERO;
   
  /*~~~>  Begin Arnoldi iteration. */
   for (i = 1; i <= Krylov_size; i++) {
      rok_JacVectorProd(FullJacobian, T, normY, Y, Fcn0, Jac0, &V(i), ode_Fun, zeta);
//      rok_dgemv(C'n', NVAR, NVAR, ONE, Jac0, NVAR, &V(i), 1, ZERO, zeta, 1);
      rok_daxpy(NVAR, w(i), dFdT, 1, zeta, 1); // zeta <- J*v_i + f'*w_i
      xi = ZERO;                                 // xi <- 0
      tau = rok_VectorNorm(NVAR, zeta);          // tau <- ||zeta||
      
      for (j = 1; j <= i; j++) {
         H(j,i) = rok_ddot(NVAR, zeta, 1, &V(j), 1);
         H(j,i) += xi * w(j);                      // H(j,i) <- (zeta,v_j) + xi*w_j
         rok_daxpy(NVAR, (-H(j,i)), &V(j), 1, zeta, 1);  // zeta <- zeta - H(j,i) * v_j
         xi = xi - H(j,i) * w(j);                  // xi <- xi - H(j,i) * w_j
      }
      
      beta = SQRT(rok_ddot(NVAR, zeta, 1, zeta, 1) + xi*xi); // beta <- ||zeta; xi||
      
      if (beta/tau <= 0.25) {
         for (j = 1; j <= i; j++) {
            rho = rok_ddot(NVAR, zeta, 1, &V(j), 1);
            rho += xi * w(j);
            rok_daxpy(NVAR, (-rho), &V(j), 1, zeta, 1);
            xi = xi - rho * w(j);
            H(j,i) += rho;
         }
         beta = SQRT(rok_ddot(NVAR, zeta, 1, zeta, 1) + xi*xi);
      }
      
      if (i < Krylov_size) {
         H(i+1,i) = SQRT(rok_ddot(NVAR, zeta, 1, zeta, 1) + xi*xi);
         rok_dcopy(NVAR, zeta, 1, &V(i+1), 1);
         rok_dscal(NVAR, ONE/H(i+1,i), &V(i+1), 1);
         w(i+1) = xi/H(i+1,i);
      }
   }
   
   rok_freevector(zeta);
   
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
char rok_PrepareMatrix (
       /* Inout argument: (step size is decreased when LU fails) */  
           double* H, 
       /* Input arguments: */
           int Direction, int Krylov_size, double gam, double Hes[], 
       /* Output arguments: */
           double Imhgh[], int Pivot[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Prepares the LHS matrix for stage calculations
  1.  Construct Imhgh <- (I - H*gamma*Hes)
  2.  Repeat LU decomposition of Imhgh until successful.
       -half the step size if LU decomposition fails and retry
       -exit after 5 consecutive fails

  Return value:       Singular (true=1=failed_LU or false=0=successful_LU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{   
  /*~~~> Local variables */     
   int i, ising, Nconsecutive;
   int size = Krylov_size*Krylov_size;
   double gamh;
   
   Nconsecutive = 0;
   
   while (1) {  /* while Singular */
     /*~~~>    Construct Imhgh <- (I - H*gamma*Hes) */
      rok_dcopy(size, Hes, 1, Imhgh, 1);
      gamh = -Direction*(*H)*gam;
      rok_dscal(size, gamh, Imhgh, 1);
      for (i = 0; i < Krylov_size; i++) {
         Imhgh[i*Krylov_size + i] += ONE;
      } /* for i */
      
#ifdef DEBUG
      DEBUG_PRINT("\nImhgh:\n");
      for (i = 0; i < Krylov_size; i++) {
         int j;
         for (j = 0; j < Krylov_size; j++) {
            DEBUG_PRINT("  %25.18e", Imhgh[i*Krylov_size+j]);
         }
         DEBUG_PRINT("\n");
      }
      DEBUG_PRINT("\n");
#endif
      
     /*~~~>    Compute LU decomposition  */
      DecompTemplate( Krylov_size, Imhgh, Pivot, &ising );
//      ising = 0;
      if (ising == 0) {
        /*~~~>    if successful done  */
#ifdef DEBUG
         DEBUG_PRINT("\nImhgh LU decomposed:\n");
         for (i = 0; i < Krylov_size; i++) {
            int j;
            for (j = 0; j < Krylov_size; j++) {
               DEBUG_PRINT("  %25.18e", Imhgh[i*Krylov_size+j]);
            }
            DEBUG_PRINT("\n");
         }
         DEBUG_PRINT("\n");
#endif
         return 0;  /* Singular = false */
      } else { /* ising .ne. 0 */
        /*~~~>    if unsuccessful half the step size; if 5 consecutive fails return */
         Nsng++; Nconsecutive++;
         printf("\nWarning: LU Decomposition returned ising = %d\n",ising);
         if (Nconsecutive <= 5) { /* Less than 5 consecutive failed LUs */
            *H = (*H)*HALF;
         } else {                  /* More than 5 consecutive failed LUs */
            return 1; /* Singular = true */
         } /* end if  Nconsecutive */
      } /* end if ising */
   
   } /* while Singular */

}  /*  rok_PrepareMatrix */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
int rok_ErrorMsg(int Code, double T, double H)
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                  Handles all error messages and returns IERR = error Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{   
   printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"); 
   printf("\nForced exit from Rosenbrock-K due to the following error:\n"); 
     
   switch (Code) {
   case -1:   
      printf("--> Improper value for maximal no of steps"); break;
   case -2:   
      printf("--> Selected Rosenbrock method not implemented"); break;
   case -3:   
      printf("--> Hmin/Hmax/Hstart must be positive"); break;
   case -4:   
      printf("--> FacMin/FacMax/FacRej must be positive"); break;
   case -5:
      printf("--> Improper tolerance values"); break;
   case -6:
      printf("--> No of steps exceeds maximum bound"); break;
   case -7:
      printf("--> Step size too small (T + H/10 = T) or H < Roundoff"); break;
   case -8:   
      printf("--> Matrix is repeatedly singular"); break;
   case -9:
      printf("--> Krylov subspace dimension must be positive"); break;
   default:
      printf("Unknown Error code: %d ",Code); 
   } /* end switch */
   
   printf("\n   Time = %15.7e,  H = %15.7e",T,H);
   printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
     
   return Code;  
     
}  /* rok_ErrorMsg  */
      

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
void Rok4a ( int *rok_S, double rok_A[], double rok_C[], 
           double rok_M[], double rok_E[], 
     double rok_Alpha[], double rok_Gamma[], 
     char rok_NewF[], double *rok_ELO, char* rok_Name )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
             4 stage, order 4, L-stable Rosenbrock-Krylov method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{   
  /*~~~> Name of the method */
    strcpy(rok_Name, "ROK-4a");  
        
  /*~~~> Number of stages */
    *rok_S = 4;
   
  /*~~~> The coefficient matrices A and C are strictly lower triangular.
    The lower triangular (subdiagonal) elements are stored in row-wise order:
    A(2,1) = rok_A[0], A(3,1)=rok_A[1], A(3,2)=rok_A[2], etc.
    The general mapping formula is:
        A_{i,j} = rok_A[ (i-1)*(i-2)/2 + j -1 ]   */
    rok_A[0] = (double)1.0;
    rok_A[1] = (double)0.1084530016931939175868117432550153454393009345950250308;
    rok_A[2] = (double)0.3915469983068060824131882567449846545606990654049749692;
    rok_A[3] = (double)0.4345304775600447762471370176270279643577768155816826992;
    rok_A[4] = (double)0.1448434925200149254157123392090093214525922718605608997;
    rok_A[5] = (double)(-0.0793739700800597016628493568360372858103690874422435989);
    
  /*~~~>     C_{i,j} = rok_C[ (i-1)*(i-2)/2 + j -1]  */
    rok_C[0] = (double)(-1.91153192976055097824558940682133229601582735199337138313);
    rok_C[1] = (double)0.3288182406115352215636148889409996289068084425348308862;
    rok_C[2] = (double)0.0;
    rok_C[3] = (double)0.0330364423979581129062589491015367687146658980867715225;
    rok_C[4] = (double)(-0.2437515237610823531197130316973934919183493340255052118);
    rok_C[5] = (double)(-0.1706260299199402983371506431639627141896309125577564011);
    
  /*~~~> does the stage i require a new function evaluation (rok_NewF(i)=TRUE)
    or does it re-use the function evaluation from stage i-1 (rok_NewF(i)=FALSE) */
    rok_NewF[0] = 1;
    rok_NewF[1] = 1;
    rok_NewF[2] = 1;
    rok_NewF[3] = 1;
    
  /*~~~> M_i = Coefficients for new step solution */
    rok_M[0] = (double)(1.0/6.0);
    rok_M[1] = (double)(1.0/6.0);
    rok_M[2] = (double)0.0;
    rok_M[3] = (double)(2.0/3.0);
    
  /*~~~> E_i = Coefficients for error estimator */    
    rok_E[0] = rok_M[0] - (double)0.5026932257368423534541250307675423348147218619695336514;
    rok_E[1] = rok_M[1] - (double)0.2786755196900585622624861213669585560493517317676223283;
    rok_E[2] = rok_M[2] - (double)0.2186312545730990842833888478654991091359264062628440203;
    rok_E[3] = rok_M[3] - (double)0.0;
    
  /*~~~> rok_ELO = estimator of local order - the minimum between the
!    main and the embedded scheme orders plus one */
    *rok_ELO = (double)4.0;   
     
  /*~~~> Y_stage_i ~ Y( T + H*Alpha_i ) */
    rok_Alpha[0] = (double)0.0;
    rok_Alpha[1] = (double)1.0; 
    rok_Alpha[2] = (double)0.5;
    rok_Alpha[3] = (double)0.5;
    
  /*~~~> Gamma_i = \sum_j  gamma_{i,j}  */     
    rok_Gamma[0] = (double)0.572816062482134855408001384976768340931514124329888624090;
    rok_Gamma[1] = (double)(-1.91153192976055097824558940682133229601582735199337138313);
    rok_Gamma[2] = (double)0.3288182406115352215636148889409996289068084425348308862;
    rok_Gamma[3] = (double)(-0.381341111);
    
}  /*  Rok4a */

void Rok4b ( int *rok_S, double rok_A[], double rok_C[],
           double rok_M[], double rok_E[],
     double rok_Alpha[], double rok_Gamma[],
     char rok_NewF[], double *rok_ELO, char* rok_Name )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *              6 stage, order 4, stiffly accurate Rosenbrock-Krylov method
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
  /*~~~> Name of the method */
    strcpy(rok_Name, "ROK-4b");

  /*~~~> Number of stages */
    *rok_S = 6;

  /*~~~> The coefficient matrices A and C are strictly lower triangular.
   The lower triangular (subdiagonal) elements are stored in row-wise order:
   A(2,1) = rok_A[0], A(3,1)=rok_A[1], A(3,2)=rok_A[2], etc.
   The general mapping formula is:
      A_{i,j} = rok_A[ (i-1)*(i-2)/2 + j -1 ]   */
    rok_A[0]  =  (double) 1.0;
    rok_A[1]  =  (double)0.530633333333333;
    rok_A[2]  =  (double)-0.030633333333333;
    rok_A[3]  =  (double)0.894444444444444;
    rok_A[4]  =  (double)0.055555555555556;
    rok_A[5]  =  (double)0.05;
    rok_A[6]  =  (double)0.738333333333333;
    rok_A[7]  =  (double)-0.121666666666667;
    rok_A[8]  =  (double)0.333333333333333;
    rok_A[9]  =  (double)0.05;
    rok_A[10] =  (double)-0.096929102825711;
    rok_A[11] =  (double)-0.121666666666667;
    rok_A[12] =  (double)1.045582889789120;
    rok_A[13] =  (double)0.173012879703258;
    rok_A[14] =  (double)0.0;


  /*~~~>     C_{i,j} = rok_C[ (i-1)*(i-2)/2 + j -1]  */
    rok_C[0] =  (double)-22.824608269858540;
    rok_C[1] =  (double)-69.343635255712726;
    rok_C[2] =  (double)-0.030633333333333;
    rok_C[3] =  (double)404.7106882480958;
    rok_C[4] =  (double)0.055555555555556;
    rok_C[5] =  (double)0.05;
    rok_C[6] =  (double)-0.571666666666667;
    rok_C[7] =  (double)-0.121666666666667;
    rok_C[8] =  (double)0.333333333333333;
    rok_C[9] =  (double)0.05;
    rok_C[10] =  (double)0.263595769492377;
    rok_C[11] =  (double)-0.121666666666667;
    rok_C[12] =  (double)-0.378916223122453;
    rok_C[13] =  (double)-0.073012879703258;
    rok_C[14] =  (double)0.0;


  /*~~~> does the stage i require a new function evaluation (rok_NewF(i)=TRUE)
   or does it re-use the function evaluation from stage i-1 (rok_NewF(i)=FALSE) */
    rok_NewF[0] = 1;
    rok_NewF[1] = 1;
    rok_NewF[2] = 1;
    rok_NewF[3] = 1;
    rok_NewF[4] = 1;
    rok_NewF[5] = 1;

  /*~~~> M_i = Coefficients for new step solution */
    rok_M[0] = (double) 0.166666666666667;
    rok_M[1] = (double)-0.243333333333333;
    rok_M[2] = (double) 0.666666666666667;
    rok_M[3] = (double) 0.100000000000000;
    rok_M[4] = (double) 0.0;
    rok_M[5] =  (double)0.31;


  /*~~~> E_i = Coefficients for error estimator */
    rok_E[0] =  (double)0.0;
    rok_E[1] =  (double)0.0;
    rok_E[2] =  (double)0.0;
    rok_E[3] =  (double)0.0;
    rok_E[4] =  (double)-0.31;
    rok_E[5] =  (double)0.31;
  /*~~~> rok_ELO = estimator of local order - the minimum between the
   main and the embedded scheme orders plus one */
    *rok_ELO = (double)4.0;

  /*~~~> Y_stage_i ~ Y( T + H*Alpha_i ) */
    rok_Alpha[0] = (double)0.0;
    rok_Alpha[1] = (double)1.0;
    rok_Alpha[2] = (double)0.5;
    rok_Alpha[3] = (double)1.0;
    rok_Alpha[4] = (double)1.0;
    rok_Alpha[5] = (double)1.0;

  /*~~~> Gamma_i = \sum_j  gamma_{i,j}  */
    rok_Gamma[0] = (double)0.31;
    rok_Gamma[1] = (double)(-1.91153192976055097824558940682133229601582735199337138313);
    rok_Gamma[2] = (double)0.3288182406115352215636148889409996289068084425348308862;
    rok_Gamma[3] = (double)(-0.381341111);
    rok_Gamma[4] = (double)0.0;
    rok_Gamma[5] = (double)0.0;
}  /*  Rok4b */

void Rok4p ( int *rok_S, double rok_A[], double rok_C[],
           double rok_M[], double rok_E[],
     double rok_Alpha[], double rok_Gamma[],
     char rok_NewF[], double *rok_ELO, char* rok_Name )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *              5 stage, order 4, parabolic Rosenbrock-Krylov method
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
{
  /*~~~> Name of the method */
    strcpy(rok_Name, "ROK-4p");

  /*~~~> Number of stages */
    *rok_S = 5;

  /*~~~> The coefficient matrices A and C are strictly lower triangular.
   The lower triangular (subdiagonal) elements are stored in row-wise order:
   A(2,1) = rok_A[0], A(3,1)=rok_A[1], A(3,2)=rok_A[2], etc.
   The general mapping formula is:
      A_{i,j} = rok_A[ (i-1)*(i-2)/2 + j -1 ]   */
    rok_A[0]  =  (double)0.757900000000000;
    rok_A[1]  =  (double)0.170400000000000;
    rok_A[2]  =  (double)0.821100000000000;
    rok_A[3]  =  (double)1.196218621274069;
    rok_A[4]  =  (double)0.297700000000000;
    rok_A[5]  =  (double)-1.433618621274069;
    rok_A[6]  =  (double)-0.010650410785863;
    rok_A[7]  =  (double)0.142100000000000;
    rok_A[8]  =  (double)-0.129349589214137;
    rok_A[9]  =  (double)0.392800000000000;


  /*~~~>     C_{i,j} = rok_C[ (i-1)*(i-2)/2 + j -1]  */
    rok_C[0] =  (double)-0.757900000000000;
    rok_C[1] =  (double)-0.295086678808293;
    rok_C[2] =  (double)0.178900000000000;
    rok_C[3] =  (double)-1.836333117783808;
    rok_C[4] =  (double)-0.247700000000000;
    rok_C[5] =  (double)1.681409044712106;
    rok_C[6] =  (double)-0.197089800872483;
    rok_C[7] =  (double)-0.684644029868020;
    rok_C[8] =  (double)0.166330242942910;
    rok_C[9] =  (double)0.00;


  /*~~~> does the stage i require a new function evaluation (rok_NewF(i)=TRUE)
   or does it re-use the function evaluation from stage i-1 (rok_NewF(i)=FALSE) */
    rok_NewF[0] = 1;
    rok_NewF[1] = 1;
    rok_NewF[2] = 1;
    rok_NewF[3] = 1;
    rok_NewF[4] = 1;
    rok_NewF[5] = 1;

  /*~~~> M_i = Coefficients for new step solution */
    rok_M[0] = (double)0.056000000000000;
    rok_M[1] = (double) 0.116601238130482;
    rok_M[2] = (double) 0.160300000000000;
    rok_M[3] = (double) -0.031109354304222;
    rok_M[4] = (double) 0.698208116173739;


  /*~~~> E_i = Coefficients for error estimator */
    rok_E[0] =  rok_M[0] + (double)0.186875355621256;
    rok_E[1] =  rok_M[0] + (double)0.250433793031115;
    rok_E[2] =  rok_M[0] - (double)0.326360736478684;
    rok_E[3] =  rok_M[0] - (double)0.110948412173687;
    rok_E[4] =  rok_M[0] - (double)1.000000000000000;
  /*~~~> rok_ELO = estimator of local order - the minimum between the
   main and the embedded scheme orders plus one */
    *rok_ELO = (double)4.0;

  /*~~~> Y_stage_i ~ Y( T + H*Alpha_i ) */
    rok_Alpha[0] = (double)0.0;
    rok_Alpha[1] = (double)0.757900000000000;
    rok_Alpha[2] = (double)0.991500000000000;
    rok_Alpha[3] = (double)0.060300000000000;
    rok_Alpha[4] = (double)0.394900000000000;

  /*~~~> Gamma_i = \sum_j  gamma_{i,j}  */
    rok_Gamma[0] = (double)0.572816062482134855408001384976768340931514124329888624090;
    rok_Gamma[1] = (double)(-1.91153192976055097824558940682133229601582735199337138313);
    rok_Gamma[2] = (double)0.3288182406115352215636148889409996289068084425348308862;
    rok_Gamma[3] = (double)(-0.381341111);
    rok_Gamma[4] = (double)0.0;

}  /*  Rok4p */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
void DecompTemplate( int N, double A[], int Pivot[], int* ising )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
        Template for the LU decomposition   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{   
/*  if (N == 4) {
     int i, j, k;
	 double r;
	 for (i = 0; i < N-1; i++) {
	    for (j = i+1; j < N; j++) {
	       r = A[(j) * N + (i)] / A[(i) * N + (i)];
		   for (k = j; k < N; k++) {
		      A[(j) * N + (k)] = A[(j) * N + (k)] - r * A[(i) * N + (k)];
		   }
		   A[(j) * N + (i)] = r;
		}
	 }
    *ising = 0;
  } else { */
     // Must use lapack
	 *ising = rok_dgetrf(N, N, A, N, Pivot);
//  }
   Ndec++;
}  /*  DecompTemplate */
 
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
 void SolveTemplate( int N, double A[], int Pivot[], double b[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
     Template for the forward/backward substitution (using pre-computed LU decomposition)   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{   
/*   if (N == 4) {
      int i, j;
	  for (i = N-1; i >= 0; i--) {
	     for (j = i+1; j < N; j++) {
		    b[i] = b[i] - A[(i) * N + (j)] * b[j];
		 }
	     b[i] = b[i] / A[(i) * N + (i)];
	  }
	  
	  for (i = 1; i < N; i++) {
	     for (j = 0; j < i; j++) {
		    b[i] = b[i] - A[(i) * N + (j)] * b[j];
		 }
	  }
   } else { */
      // Must use lapack
      rok_dgetrs('n', N, 1, A, N, Pivot, b, N);
//   }
     
   Nsol++;

}  /*  SolveTemplate */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
void FunTemplate( double T, double Y[], double Ydot[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    Template for the ODE function call.
    Updates the rate coefficients (and possibly the fixed species) at each call    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{
   swe_fun(T, Y, Ydot);
     
   Nfun++;
   
}  /*  FunTemplate */

 
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
void JacTemplate( double T, double Y[], double Jcb[] )
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    Template for the ODE Jacobian call.
    Updates the rate coefficients (and possibly the fixed species) at each call    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/   
{
   swe_jac(T, Y, Jcb);
   
   Njac++;
   
} /* JacTemplate   */                                    

/* End of INTEGRATE function                                        */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
