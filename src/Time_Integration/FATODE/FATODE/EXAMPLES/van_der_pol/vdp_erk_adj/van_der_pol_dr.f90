      SUBROUTINE FUN(NVAR, T, Y, P )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER ::NVAR
      DOUBLE PRECISION :: T, EPS
      DOUBLE PRECISION :: Y(NVAR), P(NVAR)
      DOUBLE PRECISION :: PROD
      EPS = 1.d0
      P(1)=Y(2)
      PROD=1.D0-Y(1)*Y(1)
      P(2)=(PROD)*Y(2)-Y(1)/EPS
      END SUBROUTINE FUN
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE JAC(NVAR, T, Y, JV )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER ::NVAR
      DOUBLE PRECISION::T,EPS
      DOUBLE PRECISION::Y(NVAR)
      DOUBLE PRECISION::JV(NVAR,NVAR)
      EPS=1.0D0
      JV(1,1)=0.0D0
      JV(1,2)=1.0D0
      JV(2,1)=(-2.0D0*Y(1)*Y(2)-1.0D0)/EPS
      JV(2,2)=(1.0D0-Y(1)**2)/EPS
      END SUBROUTINE JAC
      

!~~~> initialize adjoint variables
      subroutine adjinit(n,np,nadj,t,y,lambda,mu)
      integer :: n,np,nadj,k
      double precision :: t,y(n),lambda(n,nadj)
      double precision,optional :: mu(np,nadj)
!~~~> if number of parameters is not zero, extra adjoint varaible mu should be
!defined
      if(NP>0 .and. .not. present(mu)) stop 'undefined argument mu'
!~~~>  the adjoint values at the final time
      lambda(:,:) = 0.0d0
      lambda(1,1) = 1.0d0
      lambda(2,2) = 1.0d0
      end subroutine adjinit

   PROGRAM van_der_pol_dr
    USE ERK_adj_f90_Integrator
    IMPLICIT NONE
    INTEGER, PARAMETER :: NVAR = 2, NNZ = 0, NADJ = 2, NP = 0

!~~>  Control (in) and status (out) arguments for the integration
    DOUBLE PRECISION, DIMENSION(20) :: RCNTRL, RSTATUS
    INTEGER,       DIMENSION(20) :: ICNTRL, ISTATUS

    DOUBLE PRECISION :: ATOL(NVAR), RTOL(NVAR), ATOL_ADJ(NVAR),RTOL_ADJ(NVAR),VAR(NVAR),Lambda(NVAR,NADJ)
    DOUBLE PRECISION :: TSTART, TEND, T
    DOUBLE PRECISION :: time
    INTEGER :: i, iter, hz, clock0,clock1,cpu
    EXTERNAL FUN,JAC,adjinit
!~~~> Tolerances for calculating concentrations       
    DO i=1,NVAR
      RTOL(i) = 1.0d-1
      ATOL(i) = 1.0d-1
    END DO
    ATOL_ADJ(:) = .1d0
    RTOL_ADJ(:) = .1d0

    DO iter = 1, 5
!~~~> Tolerances for calculating concentrations       
    DO i=1,NVAR
      RTOL(i) = 0.1*RTOL(i)
      ATOL(i) = 0.1*ATOL(i)
    END DO

!      Initialize
     VAR(1) = 2.0d0
     VAR(2) = -0.66d0
 
     TSTART = 0.0d0
     TEND = TSTART + 0.2d0

!~~~> Default control options
    ICNTRL(1:20) = 0
    RCNTRL(1:20) = 0.0d0

!~~~> Begin time loop

    T = TSTART

    call system_clock(count_rate=hz)
    call system_clock(count=clock0)
    CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, NNZ=NNZ, Y=VAR,          &
           Lambda=Lambda, TIN=T, TOUT=TEND, ATOL=ATOL, RTOL=RTOL, FUN=FUN,    &
           JAC=JAC, ICNTRL_U=ICNTRL, ISTATUS_U=ISTATUS, RSTATUS_U=RSTATUS,    &
           ADJINIT=adjinit)
    call system_clock(count=clock1)
    cpu = clock1-clock0
    time = cpu/(real(hz))
!~~~> End time loop ~~~~~~~~~~
!    print *,VAR
    write(6,919) time
919 FORMAT(2x,'     time elapsed:',F12.9)
 
    WRITE(6,250) TEND, VAR(1:2)
250 FORMAT(' Time=',f5.2,' Value=',2E12.5,/)
    end do
    
    print *,Lambda
    END PROGRAM van_der_pol_dr


