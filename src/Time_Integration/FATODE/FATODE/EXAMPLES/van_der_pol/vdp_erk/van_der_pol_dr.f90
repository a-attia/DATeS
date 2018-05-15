      SUBROUTINE FUN(NVAR, T, Y, P )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER ::NVAR
      DOUBLE PRECISION :: T, EPS
      DOUBLE PRECISION :: Y(NVAR), P(NVAR)
      DOUBLE PRECISION :: PROD
!      EPS=1.0D-6
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
      EPS=1.0D-6
      JV(1,1)=0.0D0
      JV(1,2)=1.0D0
      JV(2,1)=(-2.0D0*Y(1)*Y(2)-1.0D0)/EPS
      JV(2,2)=(1.0D0-Y(1)**2)/EPS
      END SUBROUTINE JAC
 
   PROGRAM van_der_pol_dr
    USE ERK_f90_Integrator
    IMPLICIT NONE
    INTEGER, PARAMETER :: NVAR = 2, NNZ = 0

!~~>  Control (in) and status (out) arguments for the integration
    DOUBLE PRECISION, DIMENSION(20) :: RCNTRL, RSTATUS
    INTEGER,       DIMENSION(20) :: ICNTRL, ISTATUS

    DOUBLE PRECISION :: ATOL(NVAR), RTOL(NVAR), VAR(NVAR)
    DOUBLE PRECISION :: TSTART, TEND, T
    DOUBLE PRECISION :: time
    INTEGER :: i, iter, hz, clock0,clock1,cpu
    EXTERNAL FUN,JAC
!~~~> Tolerances for calculating concentrations       
    DO i=1,NVAR
      RTOL(i) = 1.0d-1
      ATOL(i) = 1.0d-1
    END DO

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
    CALL INTEGRATE(TIN=T,TOUT=TEND,NVAR=NVAR,VAR=VAR,RTOL=RTOL,&
          ATOL=ATOL,FUN=FUN,RSTATUS_U=RSTATUS,ISTATUS_U=ISTATUS,&
               ICNTRL_U=ICNTRL)
    call system_clock(count=clock1)
    cpu = clock1-clock0
    time = real(cpu)/(real(hz))
!~~~> End time loop ~~~~~~~~~~
!    print *,VAR
    write(6,919) time
919 FORMAT(2x,'     time elapsed:',F12.9)
 
    WRITE(6,250) TEND, VAR(1:2)
250 FORMAT(' Time=',f5.2,' Value=',2E12.5,/)
    end do
    END PROGRAM van_der_pol_dr


