!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE FUN( NVAR, T, Y, P )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IMPLICIT NONE
	INTEGER :: NVAR
	DOUBLE PRECISION :: T
	DOUBLE PRECISION :: Y(NVAR), P(NVAR)

	DOUBLE PRECISION :: mu

	mu = 10.0d0

	P(1) = Y(2)
	P(2) = mu*(1.0d0-Y(1)*Y(1))*Y(2)-Y(1)

	END SUBROUTINE FUN

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE JAC( NVAR, T, Y, J )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IMPLICIT NONE
	INTEGER :: NVAR
	DOUBLE PRECISION :: T
	DOUBLE PRECISION :: Y(NVAR), J(NVAR,NVAR)

	DOUBLE PRECISION :: mu

	mu = 10.0d0

	J(1,1) = 0.0d0
	J(1,2) = 1.0d0

	J(2,1) = -2.0d0*mu*Y(1)*Y(2)-1.0d0
	J(2,2) = mu*(1.0d0-Y(1)**2.0d0)

	END SUBROUTINE JAC

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE HESS( NVAR, T, Y, H )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	END SUBROUTINE HESS

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!	DRIVER FILE: Van Der Pol RK
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	PROGRAM vdp_rk_dr
	USE RK_f90_Integrator

	IMPLICIT NONE
	INTEGER, PARAMETER :: NVAR = 2, NNZ = 0

	DOUBLE PRECISION, DIMENSION(20) :: RCNTRL, RSTATUS
	INTEGER, DIMENSION(20) :: ICNTRL, ISTATUS

	DOUBLE PRECISION :: ATOL(NVAR), RTOL(NVAR), VAR(NVAR)
	DOUBLE PRECISION :: TSTART, TEND, T

	INTEGER :: i

	EXTERNAL FUN, JAC

	DO i=1,NVAR
		RTOL(i) = 1.0d-5
		ATOL(i) = 1.0d-5
	END DO

	VAR(1) = 2.0d0
	VAR(2) = -0.66d0

	TSTART = 0.0d0
	TEND = TSTART + 20.0d0

	ICNTRL(1:20) = 0
        ICNTRL(11) = 1
	RCNTRL(1:20) = 0.0d0

        ICNTRL(3) = 0
        ICNTRL(4) = 0

	T = TSTART
	CALL INTEGRATE( TIN=T, TOUT=TEND, N=NVAR, NNZERO=NNZ, &
           VAR=VAR, RTOL=RTOL, ATOL=ATOL, FUN=FUN, JAC=JAC, &
           RSTATUS_U=RSTATUS, ISTATUS_U=ISTATUS, ICNTRL_U=ICNTRL )

	WRITE(6,250) TEND, VAR(1:2), ISTATUS(1:8), RSTATUS(1:4)
250	FORMAT(/, &
           'Van Der Pol: RK'/,&
           ' Time=',f5.2,' Value=',2E12.5,/,/, &
           'ISTATUS:',/, &
           ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
           ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
           'RSTATUS:',/, &
           ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
           ' Nhnew =',E12.5,' Nhexit=',E12.5,/)

	END PROGRAM vdp_rk_dr
