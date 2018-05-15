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

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE DRDP( NADJ, NVAR, NRP, T, Y, RP )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NADJ, N, NRP
	DOUBLE PRECISION :: T, Y(NVAR), RP(NRP,NADJ)

	RP(:,:) = 0.0d0

	END SUBROUTINE DRDP

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE DRDY( NADJ, NVAR, NRY, T, Y, RY )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NADJ, NVAR, NRY
	DOUBLE PRECISION :: T, Y(NVAR), RY(NRY, NADJ)

	RY(:,:) = 0.0d0
	RY(1,1) = 1.0d0
	RY(2,2) = 1.0d0

	END SUBROUTINE DRDY

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE JACP( NVAR, NP, T, Y, FPJAC )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!       fpjac = df/dp
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NVAR, NP
	DOUBLE PRECISION :: T, Y(NVAR), FPJAC(NVAR,NP)

	FPJAC(1,1) = 0.0d0
	FPJAC(2,1) = Y(2)*Y(1)**2.0d0 + Y(2)

	END SUBROUTINE JACP

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE QFUN( NVAR, NR, T, Y, R )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IMPLICIT NONE
	INTEGER :: NVAR, NR
	DOUBLE PRECISION :: T
	DOUBLE PRECISION :: Y(NVAR), R(NR)

	R(1) = Y(1)
	R(2) = Y(2)

	END SUBROUTINE QFUN

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE ADJINIT( N, NP, NADJ, T, Y, LAMBDA, MU )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: N, NP, NADJ, K
	DOUBLE PRECISION :: T, Y(N), LAMBDA(N,NADJ)
	DOUBLE PRECISION, OPTIONAL :: MU(NP,NADJ)

	INTEGER :: i,j

      Lambda(1:NVAR, 1:NADJ) = 0.0d0
      DO K=1,NADJ
            Lambda(K,K) = 1.0d0
      END DO
	END SUBROUTINE ADJINIT

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!	DRIVER FILE: Van Der Pol RK ADJ
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	PROGRAM vdp_rk_adj_dr
	USE RK_ADJ_f90_Integrator

	IMPLICIT NONE
	INTEGER, PARAMETER :: NVAR = 2, NNZ = 0, NADJ = 2, NP = 1

	DOUBLE PRECISION, DIMENSION(20) :: RCNTRL, RSTATE
	INTEGER, DIMENSION(20) :: ICNTRL, ISTATE

	DOUBLE PRECISION :: ATOL(NVAR), RTOL(NVAR), VAR(NVAR), Q(NADJ)
	DOUBLE PRECISION :: ATOL_ADJ(NVAR), RTOL_ADJ(NVAR)
	DOUBLE PRECISION :: LAMBDA(NVAR,NADJ), YP_ADJ(NP,NADJ)
	DOUBLE PRECISION :: TSTART, TEND, T

	INTEGER :: i, j, mode

	EXTERNAL FUN, JAC, ADJINIT, DRDP, DRDY, JACP, QFUN

	DO i=1,NVAR
		RTOL(i) = 1.0d-5
		ATOL(i) = 1.0d-5
	END DO

	DO i=1,NVAR
		ATOL_ADJ(i) = 1.0d-4
		RTOL_ADJ(i) = 1.0d-4
	END DO

	DO i=1,NP
		DO j=1,NADJ
			YP_ADJ(i,j) = 0.0d0
		END DO
	END DO

	VAR(1) = 2.0d0
	VAR(2) = -0.66d0

	Q(:)   = 0.0d0

	TSTART = 0.0d0	
	TEND = TSTART + 20.0d0

	ICNTRL(1:20) = 0
	RCNTRL(1:20) = 0.0d0

        ICNTRL(3) = 0
        ICNTRL(4) = 0

	T = TSTART
	mode = 4
	SELECT CASE ( mode )
		CASE ( 1 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
				NNZERO=NNZ, LAMBDA=LAMBDA, TIN=T, TOUT=TEND, ATOL=ATOL, &
				RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
				ISTATUS_U=ISTATE, RSTATUS_U=RSTATE, ADJINIT=ADJINIT, &
				ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, &
				JACP=JACP, MU=YP_ADJ, Q=Q, QFUN=QFUN, DRDY=DRDY, &
				DRDP=DRDP )
		CASE ( 2 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
				NNZERO=NNZ, LAMBDA=LAMBDA, TIN=T, TOUT=TEND, ATOL=ATOL, &
				RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
				ISTATUS_U=ISTATE, RSTATUS_U=RSTATE, ADJINIT=ADJINIT, &
				ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, &
				JACP=JACP, MU=YP_ADJ )
		CASE ( 3 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
				NNZERO=NNZ, LAMBDA=LAMBDA, TIN=T, TOUT=TEND, ATOL=ATOL, &
				RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
				ISTATUS_U=ISTATE, RSTATUS_U=RSTATE, ADJINIT=ADJINIT, &
				ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, &
				Q=Q, QFUN=QFUN, DRDY=DRDY, DRDP=DRDP )
		CASE ( 4 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
				NNZERO=NNZ, LAMBDA=LAMBDA, TIN=T, TOUT=TEND, ATOL=ATOL, &
				RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
				ISTATUS_U=ISTATE, RSTATUS_U=RSTATE, ADJINIT=ADJINIT, &
				ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ )
	END SELECT


print *, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

	SELECT CASE ( mode )
		CASE ( 1 )
			WRITE(6,250) TEND, VAR(1:NVAR), LAMBDA(1:NVAR,1:NADJ), YP_ADJ(1:NP,1:NADJ), &
	      	     Q(1:NADJ), ISTATE(1:8), RSTATE(1:4)
250			FORMAT(/, &
                       'Van Der Pol: ERK ADJ'/, &
                       ' Time=', f5.2,' Value=',2E12.5,/, &
                       ' Lambda=',4E12.5,/, &
                       ' Mu= ', 2E12.5,/, &
                       ' Q=  ', 2E12.5,/,/, &
                       'ISTATUS:',/, &
                       ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
                       ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
                       'RSTATUS:',/, &
                       ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
                       ' Nhnew =',E12.5,' Nhexit=',E12.5,/)
		CASE ( 2 )
			WRITE(6,251) TEND, VAR(1:NVAR), LAMBDA(1:NVAR,1:NADJ), YP_ADJ(1:NP,1:NADJ), &
	      	     ISTATE(1:8), RSTATE(1:4)
251			FORMAT(/, &
                       'Van Der Pol: ERK ADJ'/, &
                       ' Time=', f5.2,' Value=',2E12.5,/, &
                       ' Lambda=',4E12.5,/, &
                       ' Mu= ', 2E12.5,/,/, &
                       'ISTATUS:',/, &
                       ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
                       ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
                       'RSTATUS:',/, &
                       ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
                       ' Nhnew =',E12.5,' Nhexit=',E12.5,/)
		CASE ( 3 )
			WRITE(6,252) TEND, VAR(1:NVAR), LAMBDA(1:NVAR,1:NADJ), &
	      	     Q(1:NADJ), ISTATE(1:8), RSTATE(1:4)
252			FORMAT(/, &
                       'Van Der Pol: ERK ADJ'/, &
                       ' Time=', f5.2,' Value=',2E12.5,/, &
                       ' Lambda=',4E12.5,/, &
                       ' Q=  ', 2E12.5,/,/, &
                       'ISTATUS:',/, &
                       ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
                       ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
                       'RSTATUS:',/, &
                       ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
                       ' Nhnew =',E12.5,' Nhexit=',E12.5,/)
		CASE ( 4 )
			WRITE(6,253) TEND, VAR(1:NVAR), LAMBDA(1:NVAR,1:NADJ), &
	      	     ISTATE(1:8), RSTATE(1:4)
253			FORMAT(/, &
                       'Van Der Pol: ERK ADJ'/, &
                       ' Time=', f5.2,' Value=',2E12.5,/, &
                       ' Lambda=',4E12.5,/,/, &
                       'ISTATUS:',/, &
                       ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
                       ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
                       'RSTATUS:',/, &
                       ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
                       ' Nhnew =',E12.5,' Nhexit=',E12.5,/)
	END SELECT

	END PROGRAM vdp_rk_adj_dr

