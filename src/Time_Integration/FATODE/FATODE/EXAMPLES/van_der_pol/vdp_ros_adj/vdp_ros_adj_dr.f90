!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE FUN(NVAR, T, Y, P)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER :: NVAR
      DOUBLE PRECISION :: T, MU
      DOUBLE PRECISION :: Y(NVAR), P(NVAR)
      MU = 10.0d0
      P(1) = Y(2)
      P(2) = MU*(1.0d0-Y(1)*Y(1))*Y(2)-Y(1)

      END SUBROUTINE FUN

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE JAC(neq, t, y, pd)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      integer neq
      double precision pd,t,y,mu
      dimension y(2), pd(2,2)

      MU = 10.0d0
      pd(1,1) = 0.0d0
      pd(1,2) = 1.0d0
      pd(2,1) = (-2.0d0*MU*y(1)*y(2)-1.0d0)
      pd(2,2) = MU*(1.0d0-y(1)**2)

      END SUBROUTINE JAC

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE HESS( NVAR, T, Y, H )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
	FPJAC(2,1) = (1.0d0-Y(1)**2.0d0)*Y(2)

	END SUBROUTINE JACP

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE HESSTR_VEC_F_PY( NY, NP, T, Y, U, K, TMP )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!		tmp = (f_py x k)^T * u = (d(f_p^T * u)/dy) * k
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NY, NP
	DOUBLE PRECISION :: T, Y(NY), U(NY), K(NY), TMP(NP)

	TMP = -2*U(2)*Y(1)*Y(2)*K(1) + U(2)*K(2)*(1-Y(1)**2)

	END SUBROUTINE HESSTR_VEC_F_PY

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE HESSTR_VEC_R_PY( IADJ, NY, NP, T, Y, U, K, TMP )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!       tmp = (r_py x k )^T * u = (d(r_p^T *u)/dy) *k
!       u is scalar
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NY, NP, IADJ
	DOUBLE PRECISION :: T, Y(NY), U, K(NY), TMP(NP)

	TMP(:) = 0.0d0

	END SUBROUTINE HESSTR_VEC_R_PY

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE HESSTR_VEC_R( IADJ, NVAR, T, Y, U, K, TMP )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!       tmp =(r_yy x k )^T * u = (d(r_y^T * u)/dy) * k
!       u is scalar
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: IADJ, NVAR
	DOUBLE PRECISION :: T, Y(NVAR), U, K(NVAR), TMP(NVAR)

	TMP(:) = 0.0d0

	END SUBROUTINE HESSTR_VEC_R

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE HESSTR_VEC( NVAR, T, Y, U, K, TMP )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!     tmp = (hess x k)^T * u = (d(J^T * u)/dy) * k
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NVAR
	DOUBLE PRECISION :: MU
	DOUBLE PRECISION :: T, Y(NVAR), U(NVAR), K(NVAR), TMP(NVAR)
	
	MU = 10.0d0
	
	TMP(1) = -2.0d0*MU*U(2)*(Y(2)*K(1)+Y(1)*K(2))
	TMP(2) = -2.0d0*MU*U(2)*Y(1)*K(1)

	END SUBROUTINE HESSTR_VEC

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

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE ADJINIT( NVAR, NP, NADJ, T, Y, Lambda, Mu)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      INTEGER :: NVAR, NP, NADJ, K
      DOUBLE PRECISION :: T, Y(NVAR), Lambda(NVAR,NADJ)
      DOUBLE PRECISION, INTENT(IN), OPTIONAL :: Mu(NP,NADJ)
      
      Lambda(1:NVAR, 1:NADJ) = 0.0d0
      DO K=1,NADJ
            Lambda(K,K) = 1.0d0
      END DO

      END SUBROUTINE ADJINIT

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!	DRIVER FILE: Van Der Pol ROS ADJ
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	PROGRAM vdp_ros_adj_dr
	USE ROS_ADJ_f90_Integrator

	IMPLICIT NONE
	INTEGER, PARAMETER :: NVAR = 2, NNZ = 0, NADJ = 2, NP = 1

	DOUBLE PRECISION, DIMENSION(20) :: RCNTRL, RSTATUS
	INTEGER, DIMENSION(20) :: ICNTRL, ISTATUS

	DOUBLE PRECISION :: ATOL(NVAR), RTOL(NVAR), VAR(NVAR), Q(NADJ)
	DOUBLE PRECISION :: ATOL_ADJ(NVAR), RTOL_ADJ(NVAR)
	DOUBLE PRECISION :: Y_ADJ(NVAR,NADJ), YP_ADJ(NP,NADJ)
	DOUBLE PRECISION :: TSTART, TEND, T

	INTEGER :: i, j, mode

	EXTERNAL FUN, JAC, ADJINIT, HESSTR_VEC, DRDP, DRDY, JACP, &
		   HESSTR_VEC_F_PY, QFUN, HESSTR_VEC_R_PY, HESSTR_VEC_R

	DO i=1,NVAR
		RTOL(i) = 1.0d-5
		ATOL(i) = 1.0d-5
	END DO

	DO i=1,NVAR
		ATOL_ADJ(i) = 10.0d0*ATOL(i)
		RTOL_ADJ(i) = 10.0d0*RTOL(i)
	END DO

	DO i=1,NP
		DO j=1,NADJ
			YP_ADJ(i,j) = 0.0d0
		END DO
	END DO

	VAR(1) = 2.0d0
	VAR(2) = -0.66d0
	Q(:) = 0.0d0

	TSTART = 0.0d0	
	TEND = TSTART + 20.0d0

	ICNTRL(1:20) = 0
	RCNTRL(1:20) = 0.0d0

	ICNTRL(3) = 1
	ICNTRL(4) = 0

	T = TSTART
	mode = 4
	SELECT CASE ( mode )
		CASE ( 1 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
      		     NNZERO=NNZ, LAMBDA=Y_ADJ, TIN=T, TOUT=TEND, ATOL=ATOL, &
      		     RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
      		     ISTATUS_U=ISTATUS, RSTATUS_U=RSTATUS, ADJINIT=ADJINIT, &
      		     ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, MU=YP_ADJ, &
			     HESSTR_VEC=HESSTR_VEC, DRDP=DRDP, DRDY=DRDY, JACP=JACP, &
			     HESSTR_VEC_F_PY=HESSTR_VEC_F_PY, QFUN=QFUN, Q=Q, &
			     HESSTR_VEC_R_PY=HESSTR_VEC_R_PY, HESSTR_VEC_R=HESSTR_VEC_R )
		CASE ( 2 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
      		     NNZERO=NNZ, LAMBDA=Y_ADJ, TIN=T, TOUT=TEND, ATOL=ATOL, &
      		     RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
      		     ISTATUS_U=ISTATUS, RSTATUS_U=RSTATUS, ADJINIT=ADJINIT, &
      		     ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, MU=YP_ADJ, &
			     HESSTR_VEC=HESSTR_VEC, JACP=JACP, &
			     HESSTR_VEC_F_PY=HESSTR_VEC_F_PY )
		CASE ( 3 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
      		     NNZERO=NNZ, LAMBDA=Y_ADJ, TIN=T, TOUT=TEND, ATOL=ATOL, &
      		     RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
      		     ISTATUS_U=ISTATUS, RSTATUS_U=RSTATUS, ADJINIT=ADJINIT, &
      		     ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, &
			     HESSTR_VEC=HESSTR_VEC, DRDP=DRDP, DRDY=DRDY, &
			     QFUN=QFUN, Q=Q, &
			     HESSTR_VEC_R_PY=HESSTR_VEC_R_PY, HESSTR_VEC_R=HESSTR_VEC_R )
		CASE ( 4 )
			CALL INTEGRATE_ADJ( NVAR=NVAR, NP=NP, NADJ=NADJ, Y=VAR, &
      		     NNZERO=NNZ, LAMBDA=Y_ADJ, TIN=T, TOUT=TEND, ATOL=ATOL, &
      		     RTOL=RTOL, FUN=FUN, JAC=JAC, ICNTRL_U=ICNTRL, &
      		     ISTATUS_U=ISTATUS, RSTATUS_U=RSTATUS, ADJINIT=ADJINIT, &
      		     ATOL_ADJ=ATOL_ADJ, RTOL_ADJ=RTOL_ADJ, &
			     HESSTR_VEC=HESSTR_VEC, DRDP=DRDP, &
			     HESSTR_VEC_R_PY=HESSTR_VEC_R_PY )	
	END SELECT

print *, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

	SELECT CASE ( mode )
		CASE ( 1 )
			WRITE(6,250) TEND, VAR(1:NVAR), Y_ADJ(1:NVAR,1:NADJ), YP_ADJ(1:NP,1:NADJ), &
	      	     Q(1:NADJ), ISTATUS(1:8), RSTATUS(1:4)
250			FORMAT(/, &
                       'Van Der Pol: ROS ADJ'/, &
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
			WRITE(6,251) TEND, VAR(1:NVAR), Y_ADJ(1:NVAR,1:NADJ), YP_ADJ(1:NP,1:NADJ), &
	      	     ISTATUS(1:8), RSTATUS(1:4)
251			FORMAT(/, &
                       'Van Der Pol: ROS ADJ'/, &
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
			WRITE(6,252) TEND, VAR(1:NVAR), Y_ADJ(1:NVAR,1:NADJ), &
	      	     Q(1:NADJ), ISTATUS(1:8), RSTATUS(1:4)
252			FORMAT(/, &
                       'Van Der Pol: ROS ADJ'/, &
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
			WRITE(6,253) TEND, VAR(1:NVAR), Y_ADJ(1:NVAR,1:NADJ), &
	      	     ISTATUS(1:8), RSTATUS(1:4)
253			FORMAT(/, &
                       'Van Der Pol: ROS ADJ'/, &
                       ' Time=', f5.2,' Value=',2E12.5,/, &
                       ' Lambda=',4E12.5,/,/, &
                       'ISTATUS:',/, &
                       ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
                       ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
                       'RSTATUS:',/, &
                       ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
                       ' Nhnew =',E12.5,' Nhexit=',E12.5,/)
	END SELECT

	END PROGRAM vdp_ros_adj_dr
	
