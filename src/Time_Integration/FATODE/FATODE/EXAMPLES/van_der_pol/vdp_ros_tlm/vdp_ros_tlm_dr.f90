!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE FUN(NVAR, T, Y, P)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER ::NVAR
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
      SUBROUTINE HESS( NVAR, T, Y )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	IMPLICIT NONE
	INTEGER :: NVAR
	DOUBLE PRECISION :: T, MU
	DOUBLE PRECISION :: Y(NVAR), H(NVAR,NVAR,NVAR)

	MU = 10.0d0

	H(1,1,1) = 0.0d0
	H(1,2,1) = 0.0d0
	H(2,1,1) = 0.0d0
	H(2,2,1) = 0.0d0

	H(1,1,2) = -2.0d0*MU*Y(2)
	H(1,2,2) = -2.0d0*MU*Y(1)
	H(2,1,2) = -2.0d0*MU*Y(1)
	H(2,2,2) = 0.0d0

      END SUBROUTINE HESS

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SUBROUTINE HESS_VEC( NVAR, T, Y, U, V, HV )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	INTEGER :: NVAR
	DOUBLE PRECISION :: T, Y(NVAR), U(NVAR), V(NVAR), HV(NVAR)
	DOUBLE PRECISION :: MU

	MU = 10.0d0

	HV(1) = 0
	HV(2) = -2.0d0*MU*(U(1)*(Y(2)*V(1)+Y(1)*V(2))+Y(1)*V(1)*U(2))

	END SUBROUTINE HESS_VEC

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      PROGRAM vdp_ros_tlm_dr
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      USE ros_tlm_f90_integrator
      IMPLICIT NONE
      EXTERNAL FUN, JAC, HESS_VEC

      INTEGER, PARAMETER :: NTLM = 2, NVAR = 2, NNZ = 0

      DOUBLE PRECISION :: ATOL(NVAR), RTOL(NVAR), VAR(NVAR)
      DOUBLE PRECISION :: ATOL_TLM(NVAR,NTLM), RTOL_TLM(NVAR,NTLM)
      DOUBLE PRECISION :: Y_TLM(NVAR,NTLM)

      DOUBLE PRECISION :: TSTART, TEND
      
      DOUBLE PRECISION :: RSTATUS(20), RCNTRL(20)
      INTEGER :: ISTATUS(20), ICNTRL(20)

      INTEGER :: i, k

      DO i=1,NVAR
            RTOL(i) = 1.0d-5
            ATOL(i) = 1.0d-5
      END DO

      DO k=1,NTLM
            DO i=1,NVAR
                  RTOL_TLM(i,k) = 10.0d0*RTOL(i)
                  ATOL_TLM(i,k) = 10.0d0*ATOL(i)
            END DO
      END DO

      Y_TLM(1:NVAR,1:NTLM) = 0.0d0

      DO k=1,NTLM
            Y_TLM(k,k) = 1.0d0
      END DO

      VAR(1) = 2.0d0
      VAR(2) = -0.66d0

      TSTART = 0.0d0
      TEND = 20.0d0

      ICNTRL(1:20) = 0
      RCNTRL(1:20) = 0.0d0

	ICNTRL(2) = 0
	ICNTRL(3) = 2
	ICNTRL(4) = 0
	ICNTRL(12) = 0

      CALL INTEGRATE_TLM( TIN=TSTART, TOUT=TEND, N=NVAR, &
            NNZERO=NNZ, Y_TLM=Y_TLM, Y=VAR, RTOL=RTOL, ATOL=ATOL, &
            NTLM=NTLM, FUN=FUN, JAC=JAC, RSTATUS_U=RSTATUS, &
            RCNTRL_U=RCNTRL, ISTATUS_U=ISTATUS, ICNTRL_U=ICNTRL, &
            ATOL_TLM=ATOL_TLM, RTOL_TLM=RTOL_TLM, HESS_VEC=HESS_VEC )
      
	WRITE(6,250) TEND, VAR(1:NVAR), Y_TLM(1:NVAR,1:NTLM), ISTATUS(1:8), RSTATUS(1:4)
250		FORMAT(/, &
                  'Van Der Pol: ROS TLM'/, &
                  ' Time=', f5.2,' Value=',2E12.5,/, &
                  ' Lambda=',4E12.5,/, &
                  'ISTATUS:',/, &
                  ' Nfun=',I6,' Njac=',I6,' Nstp=',I6,' Nacc=',I6,/, &
                  ' Nrej=',I6,' Ndec=',I6,' Nsol=',I6,' Nsng=',I6,/,/, &
                  'RSTATUS:',/, &
                  ' Ntexit=',E12.5,' Nhacc =',E12.5,/, &
                  ' Nhnew =',E12.5,' Nhexit=',E12.5,/)
      
      END PROGRAM vdp_ros_tlm_dr
