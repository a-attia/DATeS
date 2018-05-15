!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE FUN(NDIM, T, Y, P)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER :: NDIM
      DOUBLE PRECISION :: T, MU
      DOUBLE PRECISION :: Y(NDIM), P(NDIM)
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

      mu = 10.0d0
      pd(1,1) = 0.0d0
      pd(1,2) = 1.0d0
      pd(2,1) = (-2.0d0*mu*y(1)*y(2)-1.0d0)
      pd(2,2) = mu*(1.0d0-y(1)**2)

      END SUBROUTINE JAC

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE HESS( NDIM, T, Y, H )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      END SUBROUTINE HESS

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      PROGRAM vdp_sdirk_tlm_dr
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      USE sdirk_tlm_f90_integrator
      IMPLICIT NONE
      EXTERNAL FUN, JAC

      INTEGER, PARAMETER :: NTLM = 2, NDIM = 2, NNZ = 0

      DOUBLE PRECISION :: ATOL(NDIM), RTOL(NDIM), VAR(NDIM)
      DOUBLE PRECISION :: ATOL_TLM(NDIM,NTLM), RTOL_TLM(NDIM,NTLM), Y_TLM(NDIM,NTLM)

      DOUBLE PRECISION :: TSTART, TEND
      
      DOUBLE PRECISION :: RSTATUS(20), RCNTRL(20)
      INTEGER :: ISTATUS(20), ICNTRL(20)

      INTEGER :: i, k

      DO i=1,NDIM
            RTOL(i) = 1.0d-5
            ATOL(i) = 1.0d-5
      END DO

      DO k=1,NTLM
            DO i=1,NDIM
                  RTOL_TLM(i,k) = 10.0d0*RTOL(i)
                  ATOL_TLM(i,k) = 10.0d0*ATOL(i)
            END DO
      END DO

      Y_TLM(1:NDIM,1:NTLM) = 0.0d0
      DO k=1,NTLM
            Y_TLM(k,k) = 1.0d0
      END DO

      VAR(1) = 2.0d0
      VAR(2) = -0.66d0

      TSTART = 0.0d0
      TEND = 20.0d0

      ICNTRL(1:20) = 0
      RCNTRL(1:20) = 0.0d0

      ICNTRL(3) = 0

      CALL INTEGRATE_TLM( TIN=TSTART, TOUT=TEND, N=NDIM, &
            NNZERO=NNZ, Y_TLM=Y_TLM, Y=VAR, RTOL=RTOL, ATOL=ATOL, &
            NTLM=NTLM, FUN=FUN, JAC=JAC, RSTATUS_U=RSTATUS, &
            RCNTRL_U=RCNTRL, ISTATUS_U=ISTATUS, ICNTRL_U=ICNTRL, &
            ATOL_TLM=ATOL_TLM, RTOL_TLM=RTOL_TLM )
      
      write(6,*) "Solution", VAR(1:2)
      write(6,*) "Nfun", ISTATUS(1)
      write(6,*) "Njac", ISTATUS(2)
      write(6,*) "Nstp", ISTATUS(3)
      write(6,*) "Nacc", ISTATUS(4)
      write(6,*) "Nrej", ISTATUS(5)
      write(6,*) "Ndec", ISTATUS(6)
      write(6,*) "Nsol", ISTATUS(7)
      write(6,*) "Nsng", ISTATUS(8)

      write(6,*) "Sensitivities: Y_TLM"
      print *, "Y_TLM= ", Y_TLM
      
      END PROGRAM vdp_sdirk_tlm_dr
