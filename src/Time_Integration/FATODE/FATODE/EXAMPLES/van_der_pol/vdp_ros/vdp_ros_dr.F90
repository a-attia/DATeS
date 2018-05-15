!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE FUN(NVAR, T, Y, P)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      IMPLICIT NONE
      INTEGER ::NVAR
      DOUBLE PRECISION :: T, MU
      DOUBLE PRECISION :: Y(NVAR), P(NVAR)
      MU = 10.d0
      P(1) = Y(2)
      P(2) = MU*(1.0d0-Y(1)*Y(1))*Y(2)-Y(1)
      END SUBROUTINE FUN

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE JAC(neq, t, y, pd)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      integer neq
      double precision pd,t,y,mu
      dimension y(2), pd(2,2)

      mu = 10.d0
      pd(1,1) = 0.0d0
      pd(1,2) = 1.0d0
      pd(2,1) = (-2.0d0*mu*y(1)*y(2)-1.0d0)
      pd(2,2) = mu*(1.0d0-y(1)**2)

      END SUBROUTINE JAC

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE HESS( NVAR, T, Y, H )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      END SUBROUTINE HESS
     
 program vdp_ros_dr
     USE ROS_f90_integrator
     implicit none
     external fun,jac
     integer,parameter :: ndim = 2,nnz=0
     integer :: i,j,state
 
     double precision :: atol(ndim),rtol(ndim), var(ndim)
     double precision:: tstart, tend, t1, t2, tcpu
     double precision:: rstate(20),rcntrl(20)
     double precision :: a=30d0
     integer ::istate(20),icntrl(20)
     double precision :: Mu=1.0d0
 
       do i=1,ndim
         rtol(i) = 1.0d-5
         atol(i) = 1.0d-5
       end do

       VAR(1) = 2.0d0
       VAR(2) = -0.66d0

       tstart = 0.0d0
       tend = tstart + 20.0d0

       istate(:)=0
       icntrl(:)=0
       rcntrl(:)=0

        icntrl(1) = 0;
        icntrl(2) = 0;
        icntrl(3) = 0;
        icntrl(4) = 0;

       call cpu_time(t1)
       call integrate(tin=tstart, tout=tend, n=ndim, nnzero = nnz,var=var,&
                 rtol=rtol, atol=atol,fun=fun,jac=jac,rstatus_u=rstate,&
              rcntrl_u=rcntrl, istatus_u=istate, icntrl_u=icntrl)
       call cpu_time(t2)
       tcpu=t2-t1
     
     write(6,*) "Solution", VAR(1:2)
     write(6,*) "Nfun", ISTATE(1)
     write(6,*) "Njac", ISTATE(2)
     write(6,*) "Nstp", ISTATE(3)
     write(6,*) "Nacc", ISTATE(4)
     write(6,*) "Nrej", ISTATE(5)
     write(6,*) "Ndec", ISTATE(6)
     write(6,*) "Nsol", ISTATE(7)
     write(6,*) "Nsng", ISTATE(8)

 end program vdp_ros_dr
