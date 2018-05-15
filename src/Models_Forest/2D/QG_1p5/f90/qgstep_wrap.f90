
subroutine qgStepFunction(t, PSI, prmfname)
  use utils_mod, only: STRLEN
  use parameters_mod

  implicit none
  real(8), intent(inout) :: t
  real(8), dimension(M, N), intent(inout) :: PSI
  character(STRLEN) :: prmfname
  integer(8) :: j

  namelist /parameters/ &
       tend, &
       dtout, &
       dt, &
       rkb, &
       rkh, &
       rkh2, &
       F, &
       r, &
       verbose, &
       scheme, &
       rstart, &
       restartfname, &
       outfname

  call wopen(prmfname, 111, 'old')
  read(111, parameters)
  close(111)

  dx = 1.0d0 / dble(M - 1)
  dy = 1.0d0 / dble(N - 1)

  do j = 1, N
     CURLT(:, j) = - 2.0d0 * PI * sin(2.0d0 * PI * (j - 1) / (N - 1))
  end do

  call my(t, PSI)
end subroutine qgStepFunction

subroutine my(t, PSI)
  use parameters_mod, only: M, N, scheme
  use qgstep_mod

  real(8), intent(inout) :: t
  real(8), dimension(M, N), intent(inout) :: PSI
  real(8), dimension(M, N) :: Q
  real(8) :: tstop

  ! find the final time of model propagation
  ! I don't like how this is dependent on dtout and not tend!
  tstop = t + dtout  ! 
  ! if (tend == 0) then
  !   tstop = t + dtout
  ! elseif (tend > t) then
  !   tstop = tend
  ! else
  !   tstop = t + tend  ! if tend is smaller than current time, final time is set to current time + tend.
  ! end if

  call laplacian(PSI, dx, dy, Q)
  Q = Q - F * PSI

  do while (t < tstop)
     if (strcmp(scheme, '2ndorder') == 0) then
        call qg_step_2ndorder(t, PSI, Q)
     elseif (strcmp(scheme, 'rk4') == 0) then
        call qg_step_rk4(t, PSI, Q)
     elseif (strcmp(scheme, 'dp5') == 0) then
        call qg_step_dp5(t, PSI, Q)
     else
        write(stdout, *) 'Error: unknown scheme "', trim(scheme), '"'
        stop
     end if
  end do
  call calc_psi(PSI, Q, PSI)
end subroutine my
