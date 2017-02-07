!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           qgstep_mex.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Interface to Matlab.
!
! Description:    
!
! Revisions:

subroutine mexFunction(nlhs, plhs, nrhs, prhs)
  use mexf90_mod
  use utils_mod, only: STRLEN
  use parameters_mod

  implicit none

  integer(4), intent(in) :: nlhs, nrhs
  integer(8), intent(in), dimension(*) :: prhs
  integer(8), intent(out), dimension(*) :: plhs
  integer(8), pointer :: t, PSI
  character(STRLEN) :: prmfname
  integer(8) :: typeid
  integer(8) :: status
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

  if(nrhs /= 3) then
     call mexErrMsgTxt('Error: mexFunction(): requires 3 input arguments')
     stop
  end if

!  typeid = mxGetClassName(prhs(1))
!  write(stdout, *) "typeid =", typeid

  if (mxIsDouble(prhs(1)) .ne. 1) then
     call mexErrMsgTxt('Error: qgflux(): mexFunction(): arg #1 is not a numeric.')
  end if

  if (mxIsDouble(prhs(2)) .ne. 1) then
     call mexErrMsgTxt('Error: qgflux(): mexFunction(): arg #2 is not a numeric array.')
  end if

  if (mxIsChar(prhs(3)) .ne. 1) then
     call mexErrMsgTxt('Error: qgflux(): mexFunction(): arg #3 is not a string.')
  endif
 
  t => mxGetPr(prhs(1))
  PSI => mxGetPr(prhs(2))

  status = mxGetString(prhs(3), prmfname, STRLEN)
  if (status /= 0) then
     call mexErrMsgTxt('Error: qgflux(): mexFunction(): arg #3: string length must be less than STRLEN.')
  end if
  call wopen(prmfname, 111, 'old')
  read(111, parameters)
  close(111)

  dx = 1.0d0 / dble(M - 1)
  dy = 1.0d0 / dble(N - 1)

  do j = 1, N
     CURLT(:, j) = - 2.0d0 * PI * sin(2.0d0 * PI * (j - 1) / (N - 1))
  end do

  call my(t, PSI)
end subroutine mexFunction

subroutine my(t, PSI)
  use parameters_mod, only: M, N, scheme
  use qgstep_mod

  real(8), dimension(1), intent(inout) :: t
  real(8), dimension(M, N), intent(inout) :: PSI
  real(8), dimension(M, N) :: Q
  real(8) :: tstop

  tstop = t(1) + dtout

  call laplacian(PSI, dx, dy, Q)
  Q = Q - F * PSI

  do while (t(1) < tstop)
     if (strcmp(scheme, '2ndorder') == 0) then
        call qg_step_2ndorder(t(1), PSI, Q)
     elseif (strcmp(scheme, 'rk4') == 0) then
        call qg_step_rk4(t(1), PSI, Q)
     elseif (strcmp(scheme, 'dp5') == 0) then
        call qg_step_dp5(t(1), PSI, Q)
     else
        write(stdout, *) 'Error: unknown scheme "', trim(scheme), '"'
        stop
     end if
  end do
  call calc_psi(PSI, Q, PSI)
end subroutine my
