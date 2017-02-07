!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           qg.f90
!
! Created:        24/05/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
! 
! Purpose:        Runs QG model.
!
! History:        24 May 2007: Downloaded the original code from 
!                 http://www.seas.harvard.edu/climate/eli/Downloads/QG200205011455.tar.gz
!
! Description:   The solved equation represents ocean circulation driven by a
!                specified wind stress, showing in a time average two gyres
!                with a "Gulf Stream" on the left side of the domain:
!      
!\documentclass[intlimits,fleqn, 12pt]{article}
!\usepackage{amsmath}
!\newcommand{\mb} {\mathbf}
!\begin{document}
!\begin{align*}
!  q_t = - \psi_x - r J(\psi, q) - rkb \zeta + rkh \nabla^2 \zeta 
!  - rkh2 \nabla^4 \zeta + 2 \pi \sin(2\pi y),
!\end{align*}
!where
!\begin{align*}
!  &q = \zeta - F \psi \ & &\text{is the potential vorticity},\\
!  &\zeta = \nabla^2 \psi & &\text{-- the relative corticity},\\
!  &J(q, \psi) = q_x \psi_y - q_y \psi_x,\\
!  &r & &\text{-- a multiple for the nonlinear advection term,}\\
!  &&& \text{\ \ "sort of Rossby number"},\\
!  &rkb & &\text{-- bottom friction},\\
!  &rkh & &\text{-- horizontal friction},\\
!  &rkh2 & &\text{-- biharmonic horizontal friction},\\
!  &F & &\text{-- Froud number}.
!\end{align*}
!\end{document}

program qg

  use parameters_mod
  use data_mod
  use qgstep_mod

  implicit none

  character(STRLEN) :: prmfname
  real(8), dimension(N, M) :: Q, PSI, PSIGUESS
  real(8) :: t
  real(8) :: qmax
  integer :: nstep, step

  if (iargc() /= 1) then
     write(stdout,*) 'Usage: qg <parameter file>'
     stop
  end if
  call getarg(1, prmfname)
  if (verbose > 0) then
     write(stdout, *) 'parameter file = "' // trim(prmfname) // '"'
  end if

  call parameters_read(prmfname)
  call data_read(t, PSI)
  call data_writeopen

  if (verbose > 0) then
     write(stdout, *) 'starting the model:'
     write(stdout, *) '  N x M =', N, 'x', M
  end if

  if (tend < t) then
     tend = t + tend;
     if (verbose > 0) then
        write(stdout, *) '  readjusted the end time: tend =', tend
     end if
  end if

  nstep = floor((tend - t) / dt)

  if (verbose > 0) then
     write(stdout, *) '  nstep =', nstep
  end if

  ! start up
  !
  call laplacian(PSI, dx, dy, Q)
  Q = Q - F * PSI

  if (verbose > 0) then
     write(stdout, '(A)', advance = 'no') ' main cycle:'
  end if

  PSIGUESS = PSI
  if (floor(t / dtout) - floor((t - dt) / dtout) /= 0) then
     call calc_psi(PSIGUESS, Q, PSI)
     call data_write(t, PSI, Q)
  end if

  ! main cycle
  !
  do step = 1, nstep

     ! check for instability
     !
     qmax = maxval(Q)
     if (qmax > 1.0e+20) then
        write(stdout, *) '  qg: error: qmax > 1.0e+20 at t =', t
        stop
     end if

     if (strcmp(scheme, '2ndorder') == 0) then
        call qg_step_2ndorder(t, PSIGUESS, Q)
     elseif (strcmp(scheme, 'rk4') == 0) then
        call qg_step_rk4(t, PSIGUESS, Q)
     elseif (strcmp(scheme, 'dp5') == 0) then
        call qg_step_dp5(t, PSIGUESS, Q)
     else
        write(stdout, *) '  qg: error: unknown scheme "', trim(scheme), '"'
        stop
     end if

     if (floor(t / dtout) - floor((t - dt) / dtout) /= 0) then
        call calc_psi(PSIGUESS, Q, PSI)
        call data_write(t, PSI, Q)
     end if
  end do

  call data_writeclose

end program qg
