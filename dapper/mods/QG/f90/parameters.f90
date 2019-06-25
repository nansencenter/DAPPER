!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           parameters.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Reads and stores the model
!                 parameters.
!
! Description:    
!
! Revisions:

module parameters_mod

  use utils_mod
  implicit none

  save

  public parameters_read

  integer, parameter, private :: FID_PRM = 111

  ! This is the place to change the resolution of the model, by
  ! changing the parameter MREFIN:
  ! To make the ocean a rectangular (at your own risk!) change nx1,
  ! ny1 to (2,4), and adjust domain size (Lx, Ly)...
  !
  integer, parameter, public :: MREFIN = 7
  integer, parameter, public :: NX1 = 2
  integer, parameter, public :: NY1 = 2
  integer, parameter, public :: M = NY1 * 2 ** (MREFIN - 1) + 1
  integer, parameter, public :: N = NX1 * 2 ** (MREFIN - 1) + 1

  real(8), parameter, public :: PI = 3.14159265358979323d0

  real(8), parameter, public :: rf_coeff = 0.1 ! Roberts filter coefficient
                                               ! for the leap-frog scheme

  real(8), public :: dt            ! time step
  real(8), public :: rkb           ! bottom friction
  real(8), public :: rkh           ! horizontal friction
  real(8), public :: rkh2          ! biharmonic friction
  real(8), public :: F             ! Froud number, (L / Rd)^2
  real(8), public :: r             ! factor in front of nonlinear
                                   ! advection term
  integer, public :: rstart = 0
  real(8), public :: tend
  real(8), public :: dtout
  integer, public :: verbose = VERBOSE_DEF
  character(STRLEN), public :: scheme
  character(STRLEN), public :: restartfname
  character(STRLEN), public :: outfname

  ! secondary parameters

  real(8), dimension(N, M), public :: CURLT

  real(8), public :: dx
  real(8), public :: dy

contains

  subroutine parameters_read(prmfname)
    character(STRLEN) :: prmfname
    integer :: j
    integer :: m1 = M - 1

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

    call wopen(trim(prmfname), FID_PRM, 'old')
    read(FID_PRM, parameters)

    if (verbose > 0) then
       write(stdout, *) 'parameters:'
       write(stdout, *) '  tend =', tend
       write(stdout, *) '  dtout =', dtout
       write(stdout, *) '  dt =', dt
       write(stdout, *) '  RKB =', rkb
       write(stdout, *) '  RKH =', rkh
       write(stdout, *) '  RKH2 =', rkh2
       write(stdout, *) '  F =', F
       write(stdout, *) '  r =', r
       write(stdout, *) '  verbose =', verbose
       write(stdout, *) '  scheme = "', trim(scheme), '"'
       write(stdout, *) '  rstart =', rstart
       write(stdout, *) '  restart file = "', trim(restartfname), '"'
       write(stdout, *) '  output file = "', trim(outfname), '"'
    end if

    dx = 1.0d0 / dble(N - 1)
    dy = 1.0d0 / dble(M - 1)

    do j = 1, M
       CURLT(:, j) = - 2.0d0 * PI * sin(2.0d0 * PI * (j - 1) / m1)
    end do

    close(FID_PRM)
  end subroutine parameters_read

  subroutine parameters_get(N_out, M_out, tend_out, dtout_out, dt_out, &
       rkb_out, rkh_out, rkh2_out, F_out, r_out, verbose_out, &
       scheme_out, rstart_out, restartfname_out, outfname_out)

    integer, intent(out) :: N_out, M_out
    real(8), intent(out) :: dt_out
    real(8), intent(out) :: rkb_out
    real(8), intent(out) :: rkh_out
    real(8), intent(out) :: rkh2_out
    real(8), intent(out) :: F_out
    real(8), intent(out) :: r_out
    real(8), intent(out) :: tend_out
    real(8), intent(out) :: dtout_out
    integer, intent(out) :: verbose_out
    character(STRLEN), intent(out) :: scheme_out
    integer, intent(out) :: rstart_out
    character(STRLEN), intent(out) :: restartfname_out
    character(STRLEN), intent(out) :: outfname_out
    
    N_out = N
    M_out = M
    tend_out = tend
    dtout_out = dtout
    dt_out = dt
    rkb_out = rkb
    rkh_out = rkh
    rkh2_out = rkh2
    F_out = F
    r_out = r
    verbose_out = verbose
    scheme_out = scheme
    rstart_out = rstart
    restartfname_out = restartfname
    outfname_out = outfname
  end subroutine parameters_get
       
end module parameters_mod
