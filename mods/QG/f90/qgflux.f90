!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           qgflux.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Time derivative.
!
! Description:    
!
! Revisions:

module qgflux_mod

  use utils_mod
  use parameters_mod
  use calc_mod, only: arakawa, laplacian
  use helmholtz_mod, only: helmholtz

  implicit none

  public qg_flux

contains

  subroutine calc_psi(PSIGUESS, Q, PSI)
    real(8), dimension(:, :), intent(in) :: PSIGUESS, Q
    real(8), dimension(:, :), intent(out) :: PSI

    call helmholtz(PSIGUESS, Q, F, NX1, NY1, MREFIN, 1.0d0 / NX1, 1, 2, 3, N, M, PSI)
  end subroutine calc_psi

  subroutine qg_flux(t, Q, PSIGUESS, PSI, QFLUX)
    real(8), intent(in) :: t
    real(8), dimension(:, :), intent(in) :: Q, PSIGUESS
    real(8), dimension(:, :), intent(out) :: PSI, QFLUX

    real(8), dimension(N, M) :: JACOBIAN
    real(8), dimension(N, M) :: ZETA, ZETA2, ZETA4

    call calc_psi(PSIGUESS, Q, PSI)
    call arakawa(PSI, Q, dx, dy, JACOBIAN)
    call laplacian(PSI, dx, dy, ZETA)
    call laplacian(ZETA, dx, dy, ZETA2)
    call laplacian(ZETA2, dx, dy, ZETA4)

    QFLUX = - r * JACOBIAN - rkb * ZETA + rkh * ZETA2 - rkh2 * ZETA4 + CURLT
    QFLUX(2 : N - 1, 2 : M - 1) = QFLUX(2 : N - 1, 2 : M - 1)&
         - (0.5d0 / dx) * (PSI(3 : N, 2 : M - 1) - PSI(1 : N - 2, 2 : M - 1))
    QFLUX(1, :) = 0.0d0
    QFLUX(N, :) = 0.0d0
    QFLUX(:, 1) = 0.0d0
    QFLUX(:, M) = 0.0d0
  end subroutine qg_flux

  subroutine qg_flux_tl(t, Q0, PSI0, Q, PSIGUESS, PSI, QFLUX)
    real(8), intent(in) :: t
    real(8), dimension(:, :), intent(in) :: Q, PSIGUESS
    real(8), dimension(:, :), intent(in) :: Q0, PSI0
    real(8), dimension(:, :), intent(out) :: PSI, QFLUX

    real(8), dimension(N, M) :: J1, J2
    real(8), dimension(N, M) :: ZETA, ZETA2, ZETA4

    call calc_psi(PSIGUESS, Q, PSI)
    call arakawa(PSI0, Q, dx, dy, J1)
    call arakawa(PSI, Q0, dx, dy, J2);
    call laplacian(PSI, dx, dy, ZETA)
    call laplacian(ZETA, dx, dy, ZETA2)
    call laplacian(ZETA2, dx, dy, ZETA4)

    QFLUX = - r * J1 - r * J2 - rkb * ZETA + rkh * ZETA2 - rkh2 * ZETA4
    QFLUX(2 : N - 1, 2 : M - 1) = QFLUX(2 : N - 1, 2 : M - 1)&
         - (0.5d0 / dx) * (PSI(3 : N, 2 : M - 1) - PSI(1 : N - 2, 2 : M - 1))
    QFLUX(1, :) = 0.0d0
    QFLUX(N, :) = 0.0d0
    QFLUX(:, 1) = 0.0d0
    QFLUX(:, M) = 0.0d0
  end subroutine qg_flux_tl

end module qgflux_mod
