!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           qgstep.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Integrators.
!
! Description:    
!
! Revisions:

module qgstep_mod
  use parameters_mod, only: dt, M, N
  use qgflux_mod

contains

  subroutine qg_step_1storder(t, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX
    real(8), dimension(N, M) :: PSI1

    call qg_flux(t, Q, PSI, PSI1, QFLUX)
    PSI = PSI1
    Q = Q + dt * QFLUX
  end subroutine qg_step_1storder

  subroutine qg_step_1storder_tl(t, PSI0, Q0, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q0, PSI0
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX0
    real(8), dimension(N, M) :: QFLUX
    real(8), dimension(N, M) :: PSI01
    real(8), dimension(N, M) :: PSI1

    call qg_flux(t, Q0, PSI0, PSI01, QFLUX0)
    call qg_flux_tl(t, Q0, PSI0, Q, PSI, PSI1, QFLUX)
    PSI0 = PSI01
    PSI = PSI1
    Q0 = Q0 + dt * QFLUX0
    Q = Q + dt * QFLUX
  end subroutine qg_step_1storder_tl

  subroutine qg_step_2ndorder(t, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX
    real(8), dimension(N, M) :: PSI1
    real(8), dimension(N, M) :: Q2, Q3

    call qg_flux(t, Q, PSI, PSI1, QFLUX)
    Q2 = Q + (0.5d0 * dt) * QFLUX
    Q3 = Q + dt * QFLUX
    call qg_flux(t + 0.5d0 * dt, Q2, PSI1, PSI, QFLUX)
    Q2 = Q2 + (0.5d0 * dt) * QFLUX

    t = t + dt
    Q = 2.0d0 * Q2 - Q3
  end subroutine qg_step_2ndorder

  subroutine qg_step_2ndorder_tl(t, PSI0, Q0, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q0, PSI0
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX0
    real(8), dimension(N, M) :: QFLUX
    real(8), dimension(N, M) :: PSI01
    real(8), dimension(N, M) :: PSI1
    real(8), dimension(N, M) :: Q02, Q03
    real(8), dimension(N, M) :: Q2, Q3

    call qg_flux(t, Q0, PSI0, PSI01, QFLUX0)
    call qg_flux_tl(t, Q0, PSI0, Q, PSI, PSI1, QFLUX)
    Q02 = Q0 + (0.5d0 * dt) * QFLUX0
    Q2 = Q + (0.5d0 * dt) * QFLUX
    Q03 = Q0 + dt * QFLUX0
    Q3 = Q + dt * QFLUX
    call qg_flux(t + 0.5d0 * dt, Q02, PSI01, PSI0, QFLUX0)
    call qg_flux_tl(t + 0.5d0 * dt, Q02, PSI01, Q2, PSI1, PSI, QFLUX)
    Q02 = Q02 + (0.5d0 * dt) * QFLUX0
    Q2 = Q2 + (0.5d0 * dt) * QFLUX

    t = t + dt
    Q0 = 2.0d0 * Q02 - Q03
    Q = 2.0d0 * Q2 - Q3
  end subroutine qg_step_2ndorder_tl

  subroutine qg_step_rk4(t, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX1, QFLUX2, QFLUX3, QFLUX4
    real(8), dimension(N, M) :: PP
    real(8), dimension(N, M) :: Q2, Q3, Q4
    real(8) :: tt

    ! Given vorticity Q, this call calculates its flux QFLUX1. 
    ! Solves for PSI1 as a by-product, using PSI as the first guess
    !
    call qg_flux(t, Q, PSI, PP, QFLUX1)
    tt = t + 0.5d0
    Q2 = Q + (0.5d0 * dt) * QFLUX1
    call qg_flux(tt, Q2, PP, PSI, QFLUX2)
    Q3 = Q + (0.5d0 * dt) * QFLUX2
    call qg_flux(tt, Q3, PSI, PP, QFLUX3)
    Q4 = Q + dt * QFLUX3
    tt = t + dt
    call qg_flux(tt, Q4, PP, PSI, QFLUX4)

    t = t + dt
    Q = Q + (QFLUX1 + 2.0d0 * (QFLUX2 + QFLUX3) + QFLUX4) * (dt / 6.0d0)
  end subroutine qg_step_rk4

  subroutine qg_step_rk4_tl(t, PSI0, Q0, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q0, PSI0
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX01, QFLUX02, QFLUX03, QFLUX04
    real(8), dimension(N, M) :: QFLUX1, QFLUX2, QFLUX3, QFLUX4
    real(8), dimension(N, M) :: PP0
    real(8), dimension(N, M) :: PP
    real(8), dimension(N, M) :: Q02, Q03, Q04
    real(8), dimension(N, M) :: Q2, Q3, Q4
    real(8) :: tt

    ! Given vorticity Q, this call calculates its flux QFLUX1. 
    ! Solves for PSI1 as a by-product, using PSI as the first guess
    !
    call qg_flux(t, Q0, PSI0, PP0, QFLUX01)
    call qg_flux_tl(t, Q0, PP0, Q, PSI, PP, QFLUX1)
    tt = t + 0.5d0
    Q02 = Q0 + (0.5d0 * dt) * QFLUX01
    Q2 = Q + (0.5d0 * dt) * QFLUX1
    call qg_flux(tt, Q02, PP0, PSI0, QFLUX02)
    call qg_flux_tl(tt, Q02, PP0, Q2, PP, PSI, QFLUX2)
    Q03 = Q0 + (0.5d0 * dt) * QFLUX02
    Q3 = Q + (0.5d0 * dt) * QFLUX2
    call qg_flux(tt, Q03, PSI0, PP0, QFLUX03)
    call qg_flux_tl(tt, Q03, PP0, Q3, PSI, PP, QFLUX3)
    Q04 = Q0 + dt * QFLUX03
    Q4 = Q + dt * QFLUX3
    tt = t + dt
    call qg_flux(tt, Q04, PP0, PSI0, QFLUX04)
    call qg_flux_tl(tt, Q04, PP0, Q4, PP, PSI, QFLUX4)

    t = t + dt
    Q0 = Q0 + (QFLUX01 + 2.0d0 * (QFLUX02 + QFLUX03) + QFLUX04) * (dt / 6.0d0)
    Q = Q + (QFLUX1 + 2.0d0 * (QFLUX2 + QFLUX3) + QFLUX4) * (dt / 6.0d0)
  end subroutine qg_step_rk4_tl

  subroutine qg_step_dp5(t, PSI, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: Q, PSI
    real(8), dimension(N, M) :: QFLUX1, QFLUX2, QFLUX3, QFLUX4, QFLUX5
    real(8), dimension(N, M) :: QQ, PP
    real(8) :: tt

    call qg_flux(t, Q, PSI, PP, QFLUX1)
    tt = t + 0.2d0 * dt
    QQ = Q + (0.2d0 * dt) * QFLUX1
    call qg_flux(tt, QQ, PP, PSI, QFLUX2)
    tt = t + 0.3d0 * dt
    QQ = Q + ((3.0d0 / 40.0d0) * dt) * QFLUX1&
         + ((9.0d0 / 40.0d0) * dt) * QFLUX2
    call qg_flux(tt, QQ, PSI, PP, QFLUX3)
    tt = t + 0.8d0 * dt
    QQ = Q + ((44.0d0 / 45.0d0) * dt) * QFLUX1&
         - ((56.0d0 / 15.0d0) * dt) * QFLUX2&
         + ((32.0d0 / 9.0d0) * dt) * QFLUX3
    call qg_flux(tt, QQ, PP, PSI, QFLUX4)
    tt = t + (8.0d0 / 9.0d0) * dt
    QQ = Q + ((19372.0d0 / 6561.0d0)  * dt) * QFLUX1&
         - ((25360.0d0 / 2187.0d0) * dt) * QFLUX2&
         + ((64448.0d0 / 6561.0d0) * dt) * QFLUX3&
         - ((212.0d0 / 729.0d0) * dt) * QFLUX4
    call qg_flux(tt, QQ, PSI, PP, QFLUX5)
    tt = t + dt
    QQ = Q + ((9017.0d0 / 3168.0d0) * dt) * QFLUX1&
         - ((355.0d0 / 33.0d0) * dt) * QFLUX2&
         + ((46732.0d0 / 5247.0d0) * dt) * QFLUX3&
         + ((49.0d0 / 176.0d0) * dt) * QFLUX4&
         - ((5103.0d0 / 18656.0d0) * dt) * QFLUX5
    call qg_flux(tt, QQ, PP, PSI, QFLUX2)

    t = t + dt
    Q = Q + ((35.0d0 / 384.0d0) * dt) * QFLUX1&
         + ((500.0d0 / 1113.0d0) * dt) * QFLUX3&
         + ((125.0d0 / 192.0d0) * dt) * QFLUX4&
         - ((2187.0d0 / 6784.0d0) * dt) * QFLUX5&
         + ((11.0d0 / 84.0d0) * dt) * QFLUX2
  end subroutine qg_step_dp5

  subroutine qg_step_lf(t, PSIGUESS, QOLD, Q)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: PSIGUESS, QOLD, Q
    real(8), dimension(N, M) :: PSI
    real(8), dimension(N, M) :: QNEW, QFLUX

    call qg_flux(t, Q, PSIGUESS, PSI, QFLUX)
    QNEW = QOLD + (2.0d0 * dt) * QFLUX
    ! Robert filter
    Q = (1.0d0 - 2.0d0 * rf_coeff) * Q + rf_coeff * (QOLD + QNEW)

    t = t + dt
    PSIGUESS = PSI
    QOLD = Q
    Q = QNEW
  end subroutine qg_step_lf

end module qgstep_mod
