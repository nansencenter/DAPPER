!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           calc.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Mathematical utilities.
!
! Description:    
!
! Revisions:

module calc_mod

  use utils_mod
  implicit none

  public arakawa, laplacian

contains
    
  subroutine arakawa(A, B, dx, dy, JACOBIAN)
    real(8), dimension(:, :), intent(in) :: A, B
    real(8), intent(in) :: dx, dy
    real(8), dimension(:, :), intent(inout) :: JACOBIAN
    integer m1, n1
    integer i, j
     
    m1 = size(A, 1) - 1
    n1 = size(A, 2) - 1

    do j = 2, m1
       do i = 2, n1
          JACOBIAN(j, i) = (A(j - 1, i) - A(j, i - 1)) * B(j - 1, i - 1) +&
               (A(j - 1, i - 1) + A(j - 1, i) - A(j + 1,i - 1) -A(j + 1, i)) * B(j, i - 1) +&
               (A(j, i - 1) - A(j + 1, i)) * B(j + 1,i - 1) +&
               (A(j - 1,i + 1) + A(j, i + 1) - A(j - 1,i - 1) - A(j, i - 1)) * B(j - 1, i) +&
               (A(j, i - 1) + A(j + 1, i - 1) - A(j, i + 1) - A(j + 1, i + 1)) * B(j + 1, i) +&
               (A(j, i + 1) - A(j - 1, i)) * B(j - 1,i + 1) +&
               (A(j + 1, i) + A(j + 1,i + 1) - A(j - 1, i) - A(j - 1,i + 1)) * B(j, i + 1) +&
               (A(j + 1, i) - A(j, i + 1)) * B(j + 1, i + 1)
       end do
    end do
    JACOBIAN(2 : m1, 2 : n1) = JACOBIAN(2 : m1, 2 : n1) / (12.0d0 * dx * dy)
  end subroutine arakawa

  subroutine laplacian(A, dx, dy, L)
    real(8), intent(in), dimension(:, :) :: A
    real(8), intent(in) :: dx, dy
    real(8), dimension(:, :), intent(inout) :: L
    real(8) :: dx2, dy2
    integer M, n
      
    M = size(A, 1)
    n = size(A, 2)

    dx2 = dx * dx
    dy2 = dy * dy

    L(2 : M - 1, 2 : n - 1) =&
         (A(1 : M - 2, 2 : n - 1) + A(3 : M, 2 : n - 1)) / dx2&
         + (A(2 : M - 1, 1 : n - 2) + A(2 : M - 1, 3 : n)) / dy2&
         - A(2 : M - 1, 2 : n - 1) * (2.0d0 / dx2 + 2.0d0 / dy2)

    ! L = 0 on the boundary
    !
    L(1, :) = 0.0d0
    L(M, :) = 0.0d0
    L(:, 1) = 0.0d0
    L(:, n) = 0.0d0
  end subroutine laplacian

end module calc_mod
