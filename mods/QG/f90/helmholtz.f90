!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           helmholtz.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Solver for Helmholtz equation.
!
! Description:    
!
! Revisions:
module helmholtz_mod

  use utils_mod

  implicit none

  save

  public helmholtz

  integer, private :: iq
  real(8), allocatable, dimension(:) :: q
  integer, parameter, private :: LEN20 = 20
  integer, dimension(LEN20), private :: nst, imx, jmx
  real(8), dimension(LEN20), private :: h

contains

  subroutine helmholtz(G, F, zk2, NX1, NY1, M, H1, NU1, NU2, NCYC, NX, MY, P)
    !
    ! Solves helmholtz equation:
    ! (\nabla^2 - zk2) P  = F, with Dirichlet B.C. The B.C. and the initial
    ! guess for the solution are given in the array G.
    !
    ! The dimension of the arrays should be:
    ! (NX, MY) = (p * 2^(M - 1) + 1, q * 2^(M - 1) + 1).
    ! p and q are small integers, equal to the number of grid boxes in the
    ! coarsest grid used in the multi-grid solution.
    !
    ! Note that number of boxes on a side of the grid is equal to the number
    ! of grid points on this side - 1.
    !
    ! In the code below p = NX1; q = NX2.                                       **
    ! M is the finest level of the multigrid. M=1 means use (p,q)     **
    ! boxes; M=2 means use twice as many boxes on each lide of grid.  **
    !
    ! H1 is the delx for the coarsest grid. (NU1,NU2) should be (1,2). **
    ! NCYCL should be 3-5 for a bad initial guess, and 1 for a good   **
    ! initial guess.                                                  **
    ! The dimension of the work array q should be NX * MY * 3.          **
    !
    real(8), intent(in), dimension(NX, MY) :: G, F
    real(8), intent(in) :: zk2
    integer, intent(in) :: NX1, NY1, M
    real(8), intent(in) :: H1
    integer, intent(in) :: NU1, NU2, NCYC, NX, MY
    real(8), intent(out), dimension(NX, MY) :: P

    integer :: ir
    integer :: k, ksq
    real(8) :: wu
    integer :: ic, km

    allocate(q(NX * MY * 3))

    iq = 1
    do k = 1, M
       ksq = 2 ** (k - 1)
       call grdfn(k, NX1 * ksq + 1, NY1 * ksq + 1, H1 / ksq)
       call grdfn(k + M, NX1 * ksq + 1, NY1 * ksq + 1, H1 / ksq)
    end do
    wu = 0.0d0
    call putf1(M, G, 0)
    call putf1(2 * M, F, 2)
    do ic = 1, NCYC
      do km = 1, M
         k = 1 + M - km
         if (k /= M) then
            call putz(k)
         end if
         do ir = 1, NU1
            call relax(k, k + M, wu, M, zk2)
         end do
         if (k > 1) then
            call rescal(k, k + M, k + M - 1, zk2)
         end if
      end do
      do k = 1, M
         do ir = 1, NU2
            call relax(k, k + M, wu, M, zk2)
         end do
         if (k < M) then
            call intadd(k, k + 1)
         end if
      end do
   end do
   call getsol(M, P)
   deallocate(q)
 end subroutine helmholtz

 subroutine grdfn(k, M, N, hh)
   integer, intent(in) :: k, M, N
   real(8), intent(in) :: hh

   nst(k) = iq
   imx(k) = M
   jmx(k) = N
   h(k) = hh
   iq = iq + M * N
 end subroutine grdfn

 subroutine key(k, ist, M, N, hh)
   integer, intent(in) :: k
   integer, dimension(:), intent(out) :: ist
   integer, intent(out) :: M, N
   real(8), intent(out) :: hh

   integer :: is, i

   M = imx(k)
   N = jmx(k)
   is = nst(k) - N - 1
   do i = 1, M
      is = is + N
      ist(i) = is
   end do
   hh = h(k)
 end subroutine key

 subroutine putf1(k, F, nh)
   integer, intent(in) :: k
   real(8), dimension(:, :), intent(in) :: F
   integer, intent(in) :: nh

   integer, dimension(200) :: ist

   integer :: ii, jj
   real(8) :: hh, hh2
   integer :: i, j

   call key (k, ist, ii, jj, hh)
   hh2 = hh ** nh
   do i = 1, ii
      do j = 1, jj
         q(ist(i) + j) = F(i, j) * hh2
      end do
   end do
 end subroutine putf1

 subroutine getsol(k, P)
   integer, intent(in) :: k
   real(8), dimension(:, :), intent(out) :: P

   integer, dimension(200) :: ist
   integer :: ii, jj
   real(8) :: hh
   integer :: i, j

   call key(k, ist, ii, jj, hh)
   do i = 1, ii
      do j = 1, jj
         P(i, j) = q(ist(i) + j)
      end do
   end do
 end subroutine getsol

 subroutine putz(k)
   integer, intent(in) :: k

   integer, dimension(200) :: ist
   integer :: ii, jj
   real(8) :: hh
   integer :: i, j

   call key( k, ist, ii, jj, hh)
   do  i = 1, ii
      do j = 1, jj
         q(ist(i) + j) = 0.0d0
      end do
   end do
 end subroutine putz

 subroutine relax(k, krhs, wu, M, zk2)
   integer, intent(in) :: k
   integer, intent(in) :: krhs
   real(8), intent(inout) :: wu
   integer, intent(in) :: M
   real(8), intent(in) :: zk2

   integer, dimension(200) :: ist, irhs
   integer :: ii, jj
   real(8) :: hh
   integer:: i1, j1, ir, iq, im, ip
   real(8) diag
   integer :: i, j
   real(8) :: a

   call key(k, ist, ii, jj, hh)
   call key(krhs, irhs, ii, jj, hh)
   diag = 4.0d0 + zk2 * hh ** 2
   i1 = ii - 1
   j1 = jj - 1
   do i = 2, i1
      ir = irhs(i)
      iq = ist(i)
      im = ist(i - 1)
      ip = ist(i + 1)
      do j = 2, j1
         a = q(ir + j) - q(iq + j + 1) - q(iq + j - 1) - q(im + j) - q(ip + j)
         q(iq + j) = - a / diag
      end do
   end do
   wu = wu + 4.0d0 ** (k - M)
 end subroutine relax

 subroutine intadd(kc, kf)
   integer, intent(in) :: kc
   integer, intent(in) :: kf

   integer, dimension(200) :: istc, istf
   real(8) :: hc, hf
   integer :: iic, jjc, iif, jjf
   integer :: ic, jc, if, jf
   integer :: ifo, ifm, ico, icm
   real(8) :: a, am

   call key(kc, istc, iic, jjc, hc)
   call key(kf, istf, iif, jjf, hf)
   do ic = 2, iic
      if = 2 * ic - 1
      jf = 1
      ifo = istf(if)
      ifm = istf(if - 1)
      ico = istc(ic)
      icm = istc(ic - 1)
      do jc = 2, jjc
         jf = jf + 2
         a = 0.5d0 * (q(ico + jc) + q(ico + jc - 1))
         am = 0.5d0 * (q(icm + jc) + q(icm + jc - 1))
         q(ifo+jf) = q(ifo + jf) + q(ico + jc)
         q(ifm + jf) = q(ifm + jf) + 0.5d0 * (q(ico + jc) + q(icm + jc))
         q(ifo + jf - 1) = q(ifo + jf - 1) + a
         q(ifm + jf - 1)  =  q(ifm + jf - 1) + 0.5d0 * (a + am)
      end do
   end do
 end subroutine intadd

 subroutine rescal(kf, krf, krc, zk2)
   integer, intent(in) :: kf, krf, krc
   real(8), intent(in) :: zk2

   integer, dimension(200) :: iuf, irf, irc
   integer :: iif, jjf, iic, jjc, iic1, jjc1
   integer :: if, ic, jf, jc
   integer :: icr, ifo, ifm, ifr, ifp
   real(8) :: hf, hc
   real(8) :: s
   real(8) :: diag

   call key(kf, iuf, iif, jjf, hf)
   call key(krf, irf, iif, jjf, hf)
   call key(krc, irc, iic, jjc, hc)

   diag = 4.0d0 + zk2 * hf ** 2
   iic1 = iic - 1
   jjc1 = jjc - 1
   do ic = 2, iic1
      icr = irc(ic)
      if = 2 * ic - 1
      jf = 1
      ifr = irf(if)
      ifo = iuf(if)
      ifm = iuf(if - 1)
      ifp = iuf(if + 1)
      do jc = 2, jjc1
         jf = jf + 2
         s = q(ifo + jf + 1) + q(ifo + jf - 1) + q(ifm + jf) + q(ifp + jf)
         q(icr + jc) = 4.0d0 * (q(ifr + jf) - s + diag * q(ifo + jf))
      end do
   end do
 end subroutine rescal

end module helmholtz_mod
