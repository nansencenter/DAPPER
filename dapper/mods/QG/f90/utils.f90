!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           utils.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Utilities.
!
! Description:    
!
! Revisions:

module utils_mod

  implicit none

  save

  public wopen, strcmp

  ! Define number for standard output file
  integer, parameter, public :: stdout = 6
  ! Define default verbosity
  integer, parameter, public :: VERBOSE_DEF = 1
  ! String length
  integer, parameter, public :: STRLEN = 1024

contains

  subroutine wopen(fname, fid, mode)
    !
    ! Wrapper to open().
    !
    character(*) :: fname
    integer :: fid
    character(*) :: mode
    integer ios

    open(unit = fid, file = trim(fname), status = trim(mode), iostat = ios)
    if (ios /= 0) then
       write(stdout, *) '  qg: error: wopen(): could not open ' // trim(fname) // ' in mode = ' //  mode
       call flush(stdout)
       stop
    end if
  end subroutine wopen

  function strcmp(string1, string2)
    !
    ! Compares two strings, a-la C
    !
    integer :: strcmp
    character(*) :: string1
    character(*) :: string2

    if (LGT(trim(string1), trim(string2))) then
       strcmp = 1
    else if (LGT(trim(string2), trim(string1))) then
       strcmp = -1
    else
       strcmp = 0
    end if
  end function strcmp

end module utils_mod
