!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           mexf90.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. Fortran interface to Matlab
!                 interface to Fortran, sorry for that.
!
! Description:    
!
! Revisions:

module mexf90_mod
  interface
     function mxGetPr(pm)
       integer(8), pointer :: mxGetPr
       integer(8) :: pm
     end function mxGetPr

     function mxGetM(pm)
       integer(8) :: mxGetM
       integer(8) :: pm
     end function mxGetM

     function mxGetN(pm)
       integer(8) :: mxGetN
       integer(8) :: pm
     end function mxGetN

     function mxCreateDoubleMatrix(m, n, type)
       integer(8) :: mxCreateDoubleMatrix
       integer(8) :: m,n,type
     end function mxCreateDoubleMatrix

     function mxGetScalar(pm)
       integer(8) :: pm
       double precision :: mxGetScalar
     end function mxGetScalar

     function mxIsNumeric(p)
       integer(8) :: mxIsNumeric
       integer(8) :: p
     end function mxIsNumeric

     function mxIsDouble(p)
       integer(8) :: mxIsDouble
       integer(8) :: p
     end function mxIsDouble

     function mxIsChar(p)
       integer(8) :: mxIsChar
       integer(8) :: p
     end function mxIsChar

     function mxGetString(p, string, STRLEN)
       integer(8) :: mxGetString
       integer(8) :: p
       character*(*) :: string
       integer(4) :: STRLEN
     end function mxGetString
  end interface
end module mexf90_mod
