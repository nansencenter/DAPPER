!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           data.f90
!
! Created:        31/08/2007
!
! Last modified:  08/02/2008
!
! Author:         Pavel Sakov
!                 CSIRO Marine and Atmospheric Research
!                 NERSC
!
! Purpose:        Fortran code for QG model. I/O.
!
! Description:    
!
! Revisions:

module data_mod

  use parameters_mod
  use utils_mod
  use nfw_mod
  implicit none

  public data_read, data_write, data_writeopen, data_writeclose

  integer, private :: nc_out
  integer, private :: dimids(3)
  integer, private :: id_t, id_q, id_psi

contains

  subroutine data_read(t, PSI)
    real(8), intent(out) :: t
    real(8), dimension(N, M), intent(out) :: PSI
    integer :: nc_init
    integer :: rid, iid, jid, tid, psiid
    integer :: nr, ni, nj
    integer :: start(3), length(3)
    real(8) :: tt(1)

    if (rstart /= 0) then
       if (verbose > 0) then
          write(stdout, *) 'reading record #', rstart, 'from "', trim(restartfname), '":'
       end if
       call nfw_open(trim(restartfname), nf_nowrite, nc_init)
       call nfw_inq_dimid(trim(restartfname), nc_init, 'record', rid)
       call nfw_inq_dimlen(trim(restartfname), nc_init, rid, nr)
       if (nr <= rstart) then
          write(stdout, *) '  qg: error: data_read(): could not open dump r =', rstart, ' in'
          write(stdout, *) '  "', trim(restartfname), '": only', nr, 'dumps in the file'
          stop
       end if
       call nfw_inq_dimid(trim(restartfname), nc_init, 'i', iid)
       call nfw_inq_dimlen(trim(restartfname), nc_init, iid, ni)
       call nfw_inq_dimid(trim(restartfname), nc_init, 'j', jid)
       call nfw_inq_dimlen(trim(restartfname), nc_init, jid, nj)
       if (ni /= M .or. nj /= N) then
          write(stdout, *) '  qg: error: data_read(): grid dimensions do not match'
          write(stdout, *) '  model:', M, 'x', N
          write(stdout, *) '  "', trim(restartfname), '":', ni, 'x', nj
          stop
       end if
       call nfw_inq_varid(trim(restartfname), nc_init, 't', tid)
       start(1) = rstart
       length(1) = 1
       call nfw_get_vara_double(trim(restartfname), nc_init, tid, start, length, tt)
       t = tt(1)
       call nfw_inq_varid(trim(restartfname), nc_init, 'psi', psiid)
       start(1) = 1
       start(2) = 1
       start(3) = rstart
       length(1) = M
       length(2) = N
       length(3) = 1
       call nfw_get_vara_double(trim(restartfname), nc_init, psiid, start, length, PSI)
       call nfw_close(trim(restartfname), nc_init)
       if (verbose > 0) then
          write(stdout, *) '  done, start time =', t
       end if
    else    
       t = 0.0d0
       PSI = 0.0d0
    end if
  end subroutine data_read

  subroutine data_writeopen
    integer :: itmp(1)
    real(8) :: rtmp(1)
    integer :: strlen

    call nfw_create(trim(outfname), nf_clobber, nc_out)

    ! write the run parameters
    !
    rtmp(1) = real(dt)
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'dt', nf_float, 1, rtmp)
    rtmp(1) = rkb
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'rkb', nf_float, 1, rtmp)
    rtmp(1) = rkh
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'rkh', nf_float, 1, rtmp)
    rtmp(1) = rkh2
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'rkh2', nf_float, 1, rtmp)
    rtmp(1) = F
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'F', nf_float, 1, rtmp)
    rtmp(1) = R
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'R', nf_float, 1, rtmp)
    rtmp(1) = rf_coeff
    call nfw_put_att_double(trim(outfname), nc_out, nf_global, 'rf_coeff', nf_float, 1, rtmp)
    strlen = len_trim(scheme)
    call nfw_put_att_text(trim(outfname), nc_out, nf_global, 'scheme', strlen, scheme)
    strlen = len_trim(restartfname)
    call nfw_put_att_text(trim(outfname), nc_out, nf_global, 'restartfname', strlen, restartfname)
    itmp(1) = rstart
    call nfw_put_att_int(trim(outfname), nc_out, nf_global, 'rstart', nf_int, 1, itmp)
  

    ! create dimensions and variables
    !
    call nfw_def_dim(trim(outfname), nc_out, 'i', N, dimids(1))
    call nfw_def_dim(trim(outfname), nc_out, 'j', M, dimids(2))
    call nfw_def_dim(trim(outfname), nc_out, 'record', nf_unlimited, dimids(3))
    call nfw_def_var(trim(outfname), nc_out, 't', nf_float, 1, dimids(3), id_t)
    call nfw_def_var(trim(outfname), nc_out, 'q', nf_float, 3, dimids, id_q)
    call nfw_def_var(trim(outfname), nc_out, 'psi', nf_float, 3, dimids, id_psi)

    call nfw_enddef(trim(outfname), nc_out)
  end subroutine data_writeopen

  subroutine data_write(t, PSI, Q)
    real(8), intent(in) :: t
    real(8), dimension(:, :), intent(in) :: PSI, Q
    integer :: nr, start(3), length(3)
    real(8) :: tmp(1)
  
    start(1) = 1
    start(2) = 1
    call nfw_inq_dimlen(trim(outfname), nc_out, dimids(3), nr)
    start(3) = nr + 1
    length(1) = N
    length(2) = M
    length(3) = 1
    tmp(1) = t
    call nfw_put_vara_double(trim(outfname), nc_out, id_t, start(3), length(3), tmp)
    call nfw_put_vara_double(trim(outfname), nc_out, id_psi, start, length, PSI)
    call nfw_put_vara_double(trim(outfname), nc_out, id_q, start, length, Q)
    call nfw_sync(trim(outfname), nc_out)

    if (verbose == 1) then
       write(stdout, '(A)', advance = 'no') '.'
    end if
  end subroutine data_write

  subroutine data_writeclose
    call nfw_close(trim(outfname), nc_out)
    if (verbose == 1) then
       write(stdout, *) ''
    end if
  end subroutine data_writeclose

end module data_mod
