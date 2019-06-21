!! Copyright (C) 2008 Pavel Sakov
!! 
!! This file is part of EnKF-Matlab. EnKF-Matlab is a free software. See 
!! LICENSE for details.

! File:           nfw.f90
!
! Author:         Pavel Sakov, CSIRO Marine Research
!
! Created:        17 March 2005
!
! Last modified:  08/02/2008
!
! Purpose:        Contains wrappers to netcdf functions, mainly for easier
!                 error handling.
! 
! Description: Each subroutine in nfw.f90 is a simple wrapper of a similar
!              function in the NetCDF Fortran interface. The rules of use are
!              pretty simple: for a given NetCDF Fortran function, replace
!              prefix "nf_" by "nfw_" and add the NetCDF file name as the
!              first argument.
!
!              Here is the current list of subroutines in nfw_mod:
!
!              nfw_create(fname, mode, ncid)
!              nfw_open(fname, mode, ncid)
!              nfw_enddef(fname, ncid)
!              nfw_sync(fname, ncid)
!              nfw_close(fname, ncid)
!              nfw_inq_unlimdim(fname, ncid, unlimdimid)
!              nfw_inq_dimid(fname, ncid, name, dimid)
!              nfw_inq_dimlen(fname, ncid, dimid, len)
!              nfw_def_dim(fname, ncid, name, len, dimid)
!              nfw_def_var(fname, ncid, name, type, ndims, dimids, varid)
!              nfw_inq_varid(fname, ncid, name, varid)
!              nfw_inq_varname(fname, ncid, varid, name)
!              nfw_inq_varndims(fname, ncid, varid, ndims)
!              nfw_inq_vardimid(fname, ncid, varid, dimids)
!              nfw_put_var_double(fname, ncid, varid, v)
!              nfw_get_var_int(fname, ncid, varid, v)
!              nfw_get_var_double(fname, ncid, varid, v)
!              nfw_put_vara_int(fname, ncid, varid, start, count, v)
!              nfw_put_vara_double(fname, ncid, varid, start, count, v)
!              nfw_get_vara_int(fname, ncid, varid, start, count, v)
!              nfw_get_vara_double(fname, ncid, varid, start, count, v)
!              nfw_get_att_double(fname, ncid, varid, attname, v)
!              nfw_get_att_int(fname, ncid, varid, attname, v)
!              nfw_put_att_text(fname, ncid, varid, attname, len, text)
!              nfw_put_att_int(fname, ncid, varid, attname, type, len, v)
!              nfw_put_att_real(fname, ncid, varid, attname, type, len, v)
!              nfw_put_att_double(fname, ncid, varid, attname, type, len, v)
!
!              Derived procedures:
!
!              nfw_get_var_double_record(fname, ncid, varid, v)
!
! Modifications:

module nfw_mod
  implicit none
  include 'netcdf.inc'

  character(*), private, parameter :: nfw_version = "0.02"
  integer, private, parameter :: logunit = 6
  character(*), private, parameter :: errprefix = "  nfw: error: "
  private quit1, quit2, quit3

contains

  ! Common exit point -- for the sake of debugging
  subroutine quit
    stop
  end subroutine quit

  subroutine quit1(fname, procname, status)
    character*(*) :: fname
    character*(*) :: procname
    integer :: status
 
    write(logunit, *)
    write(logunit, *) errprefix, '"', trim(fname), '": ', procname, '(): ',&
         nf_strerror(status)
    call flush(logunit)
    call quit
  end subroutine quit1

  subroutine quit2(fname, procname, name, status)
    character*(*) :: fname
    character*(*) :: procname
    character*(*) :: name
    integer :: status

    write(logunit, *)
    write(logunit, *) errprefix, '"', trim(fname), '": ', procname, '(): "',&
         trim(name), '": ', nf_strerror(status)
    call flush(logunit)
    call quit
  end subroutine quit2

  subroutine quit3(fname, procname, name1, name2, status)
    character*(*) :: fname
    character*(*) :: procname
    character*(*) :: name1
    character*(*) :: name2
    integer :: status

    write(logunit, *)
    write(logunit, *) errprefix, '"', trim(fname), '": ', procname, '(): "',&
         trim(name1), '": "', trim(name2), '": ', nf_strerror(status)
    call flush(logunit)
    call quit
  end subroutine quit3

  subroutine nfw_create(fname, mode, ncid)
    character*(*) :: fname
    integer :: mode
    integer :: ncid
    integer :: status

    status = nf_create(trim(fname), mode, ncid)
    if (status /= 0) call quit1(fname, 'nf_create', status)
  end subroutine nfw_create

  subroutine nfw_open(fname, mode, ncid)
    character*(*) :: fname
    integer :: mode
    integer :: ncid
    integer :: status

    status = nf_open(trim(fname), mode, ncid)
    if (status /= 0) call quit1(fname, 'nf_open', status)
  end subroutine nfw_open

  subroutine nfw_enddef(fname, ncid)
    character*(*) :: fname
    integer :: ncid
    integer :: status

    status = nf_enddef(ncid)
    if (status /= 0) call quit1(fname, 'nf_enddef', status)
  end subroutine nfw_enddef

  subroutine nfw_sync(fname, ncid)
    character*(*) :: fname
    integer :: ncid
    integer :: status

    status = nf_sync(ncid)
    if (status /= 0) call quit1(fname, 'nf_sync', status)
  end subroutine nfw_sync

  subroutine nfw_close(fname, ncid)
    character*(*) :: fname
    integer :: ncid
    integer :: status

    status = nf_close(ncid)
    if (status /= 0) call quit1(fname, 'nf_close', status)
  end subroutine nfw_close

  subroutine nfw_inq_unlimdim(fname, ncid, unlimdimid)
    character*(*) :: fname
    integer :: ncid
    integer :: unlimdimid
    integer :: status
    
    status = nf_inq_unlimdim(ncid, unlimdimid)
    if (status /= 0) call quit1(fname, 'nf_inq_unlimdimid', status)
  end subroutine nfw_inq_unlimdim

  subroutine nfw_inq_dimid(fname, ncid, name, dimid)
    character*(*) :: fname
    integer :: ncid
    character*(*) :: name
    integer :: dimid
    integer :: status
    
    status = nf_inq_dimid(ncid, trim(name), dimid)
    if (status /= 0) call quit2(fname, 'nf_inq_dimid', name, status)
  end subroutine nfw_inq_dimid

  subroutine nfw_inq_dimlen(fname, ncid, dimid, len)
    character*(*) :: fname
    integer :: ncid
    integer :: dimid
    integer :: len
    integer :: status

    status = nf_inq_dimlen(ncid, dimid, len)
    if (status /= 0) call quit1(fname, 'nf_inq_dimlen', status)
  end subroutine nfw_inq_dimlen

  subroutine nfw_def_dim(fname, ncid, name, len, dimid)
    character*(*) :: fname
    integer :: ncid
    character*(*) :: name
    integer :: len
    integer :: dimid
    integer :: status

    status = nf_def_dim(ncid, name, len, dimid)
    if (status /= 0) call quit2(fname, 'nf_def_dim', name, status)
  end subroutine nfw_def_dim

  subroutine nfw_def_var(fname, ncid, name, type, ndims, dimids, varid)
    character*(*) :: fname
    integer :: ncid
    character*(*) :: name
    integer :: type
    integer :: ndims
    integer :: dimids(*)
    integer :: varid
    integer :: status

    status = nf_def_var(ncid, name, type, ndims, dimids, varid)
    if (status /= 0) call quit2(fname, 'nf_def_var', name, status)
  end subroutine nfw_def_var

  subroutine nfw_inq_varid(fname, ncid, name, varid)
    character*(*) :: fname
    integer :: ncid
    character*(*) :: name
    integer :: varid
    integer :: status
   
    status = nf_inq_varid(ncid, trim(name), varid)
    if (status /= 0) call quit2(fname, 'nf_inq_varid', name, status)
  end subroutine nfw_inq_varid

  subroutine nfw_inq_varname(fname, ncid, varid, name)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    character*(*) :: name
    integer :: status

    status = nf_inq_varname(ncid, varid, name)
    if (status /= 0) call quit1(fname, 'nf_inq_varname', status)
  end subroutine nfw_inq_varname

  subroutine nfw_inq_varndims(fname, ncid, varid, ndims)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: ndims
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_inq_varndims(ncid, varid, ndims)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_inq_varndims', name, status)
    end if
  end subroutine nfw_inq_varndims

  subroutine nfw_inq_vardimid(fname, ncid, varid, dimids)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: dimids(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_inq_vardimid(ncid, varid, dimids)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_inq_vardimid', name, status)
    end if
  end subroutine nfw_inq_vardimid

  subroutine nfw_put_var_double(fname, ncid, varid, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    real(8) :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_put_var_double(ncid, varid, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_put_var_double', name, status)
    end if
  end subroutine nfw_put_var_double

  subroutine nfw_get_var_int(fname, ncid, varid, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_get_var_int(ncid, varid, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_get_var_int', name, status)
    end if
  end subroutine nfw_get_var_int

  subroutine nfw_get_var_double(fname, ncid, varid, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    real(8) :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_get_var_double(ncid, varid, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_get_var_double', name, status)
    end if
  end subroutine nfw_get_var_double

  subroutine nfw_put_vara_int(fname, ncid, varid, start, count, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: start(*)
    integer :: count(*)
    integer :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_put_vara_int(ncid, varid, start, count, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_put_vara_int', name, status)
    end if
  end subroutine nfw_put_vara_int

  subroutine nfw_put_vara_double(fname, ncid, varid, start, count, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: start(*)
    integer :: count(*)
    real(8) :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_put_vara_double(ncid, varid, start, count, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_put_vara_double', name, status)
    end if
  end subroutine nfw_put_vara_double

  subroutine nfw_get_vara_int(fname, ncid, varid, start, count, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: start(*)
    integer :: count(*)
    integer :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_get_vara_int(ncid, varid, start, count, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_get_vara_int', name, status)
    end if
  end subroutine nfw_get_vara_int

  subroutine nfw_get_vara_double(fname, ncid, varid, start, count, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: start(*)
    integer :: count(*)
    real(8) :: v(*)
    character*(NF_MAX_NAME) :: name
    integer :: status

    status = nf_get_vara_double(ncid, varid, start, count, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_get_vara_double', name, status)
    end if
  end subroutine nfw_get_vara_double

  subroutine nfw_get_att_double(fname, ncid, varid, attname, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    real(8) :: v(*)
    character*(*) :: attname
    character*(NF_MAX_NAME) :: varname
    integer :: status

    status = nf_get_att_double(ncid, varid, attname, v)
    if (status /= 0) then
       if (varid /= nf_global) then
          call nfw_inq_varname(fname, ncid, varid, varname)
       else
          varname = 'NF_GLOBAL'
       end if
       call quit3(fname, 'nf_get_att_double', varname, attname, status)
    end if
  end subroutine nfw_get_att_double

  subroutine nfw_get_att_int(fname, ncid, varid, attname, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    integer :: v(*)
    character*(*) :: attname
    character*(NF_MAX_NAME) :: varname
    integer :: status

    status = nf_get_att_int(ncid, varid, attname, v)
    if (status /= 0) then
       if (varid /= nf_global) then
          call nfw_inq_varname(fname, ncid, varid, varname)
       else
          varname = 'NF_GLOBAL'
       end if
       call quit3(fname, 'nf_get_att_int', varname, attname, status)
    end if
  end subroutine nfw_get_att_int

  subroutine nfw_put_att_text(fname, ncid, varid, attname, len, text)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    character*(*) :: attname
    integer :: len
    character*(*) :: text
    integer :: status
    character*(NF_MAX_NAME) :: varname

    status = nf_put_att_text(ncid, varid, attname, len, trim(text))
    if (status /= 0) then
       if (varid /= nf_global) then
          call nfw_inq_varname(fname, ncid, varid, varname)
       else
          varname = 'NF_GLOBAL'
       end if
       call quit3(fname, 'nf_put_att_text', varname, attname, status)
    end if
  end subroutine nfw_put_att_text

  subroutine nfw_put_att_int(fname, ncid, varid, attname, type, len, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    character*(*) :: attname
    integer :: type
    integer :: len
    integer :: v(*)
    integer :: status
    character*(NF_MAX_NAME) :: varname

    status = nf_put_att_int(ncid, varid, attname, type, len, v)
    if (status /= 0) then
       if (varid /= nf_global) then
          call nfw_inq_varname(fname, ncid, varid, varname)
       else
          varname = 'NF_GLOBAL'
       end if
       call quit3(fname, 'nf_put_att_int', varname, attname, status)
    end if
  end subroutine nfw_put_att_int

  subroutine nfw_put_att_real(fname, ncid, varid, attname, type, len, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    character*(*) :: attname
    integer :: type
    integer :: len
    real :: v(*)
    integer :: status
    character*(NF_MAX_NAME) :: varname

    status = nf_put_att_real(ncid, varid, attname, type, len, v)
    if (status /= 0) then
       if (varid /= nf_global) then
          call nfw_inq_varname(fname, ncid, varid, varname)
       else
          varname = 'NF_GLOBAL'
       end if
       call quit3(fname, 'nf_put_att_real', varname, attname, status)
    end if
  end subroutine nfw_put_att_real

  subroutine nfw_put_att_double(fname, ncid, varid, attname, type, len, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    character*(*) :: attname
    integer :: type
    integer :: len
    real(8) :: v(*)
    integer :: status
    character*(NF_MAX_NAME) :: varname

    status = nf_put_att_double(ncid, varid, attname, type, len, v)
    if (status /= 0) then
       if (varid /= nf_global) then
          call nfw_inq_varname(fname, ncid, varid, varname)
       else
          varname = 'NF_GLOBAL'
       end if
       call quit3(fname, 'nf_put_att_double', varname, attname, status)
    end if
  end subroutine nfw_put_att_double

! Derived subroutines

  ! Reads the first record only
  subroutine nfw_get_var_double_firstrecord(fname, ncid, varid, v)
    character*(*) :: fname
    integer :: ncid
    integer :: varid
    real(8) :: v(*)
    integer :: ndims
    integer :: unlimdimid
    integer :: dimids(NF_MAX_VAR_DIMS)
    integer :: dimlen(NF_MAX_VAR_DIMS)
    integer :: dstart(NF_MAX_VAR_DIMS)
    integer :: i
    character*(NF_MAX_NAME) :: name
    integer :: status

    call nfw_inq_varndims(fname, ncid, varid, ndims)
    call nfw_inq_vardimid(fname, ncid, varid, dimids)
    call nfw_inq_unlimdim(fname, ncid, unlimdimid)
    
    do i = 1, ndims
       call nfw_inq_dimlen(fname, ncid, dimids(i), dimlen(i))
       dstart(i) = 1
    end do

    ! check size of v
    if (dimids(ndims) == unlimdimid) then
       dimlen(ndims) = 1 ! 1 record only
    end if

    status = nf_get_vara_double(ncid, varid, dstart, dimlen, v)
    if (status /= 0) then
       call nfw_inq_varname(fname, ncid, varid, name)
       call quit2(fname, 'nf_get_vara_double', name, status)
    end if
  end subroutine nfw_get_var_double_firstrecord

end module nfw_mod
