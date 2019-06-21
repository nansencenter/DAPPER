!! Copyright (C) 2016 Patrick N. Raanes
!! 
!! Composed from qg.f90

module interface_mod
  use qgstep_mod
  use parameters_mod, only: dt, M, N, scheme, dtout
  use qgflux_mod
  use utils_mod, only: STRLEN

contains

  subroutine step(t, PSI, prmfname)
    real(8), intent(inout) :: t
    real(8), dimension(N, M), intent(inout) :: PSI
    character(STRLEN) :: prmfname

    ! 1st axis (len N) is the x-axis
    ! 2nd axis (len M) is the y-axis (curl = sin(...y))
    real(8), dimension(N, M) :: Q
    real(8) :: tstop

    call parameters_read(prmfname)
  
    tstop = t + dtout

    call laplacian(PSI, dx, dy, Q)
    Q = Q - F * PSI

    do while (t < tstop)
       if (strcmp(scheme, '2ndorder') == 0) then
          call qg_step_2ndorder(t, PSI, Q)
       elseif (strcmp(scheme, 'rk4') == 0) then
          call qg_step_rk4(t, PSI, Q)
       elseif (strcmp(scheme, 'dp5') == 0) then
          call qg_step_dp5(t, PSI, Q)
       else
          write(stdout, *) 'Error: unknown scheme "', trim(scheme), '"'
          stop
       end if
    end do
    call calc_psi(PSI, Q, PSI)
  end subroutine step

end module interface_mod

