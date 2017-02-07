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
    real(8), dimension(1), intent(inout) :: t
    real(8), dimension(M, N), intent(inout) :: PSI
    character(STRLEN) :: prmfname

    real(8), dimension(M, N) :: Q
    real(8) :: tstop

    call parameters_read(prmfname)
  
    tstop = t(1) + dtout

    call laplacian(PSI, dx, dy, Q)
    Q = Q - F * PSI

    do while (t(1) < tstop)
       if (strcmp(scheme, '2ndorder') == 0) then
          call qg_step_2ndorder(t(1), PSI, Q)
       elseif (strcmp(scheme, 'rk4') == 0) then
          call qg_step_rk4(t(1), PSI, Q)
       elseif (strcmp(scheme, 'dp5') == 0) then
          call qg_step_dp5(t(1), PSI, Q)
       else
          write(stdout, *) 'Error: unknown scheme "', trim(scheme), '"'
          stop
       end if
    end do
    call calc_psi(PSI, Q, PSI)
  end subroutine step

end module interface_mod

