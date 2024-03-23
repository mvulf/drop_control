import numpy as np
from typing import Tuple, Dict, Optional, Callable, Type, Any


class HydraulicSystem:
    """System class: hydraulic system. State transition function"""
    
    # SERVICE DATA
    dim_action: int = 1
    dim_observation: int = 2
    # 2 states of the piston and 1 state of 
    # the real throttle position (changes with constant speed)
    dim_state: int = 3
    
    x_th_eps: float = 0.5 # backlash (should be as small as possible)
    dx_th_eps: float = 0.1 # used as limit for positions checking
    
    p_atm: float = 1e5 # Pa
    g: float = 9.81 # m/s^2
    # # Let us exclute evaporation pressure
    # p_sv_hydr: float = 2340 # Pa
    
    def __init__(
        self,
        p_l_gauge: float = 1.5e5,
        p_hydr_init_gauge: float = 0.,
        x_th_limits:Tuple[float, float] = (0., 20.),
        freq_th:float = 500.0,
        m_p: float = 20e-3,
        D_th: float = 200e-6,
        D_hydr: float = 20e-3,
        D_work: float = 20e-3,
        D_exit: float = 0.33e-3,
        l_exit: float = 8.5e-3,
        p_c: float = 10e3,
        eta: float = 0.70,
        zeta_th = 5.0,
        rho_hydr = 1e3,
        rho_work = 1e3,
        beta_v_hydr: float = 0.49e-9,
        beta_v_work: float = 0.49e-9,
        sigma_work: float = 73e-3,
        mu_work: float = 1.0e-3,
        v_j: float = 200.,
        jet_length_std = 1e-1,
        jet_velocity_std = 5e-2,  
    ) -> None:
        """Droplet generator (hydraulic) system

        Args:
            p_l_gauge: Gauge liquid pressure before throttle [Pa]. Defaults to 1.5e5.
            p_hydr_init_gauge: Gauge hydraulic container pressure [Pa]. Defaults to 0..
            x_th_limits: Real throttle position limits [µm]. Defaults to (0, 20).
            freq_th: Frottle frequency [Hz]. Defaults to 500.0.
            m_p: Piston mass [kg]. Defaults to 20e-3.
            D_th: Equivalent throttle diameter [m]. Defaults to 200e-6.
            D_hydr: Hydraulic container diameter [m]. Defaults to 20e-3.
            D_work: Working container diameter [m]. Defaults to 20e-3.
            D_exit: Exit orifice diameter [m]. Defaults to 0.33e-3.
            l_exit: Exit orifice length [m]. Defaults to 8.5e-3.
            p_c: Pressure difference on the piston to start movement [Pa]. Defaults to 10e3.
            eta: Mechanical efficiency. Defaults to 0.70.
            zeta_th: Hydraulic throttle coefficient. Might be find empirically (from real equipment). Now it is taken for the valve, see 'Идельчик И. Е. Справочник по гидравлическим сопротивлениям. М., "Машиностроение", 1975'. Defaults to 5.0.
            rho_hydr: Hydraulic liquid density [kg/m^3]. Defaults to 1e3.
            rho_work: Working liquid density [kg/m^3]. Defaults to 1e3.
            beta_v_hydr: Hydraulic liquid compressibility [Pa^-1]. Defaults to 0.49e-9.
            beta_v_work: Working liquid compressibility [Pa^-1]. Defaults to 0.49e-9.
            sigma_work: Working liquid surface tension [N/m]. Defaults to 73e-3.
            mu_work: Working liquid viscosity[Pa*s]. Defaults to 1.0e-3.
            v_j: Jet speed for the stable operation (found experimentaly) [mm/s]. Defaults to 200..
            jet_length_std: Standard deviation of Relative jet length observation. Defaults to 5e-2.
            jet_velocity_std: Standard deviation of Relative jet velocity observation. Defaults to 1e-2.
        """

        self.p_l: float = p_l_gauge + self.p_atm # Absolute liquid pressure before throttle, Pa
        self.p_hydr_init: float = p_hydr_init_gauge + self.p_atm # Pa
    
        self.x_th_limits = x_th_limits
        # Max throttle speed
        self.v_th_max = freq_th*(x_th_limits[1] - x_th_limits[0]) # µm/s. WAS 0.5*(x_th_limits[1] - x_th_limits[0])/dt_update_action. Incorrect, since depends on dt_update_action. 0.5/1e-3 = 500
        self.m_p = m_p
        self.D_th = D_th
        self.D_hydr = D_hydr
        self.D_work = D_work
        self.D_exit = D_exit
        self.l_exit = l_exit
        
        # Gravity force
        self.F_g = self.m_p*self.g # N
        
        # FUNCTION GET AREA BY DIAMETER
        get_area = lambda D: np.pi*D**2/4 # returns area in [m^2], if input in [m]
        self.A_hydr = get_area(D_hydr) # m^2
        self.A_work = get_area(D_work) # m^2
        self.A_max = max(self.A_hydr, self.A_work) # m^2
        self.D_work_exit_2_ratio = D_work**2/D_exit**2
        # Friction params
        self.p_c = p_c
        self.eta = eta
        self.F_c = self.p_c * self.A_max # Coulomb friction force, N
        # Hydraulic coeficients
        self.zeta_th = zeta_th
        self.C_D = 0.827 - 0.0085*l_exit/D_exit # see https://doi.org/10.1201/9781420040470
        self.zeta_exit = 1/self.C_D**2
        
        # Liquid params
        self.rho_hydr = rho_hydr
        self.rho_work = rho_work
        
        self.beta_v_hydr = beta_v_hydr
        self.beta_v_work = beta_v_work
        
        self.sigma_work = sigma_work
        self.mu_work = mu_work
        
        self.v_j = v_j # jet speed for the stable operation (found experimentaly) [mm/s]
        
        # capillar pressure difference to othercome for drop exiting
        self.p_capillar_max = 4*sigma_work/D_exit
        
        # Dimensionless jet numbers
        self.We_j = rho_work*v_j**2*D_exit/(1e6*sigma_work)
        self.Re_j = rho_work*v_j*D_exit/(1e3*mu_work)
        self.Oh_j = np.sqrt(self.We_j)/self.Re_j
        
        # Critical jet length
        # l_crit = 19.5*np.sqrt(We_j)*(1 + 3*Oh_j)**0.85 * D_exit # see https://doi.org/10.1201/9781420040470
        self.l_crit = 13.4e3*(np.sqrt(self.We_j)\
            + 3*self.We_j/self.Re_j) * D_exit # see https://doi.org/10.1007/s00348-003-0629-6
        # Estimated Droplet diameter
        self.D_drop = 1e3*(1.5*np.pi*np.sqrt(2 + 3*self.Oh_j))**(1/3) * D_exit
        
        # Coefs of pressure losses
        self.ploss_coef_h = (zeta_th*rho_hydr*D_hydr**4)/(32*D_th**2)
        self.ploss_coef_t = (self.zeta_exit*rho_work*D_work**4)/\
            (2e12*D_exit**4)
        
        # NOISES
        self.jet_length_std = jet_length_std
        self.jet_velocity_std = jet_velocity_std
        
        self.reset()
    
    
    def reset(self, step_size=None) -> None:
        """Reset system to initial state."""
        # # Policy update time. NOW GET FROM SIMULATOR
        # dt_update_action = 1e-3 # s. Was 1e-4 s
        self.step_size = step_size # Setup step size from simulator
        # p_h|_{x_{th}>0}
        self.p_hydr_last = self.p_hydr_init
        # x_p|_{x_{th}>0}
        self.x_p_last = None # define later, if None
        # Initial piston position
        self.x_p_init = None # define later, if None
    
        
    def get_pressure_hydraulic(self, x_p: float, v_p: float, x_th: float) -> float:
        """ Get pressure in the hydraulic container

        Args:
            x_p (float): piston position [µm]
            v_p (float): piston velocity [µm/s]
            x_th (float): throttle position [µm]

        Returns:
            float: pressure in the hydraulic container [Pa]
        """
        
        # Define last piston position first time as init piston position
        if self.x_p_last is None:
            self.x_p_last = x_p
        
        # Calculate
        if x_th > 0:
            pressure_hydraulic = self.p_l
            # dynamic pressure loss happends only when there is a flow rate
            if v_p != 0: 
                # self.x_th_eps refers to somekind of backslash
                pressure_hydraulic -= v_p*(abs(v_p)/\
                    max(self.x_th_eps, x_th)**2)*self.ploss_coef_h
        else:
            # if x_p < 0:
            #     print('WARNING: piston position might be positive!')
            # assert x_p > 0, 'piston position might be positive'
            pressure_hydraulic = self.p_hydr_last +\
                (self.x_p_last/x_p - 1)/self.beta_v_hydr
        # LET US NOT CHECK NEGATIVE PRESSURE
        # # Pressure cannot be smaller than saturated vapor pressure
        # pressure_hydraulic = max(self.p_sv_hydr, pressure_hydraulic)
        # Keep for future logging of the low hydraulic pressure.
        # if pressure_hydraulic == self.p_sv_hydr:
        #     print('WARNING: low hydraulic pressure')
        #     print(f'dx_p = {x_p - self.x_p_init:.3e}')
        #     print(f'v_p = {v_p:.3e}')
        
        # Save piston position and hydraulic pressure if throttle is opened
        if x_th > 0:
            self.x_p_last = x_p
            self.p_hydr_last = pressure_hydraulic
        return pressure_hydraulic
    
    
    def get_pressure_test(self, x_p: float, v_p: float) -> float:
        """ Get pressure in the test container

        Args:
            x_p (float): piston position [µm]
            v_p (float): piston velocity [µm/s]

        Returns:
            float: pressure in the test container [Pa]
        """
        
        # if x_p < 0:
        #         print('WARNING: piston position might be positive!')
        # assert x_p > 0, 'piston position might be positive'
        # Define init piston position, if it is first time
        if self.x_p_init is None:
            self.x_p_init = x_p
        
        # Position difference
        dx_p = x_p - self.x_p_init
        
        pressure_capillar = min(
            self.p_capillar_max, 
            abs(dx_p/x_p)/self.beta_v_work
        )
        
        pressure_test = self.p_atm + np.sign(dx_p) * pressure_capillar
            
        # dynamic pressure loss happends only when there is a flow rate
        if v_p != 0:
            pressure_test += v_p*abs(v_p) * self.ploss_coef_t
        
        return pressure_test
        
    
    def get_force_hydraulic(self, x_p: float, v_p: float, x_th: float) -> float:
        """ Get hydraulic force acting on the piston

        Args:
            x_p (float): piston position [µm]
            v_p (float): piston velocity [µm/s]
            x_th (float): throttle position [µm]

        Returns:
            float: hydraulic force [N]
        """
        
        p_h = self.get_pressure_hydraulic(x_p, v_p, x_th)
        p_t = self.get_pressure_test(x_p, v_p)
        
        return self.A_hydr*p_h - self.A_work*p_t
        
    
    def get_force_friction(self, v_p: float, F_h: float) -> float:
        """ Get friction force acting on the piston

        Args:
            v_p (float): piston velocity [µm/s]
            F_h (float): Hydraulic force [N]

        Returns:
            float: friction force [N]
        """
        
        if v_p > 0:
            return - np.sign(v_p) * max(self.F_c, (1-self.eta)*F_h)
        # If piston does not move
        return -np.sign(self.F_g + F_h) * self.F_c
    
    
    def get_acceleration(self, x_p: float, v_p: float, x_th: float) -> float:
        """ Get piston acceleration

        Args:
            x_p (float): piston position [µm]
            v_p (float): piston velocity [µm/s]
            x_th (float): throttle position [µm]

        Returns:
            float: piston acceleration [m/s^2]
        """
        F_h = self.get_force_hydraulic(x_p, v_p, x_th)
        F_fr = self.get_force_friction(v_p, F_h)
        
        if (abs(v_p) > 0) or (abs(F_h + self.F_g) > abs(F_fr)):
            return (self.g + 1/self.m_p * (F_h + F_fr))*1e6
        return 0 # if piston does not move and acting force lower than friction
    
    
    def compute_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Calculate right-hand-side (rhs) for ODE solver (or Euler integrator)

        Args:
            state (np.ndarray): current state
            action (np.ndarray): current action

        Returns:
            np.ndarray: rhs for the ODE solver
        """
        
        # Get states
        x_p = state[0]
        v_p = state[1]
        # real throttle position
        # If real throttle position out of bounds - end throttle movement and set in bounds
        x_th = max(self.x_th_limits[0], state[2])
        x_th = min(self.x_th_limits[1], x_th)
        
        # Get and modify action
        x_th_act = max(self.x_th_limits[0], action[0]) # consider negative action as closed throttle
        # For expanded actions
        x_th_act = min(self.x_th_limits[1], x_th_act)
        assert (x_th_act >= self.x_th_limits[0]) and (x_th_act <= self.x_th_limits[1]), 'action out of the bounds'
        
        # Get Dstates
        Dstate = np.zeros(self.dim_state)
        # \dot{x_th}
        # if real throttle position is differ from the set one, change it
        if abs(x_th_act - x_th) > self.dx_th_eps:
            Dstate[2] = np.sign(x_th_act - x_th) * self.v_th_max
        else:
            x_th = x_th_act # set throttle position exact as what we want to act
            state[2] = x_th # NOTE: x_th in ODE-SOLVER (state[2]) were differ from setted here
            Dstate[2] = 0
        
        # \dot{x_p}
        Dstate[0] = v_p
        # \dot{v_p}
        Dstate[1] = self.get_acceleration(x_p, v_p, x_th)
        
        return Dstate
    
    
    def get_jet_velocity(self, v_p: float) -> float:
        """Get exit jet velocity

        Args:
            v_p (float): piston velocity [µm/s]

        Returns:
            float: exit jet velocity [mm/s]
        """
        return 1e-3 * v_p * self.D_work_exit_2_ratio
    
    
    def get_jet_length(self, x_p: float) -> float:
        """Get objective (jet length (which necessary to compare with l_crit))

        Args:
            x_p (float): piston position [µm]

        Returns:
            float: objective [mm]
        """
        return 1e-3 * (x_p - self.x_p_init) * self.D_work_exit_2_ratio
    
    
    def get_clean_observation(self, state: np.ndarray) -> np.ndarray:
        """Get clean observations (relative jet length and relative jet velocity), without sensors noise

        Args:
            state (np.ndarray): system state

        Returns:
            np.ndarray: observation (rel. jet length, rel. velocity [1/control time step])
        """
        x_p = state[0]
        v_p = state[1]
        
        # Define init piston position, if it is first time
        if self.x_p_init is None:
            self.x_p_init = x_p
        
        observation = np.zeros(self.dim_observation)
        # relative jet length
        observation[0] = self.get_jet_length(x_p) / self.l_crit
        # relative jet velocity
        observation[1] = self.get_jet_velocity(v_p) / self.l_crit *\
            self.step_size # get relative velocity in [1/control time step]
        
        return observation
    
    
    def get_observation(self, state: np.ndarray) -> np.ndarray:
        """Get noised observations (relative jet length and relative jet velocity), with std set in system initialization

        Args:
            state (np.ndarray): system state

        Returns:
            np.ndarray: noised observation (rel. jet length, rel. velocity [1/control time step])
        """
        observation = self.get_clean_observation(state)
        
        # relative jet length with noise
        observation[0] += np.random.normal(
            scale=self.jet_length_std
        )
        # relative jet velocity with noise
        observation[1] += np.random.normal(
            scale=self.jet_velocity_std
        )
        
        return observation