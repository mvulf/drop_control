from typing import Tuple, Dict, Optional, Callable, Type, Any

from regelum.system import System

import numpy as np
from regelum.utils import rg
import casadi as ca


class HydraulicSystem(System):
    
    _name = 'HydraulicSystem'
    _system_type = 'diff_eqn'
    _dim_state = 5
    _dim_inputs = 1
    # _dim_observation = 4
    _dim_observation = 2
    _state_naming = [
        "piston position [µm]", 
        "piston velocity [µm/s]", 
        "throttle position [µm]",
        "hydraulic pressure [Pa]",
        "working pressure [Pa]",
    ]
    _observation_naming = [
        "jet length [mm]", 
        "jet velocity [mm/s]",
        # "hydraulic pressure [Pa]",
        # "working pressure [Pa]",
    ]
    _inputs_naming = ["throttle action [µm]"]
    _action_bounds = [[-20.0, 20.0]]
    
    def __init__(
        self,
        init_state,
        *args,
        system_parameters_init=None,
        **kwargs,
    ):
        """Droplet generator (hydraulic) system

        Args:
            system_parameters_init: parameters of the system:
                p_l_gauge: Gauge liquid pressure before throttle [Pa]. 
                    Defaults to 1.5e5.
                x_th_limits: Real throttle position limits [µm]. 
                    Defaults to (0, 20).
                freq_th: Frottle frequency [Hz]. Defaults to 500.0.
                m_p: Piston mass [kg]. Defaults to 20e-3.
                D_th: Equivalent throttle diameter [m]. Defaults to 200e-6.
                D_hydr: Hydraulic container diameter [m]. Defaults to 20e-3.
                D_work: Working container diameter [m]. Defaults to 20e-3.
                h_work_init: Initial height of working liquid [µm]Defaults to 1e3.
                D_exit: Exit orifice diameter [m]. Defaults to 0.33e-3.
                l_exit: Exit orifice length [m]. Defaults to 8.5e-3.
                p_coulomb: Pressure difference on the piston to start movement [Pa].
                    Defaults to 10e3.
                eta: Mechanical efficiency. Defaults to 0.70.
                zeta_th: Hydraulic throttle coefficient. 
                    Might be find empirically (from real equipment). 
                    Now it is taken for the valve, see 
                    'Идельчик И. Е. Справочник по гидравлическим 
                    сопротивлениям. М., "Машиностроение", 1975'. 
                    Defaults to 5.0.
                rho_hydr: Hydraulic liquid density [kg/m^3]. Defaults to 1e3.
                rho_work: Working liquid density [kg/m^3]. Defaults to 1e3.
                beta_v_hydr: Hydraulic liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                beta_v_work: Working liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                sigma_work: Working liquid surface tension [N/m]. 
                    Defaults to 73e-3.
                mu_work: Working liquid viscosity[Pa*s]. Defaults to 1.0e-3.
                v_j: Jet speed for the stable operation (found experimentaly)
                    [mm/s]. Defaults to 200.
                jet_length_std: Standard deviation of Relative jet length 
                    observation. Defaults to 10e-2
                jet_velocity_std: Standard deviation of Relative jet velocity 
                    observation. Defaults to 2e-2
                pressure_std: Standard deviation of pressure sensors (manometers). 
                    Defaults to 0. We do not use it now.
                p_atm: Atmosphere (ambient) pressure, Pa. 
                    Defaults to 1e5.
                g: Gravity constant, m/s^2. Defaults to 9.81. 
        """
        
        self.init_state = init_state
        
        if system_parameters_init is None:
            system_parameters_init = {
                "p_l_gauge": 1.5e5,
                "x_th_limits": (0., 20.),
                "freq_th": 500.0,
                "m_p": 20e-3,
                "D_th": 200e-6, # WAS 200e-6, then 5e-3
                "D_hydr": 20e-3,
                "D_work": 20e-3,
                "h_work_init": 1e3,
                "D_exit": 0.33e-3,
                "l_exit": 8.5e-3,
                "p_coulomb": 10e3, # WAS 10e3, then 1e3
                "eta": 0.70,
                "zeta_th": 5.0, # WAS 5.0, then 0.1
                "rho_hydr": 1e3,
                "rho_work": 1e3,
                "beta_v_hydr": 0.49e-9,
                "beta_v_work": 0.49e-9,
                "sigma_work": 73e-3,
                "mu_work": 1.0e-3,
                "v_j": 200.,
                "jet_length_std": 0, # Was 10e-2
                "jet_velocity_std": 0, # Was 2e-2
                "pressure_std": 0.,
                "p_atm": 1e5, # Atmosphere (ambient) pressure, Pa
                "g": 9.81, # gravity constant, m/s^2
            }
        
        super().__init__(
            *args,
            system_parameters_init=system_parameters_init,
            **kwargs
        )
        
        # Get absolute pressure
        p_atm, p_l_gauge = (
            self._parameters["p_atm"],
            self._parameters["p_l_gauge"],
        )
        self.update_system_parameters(
            {
                "p_l": p_atm + p_l_gauge,
            }
        )
        
        # Piston mass, gravity parameters; Gravity force
        m_p, g = (
            self._parameters["m_p"],
            self._parameters["g"]
        )
        self.update_system_parameters(
            {
                "F_g": m_p*g, # Gravity force, N
            }
        )
        
        # Geometry parameters        
        D_th, D_hydr, D_work, D_exit, l_exit = (
            self._parameters["D_th"],
            self._parameters["D_hydr"],
            self._parameters["D_work"],
            self._parameters["D_exit"],
            self._parameters["l_exit"],
        )
        # FUNCTION GET AREA BY DIAMETER
        # returns area in [m^2], if input in [m]
        get_area = lambda D: np.pi*D**2/4
        self.update_system_parameters(
            {
                "A_hydr": get_area(D_hydr), # m^2
                "A_work": get_area(D_work), # m^2
                "D_work_exit_2_ratio": (D_work/D_exit)**2,
            }
        )
        
        # Friction force
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        A_max = max(A_hydr, A_work) # m^2
        # Coulomb friction force, N
        self.update_system_parameters(
            {
                "F_coulomb": self._parameters["p_coulomb"] * A_max
            }
        )
        
        # Discharge coefficients
        zeta_th = self._parameters["zeta_th"]
        C_D_th = 1/zeta_th**(1/2)
        C_D_exit = 0.827 - 0.0085*l_exit/D_exit
        
        # Liquid params
        rho_hydr, rho_work, sigma_work, mu_work = (
            self._parameters["rho_hydr"],
            self._parameters["rho_work"],
            self._parameters["sigma_work"],
            self._parameters["mu_work"],
        )
        
        # Volume flow rate coefficients
        B_th = 4*C_D_th*D_th/D_hydr**2 * (2/rho_hydr)**(1/2) # WAS multiplyied by 1e-6
        B_exit = 1e6*C_D_exit*(D_exit/D_work)**2 * (2/rho_work)**(1/2) # WAS NOT multiplyied by 1e6
        
        self.update_system_parameters(
            {
                "B_th": B_th,
                "B_exit": B_exit,
            }
        )
        
        # Bulk modulus
        beta_v_hydr = self._parameters["beta_v_hydr"]
        beta_v_work = self._parameters["beta_v_work"]
        self.update_system_parameters(
            {
                "K_hydr": 1/beta_v_hydr,
                "K_work": 1/beta_v_work,
            }
        )
        
        # Dimensionless jet numbers
        v_j = self._parameters["v_j"]
        We_j = rho_work*v_j**2*D_exit/(1e6*sigma_work)
        Re_j = rho_work*v_j*D_exit/(1e3*mu_work)
        Oh_j = np.sqrt(We_j)/Re_j
        
        # JET LENGTH AND DROP DIAMETER
        # Critical jet length
        # see https://doi.org/10.1007/s00348-003-0629-6
        l_crit = 13.4e3*(np.sqrt(We_j)\
            + 3*We_j/Re_j) * D_exit
        # Estimated Droplet diameter
        D_drop = 1e3*(1.5*np.pi*np.sqrt(2 + 3*Oh_j))**(1/3) * D_exit
        self.update_system_parameters(
            {
                "l_crit": l_crit,
                "D_drop": D_drop,
            }
        )
        
     
    def get_observation(self, state):
        """Get observation with normal noise, based on state only
        """
        return self._get_observation(
            time=None,
            state=state,
            inputs=None,
        )
        
        
    def _get_observation(self, time, state, inputs):
            """ Get observation with normal noise
            """
            observation = self.get_clean_observation(state)
            
            # relative jet length with noise
            observation[0] += self._parameters["l_crit"] * np.random.normal(
                scale=self._parameters["jet_length_std"]
            )
            # relative jet velocity with noise
            observation[1] += (
                1e3 * self._parameters["l_crit"] # 1e3 is instead of "1/sampling_time"
                * np.random.normal(
                    scale=self._parameters["jet_velocity_std"]
                )
            )
            
            # # Pressures with noise
            # observation[2] += np.random.normal(
            #     scale=self._parameters["pressure_std"]
            # )
            # observation[3] += np.random.normal(
            #     scale=self._parameters["pressure_std"]
            # )

            return observation
        
    
    def get_clean_observation(self, state):
        """Get clean observations 
        (relative jet length and relative jet velocity), without sensors noise
        Assumption: volume flow rates for working container and jet are the same

        Args:
            state: system state

        Returns:
            observation (jet length, jet velocity)
        """
        
        observation = np.zeros(
            self.dim_observation,
        )
        
        # Current and init state parameters
        # Get current state parameters
        x_p, v_p, _, p_hydr, p_work = [
            state[i] for i in range(self.dim_state)
        ]
        x_p_init = self.init_state[0]
        
        D_work_exit_2_ratio = self._parameters["D_work_exit_2_ratio"]
        
        # ABSOLUTE Jet length. NOTE: no "/ self.l_crit"
        observation[0] = ( # [mm]
            1e-3 * (x_p - x_p_init) * D_work_exit_2_ratio
        )
        
        # ABSOLUTE Jet velocity. NOTE: no "self.l_crit *\self.step_size"
        observation[1] = ( # [mm/s]
            1e-3 * v_p * D_work_exit_2_ratio
        )
        
        # # Pressures
        # observation[2] = p_hydr
        # observation[3] = p_work
        
        return observation


    def compute_dynamics_casadi(self, state, action):
        """CasADi-compatible version of compute_dynamics
        
        Args:
            state: CasADi symbolic state vector
            action: CasADi symbolic action vector

        Returns:
            CasADi symbolic dynamics vector
        """
        return self._compute_state_dynamics_casadi(
            time=None,
            state=state,
            inputs=action,
        )

    def _compute_state_dynamics_casadi(self, time, state, inputs):
        """CasADi-compatible state dynamics computation"""
        
        # Get current state parameters
        x_p, v_p, x_th, p_hydr, p_work = [
            state[i] for i in range(self.dim_state)
        ]
        
        # CLIP THROTTLE POSITION
        x_th_limits = self._parameters["x_th_limits"]
        x_th = ca.fmax(x_th_limits[0], ca.fmin(x_th_limits[1], x_th))
        x_th_act = inputs[0]
        
        # HYDRAULIC FORCE
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        F_hydr = A_hydr*p_hydr - A_work*p_work
        
        # Required dynamic parameters
        F_coulomb, eta, F_g, g, m_p = (
            self._parameters["F_coulomb"],
            self._parameters["eta"],
            self._parameters["F_g"],
            self._parameters["g"],
            self._parameters["m_p"],
        )
        
        # FRICTION FORCE - use CasADi conditional logic
        F_fr_hydr = (1-eta)*F_hydr
        F_fr_dynamic = -ca.sign(v_p) * ca.fmax(F_coulomb, F_fr_hydr)
        F_fr_static = -ca.sign(F_g + F_hydr) * F_coulomb
        F_friction = ca.if_else(v_p != 0, F_fr_dynamic, F_fr_static)
        
        # ACCELERATION - use CasADi conditional logic
        cond_velocity = ca.if_else(v_p != 0, 1, 0)
        cond_fr_overcome = ca.if_else(
            ca.fabs(F_hydr + F_g) > ca.fabs(F_friction), 1, 0
        )
        
        acceleration = ca.if_else(
            (cond_velocity + cond_fr_overcome) > 0,
            1e6*(g + 1/m_p * (F_hydr + F_friction)),
            0
        )
        
        # Build dynamics vector using CasADi concatenation
        Dstate = ca.vertcat(
            v_p,  # \dot{x_p}
            acceleration,  # \dot{v_p}
            self._parameters["freq_th"] * (x_th_act - x_th),  # \dot{x_th}
            self._compute_pressure_dynamics_hydraulic_casadi(x_p, v_p, x_th, p_hydr),  # \dot{p_hydr}
            self._compute_pressure_dynamics_working_casadi(x_p, v_p, p_work)  # \dot{p_work}
        )
        
        return Dstate

    def _compute_pressure_dynamics_hydraulic_casadi(self, x_p, v_p, x_th, p_hydr):
        """CasADi-compatible hydraulic pressure dynamics"""
        p_l, B_th, K_hydr = (
            self._parameters["p_l"],
            self._parameters["B_th"],
            self._parameters["K_hydr"],
        )
        
        return K_hydr * (
            ca.sign(p_l - p_hydr) * B_th * x_th * ca.sqrt(ca.fabs(p_l - p_hydr))
            - v_p
        ) / x_p

    def _compute_pressure_dynamics_working_casadi(self, x_p, v_p, p_work):
        """CasADi-compatible working pressure dynamics"""
        x_p_init = self.init_state[0]
        p_atm, B_exit, K_work, h_work_init = (
            self._parameters["p_atm"],
            self._parameters["B_exit"],
            self._parameters["K_work"],
            self._parameters["h_work_init"],
        )
        
        return K_work * (
            v_p 
            - ca.sign(p_work - p_atm) * B_exit * ca.sqrt(ca.fabs(p_work - p_atm))
        ) / (h_work_init - x_p + x_p_init)
    
    
    ###########################################################################
    # Previous version of the system, for compatibility with regelum and any simulators
    ###########################################################################
    
    def compute_closed_loop_rhs(self, time, state):
        return self._compute_state_dynamics(time, state, self.inputs)    
        
    def compute_dynamics(self, state, action):
        """Calculate right-hand-side (rhs) for ODE solver (or Euler integrator).
        NOTE: this function is created for compatibility with current simulator only.
        No need to use it in regelum.
        NOTE: SEE compute_dynamics_casadi for CasADi-compatible version.

        Args:
            state (np.ndarray): current state
            action (np.ndarray): current action

        Returns:
            np.ndarray: rhs for the ODE solver
        """
        return self._compute_state_dynamics(
            time=None,
            state=state,
            inputs=action,
        )    
        
    def _compute_state_dynamics(self, time, state, inputs):
        """Compute state dynamics for the system
        NOTE: see _compute_state_dynamics_casadi for CasADi-compatible version.

        Args:
            time: time
            state: state
            inputs: inputs

        Returns:
            Dstate: state dynamics
        """
        
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )
        
        # Get current state parameters
        x_p, v_p, x_th, p_hydr, p_work = [
            state[i] for i in range(self.dim_state)
        ]
        
        # CLIP THROTTLE POSITION
        x_th_limits = self._parameters["x_th_limits"]
        # # If real throttle position out of bounds - 
        # # end throttle movement and set in bounds
        x_th = rg.if_else(x_th > x_th_limits[0], x_th, x_th_limits[0])
        x_th = rg.if_else(x_th < x_th_limits[1], x_th, x_th_limits[1])
        x_th_act = inputs[0]
        
        # HYDRAULIC FORCE
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        F_hydr = A_hydr*p_hydr - A_work*p_work
        
        # Required dynamic parameters
        F_coulomb, eta, F_g, g, m_p = (
            self._parameters["F_coulomb"],
            self._parameters["eta"],
            self._parameters["F_g"],
            self._parameters["g"],
            self._parameters["m_p"],
        )
        
        # # FRICTION FORCE
        F_fr_hydr = (1-eta)*F_hydr
        # If piston moves
        F_fr_dynamic = -rg.sign(v_p) * rg.if_else(
            F_coulomb > F_fr_hydr,
            F_coulomb,
            F_fr_hydr
        )
        # If piston does not move
        F_fr_static = -rg.sign(F_g + F_hydr) * F_coulomb
        
        F_friction = rg.if_else(v_p != 0, F_fr_dynamic, F_fr_static)
            
        # # ACCELERATION
        cond_velocity = rg.if_else(
            v_p != 0,
            1,
            0
        )
        cond_fr_overcome = rg.if_else(
            rg.abs(F_hydr + F_g) > rg.abs(F_friction),
            1,
            0
        )
        # return 0, if piston does not move and acting force lower than friction
        acceleration = rg.if_else(
            (cond_velocity + cond_fr_overcome) > 0, # OR
            1e6*(g + 1/m_p * (F_hydr + F_friction)),
            0
        )
        
        # RHS
        # \dot{x_p}
        Dstate[0] = state[1]
        
        # \dot{v_p}
        Dstate[1] = acceleration
        
        # \dot{x_th}
        freq_th = self._parameters["freq_th"]
        Dstate[2] = freq_th * (x_th_act - x_th)
        
        # \dot{p_hydr}
        p_l, B_th, K_hydr = (
            self._parameters["p_l"],
            self._parameters["B_th"],
            self._parameters["K_hydr"],
        )
        Dstate[3] = (
            K_hydr*(
                rg.sign(p_l - p_hydr)*B_th*x_th*rg.abs(p_l - p_hydr)**(1/2)
                - v_p
            ) / x_p
        )
        
        # \dot{p_work}
        x_p_init = self.init_state[0]
        p_atm, B_exit, K_work, h_work_init = (
            self._parameters["p_atm"],
            self._parameters["B_exit"],
            self._parameters["K_work"],
            self._parameters["h_work_init"],
        )
        Dstate[4] = (
            K_work*(
                v_p 
                - rg.sign(p_work - p_atm)*B_exit*rg.abs(p_work - p_atm)**(1/2)
            )/(h_work_init - x_p + x_p_init)
        )
        
        return Dstate