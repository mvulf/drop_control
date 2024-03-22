import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Optional, Callable, Type, Any

from src.system import HydraulicSystem

class Simulator:
    """Let us implement it with ODE-solver scipy"""
    
    def __init__(
        self,
        system: HydraulicSystem,
        N_steps: int,
        state_init: np.ndarray,
        step_size: float = 1e-3, # s. Was 1e-4 s
        atol = 1e-6, # was 1e-7
        rtol = 1e-3, # was 1e-4
        # DELETE:
        # time_scale_first = 1e-6, # 1e-7, # 1e-9 , for the first step
        # time_scale_max = np.inf, # 1e-4, # for the max step in solve_ivp
    ):
        self.system = system
        self.N_steps = N_steps
        self.step_size = step_size
        
        # # Delete
        # if time_scale_first is None:
        #     self.first_step = None # solve_ivp will choose first time step
        # else:
        #     # define first step for the solve_ivp
        #     self.first_step = time_scale_first * step_size
        # self.max_step = time_scale_max*step_size # max_step for the ivp
            
        self.atol = atol
        self.rtol = rtol
        
        self.state_init = np.zeros(system.dim_state)
        self.state_init[:-1] = state_init.copy()
        
        self.reset()
        
    
    def reset(self) -> None:
        """Resets the system to initial state"""
        self.current_step_idx = 0
        self.state = self.state_init.copy()
        self.action = np.zeros(self.system.dim_action)
        self.system.reset(step_size=self.step_size)
    
    
    def set_action(self, action: np.ndarray) -> None:
        """ Save current action to 'self.action'

        Args:
            action (np.ndarray): current action
        """
        self.action = action.copy()
    
    
    def system_transition_function(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """ Ger next state by the action

        Args:
            state (np.ndarray): system state
            action (np.ndarray): system action

        Returns:
            np.ndarray: next state
        """
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """ Get rhs (Dstates) for the ode-solver

            Args:
                t (float): time
                y (np.ndarray): state (in scipy notation)

            Returns:
                np.ndarray: rhs (Dstates) for the ode-solver
            """
            return self.system.compute_dynamics(y, action)
        
        next_state = solve_ivp(
            fun=rhs, 
            t_span=(0, self.step_size), 
            y0=state,
            rtol=self.rtol,
            atol=self.atol,
            # # DELETE
            # first_step=self.first_step,
            # max_step=self.max_step,
        ).y.T[-1]
        
        return next_state
    
    
    def step(self) -> bool:
        """ Do one integration step with step_size

        Returns:
            bool: status of simulation. 'True' - simulation continues, 'False' - simulation stopped
        """
        
        if self.current_step_idx <= self.N_steps:
            self.state = self.system_transition_function(self.state, self.action)
            self.current_step_idx += 1
            return True
        return False
    
    
    def get_sim_step_data(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """ Get current step id, observation and action

        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
                current_step_idx, state, observation with noise, action
        """
        
        return (
            int(self.current_step_idx),
            np.copy(self.state), # state
            self.system.get_observation(self.state), # observation with noise
            np.copy(self.action)
        )