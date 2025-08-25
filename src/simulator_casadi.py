###############################################################################
# Casadi implementation
###############################################################################

import numpy as np
import casadi as ca
from typing import Tuple, Dict, Optional, Callable, Type, Any

from src.system import HydraulicSystem


class CasADiSimulator:
    """CasADi-based simulator for improved performance on stiff systems"""
    
    def __init__(
        self,
        system: HydraulicSystem,
        N_steps: int,
        state_init: np.ndarray,
        step_size: float = 1e-3,
        integration_method: str = 'rk4',
        n_substeps: int = 10,
        use_symbolic: bool = True,
    ):
        """
        Initialize CasADi simulator
        
        Args:
            system: Hydraulic system
            N_steps: Number of simulation steps
            state_init: Initial state
            step_size: Time step size
            integration_method: Integration method ('rk4', 'euler', 'collocation')
            n_substeps: Number of substeps for integration
            use_symbolic: Whether to use symbolic computation
        """
        self.system = system
        self.N_steps = N_steps
        self.step_size = step_size
        self.integration_method = integration_method
        self.n_substeps = n_substeps
        self.use_symbolic = use_symbolic
        
        # Initialize state
        self.state_init = np.zeros(system.dim_state)
        self.state_init = state_init.copy()
        
        # CasADi variables
        self.dt_sub = step_size / n_substeps
        
        # Build CasADi functions if using symbolic computation
        if use_symbolic:
            self._build_casadi_functions()
        
        self.reset()
    
    def _build_casadi_functions(self):
        """Build CasADi symbolic functions for system dynamics"""
        # State and action variables
        x = ca.SX.sym('x', self.system.dim_state)
        u = ca.SX.sym('u', self.system.dim_inputs)
        
        # Create symbolic dynamics function using CasADi-compatible method
        if hasattr(self.system, 'compute_dynamics_casadi'):
            dx = self.system.compute_dynamics_casadi(x, u)
        else:
            # Fallback to regular dynamics (may not work with CasADi)
            dx = self.system.compute_dynamics(x, u)
        
        # Create CasADi function for dynamics
        self.dynamics_fn = ca.Function('dynamics', [x, u], [dx])
        
        # Create integration function based on method
        if self.integration_method == 'rk4':
            self._build_rk4_integrator()
        elif self.integration_method == 'euler':
            self._build_euler_integrator()
        elif self.integration_method == 'collocation':
            self._build_collocation_integrator()
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
    
    def _build_rk4_integrator(self):
        """Build RK4 integrator using CasADi"""
        # State and action variables
        x = ca.SX.sym('x', self.system.dim_state)
        u = ca.SX.sym('u', self.system.dim_inputs)
        dt = ca.SX.sym('dt')
        
        # RK4 coefficients
        k1 = self.dynamics_fn(x, u)
        k2 = self.dynamics_fn(x + dt/2 * k1, u)
        k3 = self.dynamics_fn(x + dt/2 * k2, u)
        k4 = self.dynamics_fn(x + dt * k3, u)
        
        # RK4 update
        x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Create integrator function
        self.integrator_fn = ca.Function('rk4_integrator', [x, u, dt], [x_next])
    
    def _build_euler_integrator(self):
        """Build explicit Euler integrator using CasADi"""
        # State and action variables
        x = ca.SX.sym('x', self.system.dim_state)
        u = ca.SX.sym('u', self.system.dim_inputs)
        dt = ca.SX.sym('dt')
        
        # Euler update
        dx = self.dynamics_fn(x, u)
        x_next = x + dt * dx
        
        # Create integrator function
        self.integrator_fn = ca.Function('euler_integrator', [x, u, dt], [x_next])
    
    def _build_collocation_integrator(self):
        """Build collocation integrator using CasADi (more accurate for stiff systems)"""
        # State and action variables
        x = ca.SX.sym('x', self.system.dim_state)
        u = ca.SX.sym('u', self.system.dim_inputs)
        dt = ca.SX.sym('dt')
        
        # Use CasADi's built-in collocation integrator
        # This is more suitable for stiff systems
        dae = {
            'x': x,
            'ode': self.dynamics_fn(x, u),
            'p': u
        }
        
        # Create collocation integrator
        self.integrator_fn = ca.integrator('collocation_integrator', 'collocation', dae, {
            'tf': dt,
            'collocation_scheme': 'legendre',
            'collocation_deg': 3,
            'collocation_num_intervals': 1
        })
    
    def reset(self) -> None:
        """Reset simulator to initial state"""
        self.current_step_idx = 0
        self.state = self.state_init.copy()
        self.action = np.zeros(self.system.dim_inputs)
        self.integration_history = []
    
    def set_action(self, action: np.ndarray) -> None:
        """Set current action"""
        self.action = action.copy()
    
    def system_transition_function(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compute next state using CasADi integration"""
        if self.use_symbolic:
            # Use CasADi symbolic integration
            if self.integration_method == 'collocation':
                # Collocation integrator returns a different structure
                result = self.integrator_fn(x0=state, p=action)
                next_state = result['xf'].full().flatten()
            else:
                # For RK4 and Euler, integrate over substeps
                current_state = state.copy()
                for _ in range(self.n_substeps):
                    current_state = self.integrator_fn(current_state, action, self.dt_sub).full().flatten()
                next_state = current_state
        else:
            # Fallback to numerical integration (slower but more robust)
            next_state = self._numerical_integration(state, action)
        
        # Store integration info
        integration_info = {
            'method': self.integration_method,
            'n_substeps': self.n_substeps,
            'dt_sub': self.dt_sub
        }
        
        return next_state, integration_info
    
    def _numerical_integration(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Fallback numerical integration method"""
        current_state = state.copy()
        
        for _ in range(self.n_substeps):
            # Simple Euler integration
            dx = self.system.compute_dynamics(current_state, action)
            current_state = current_state + self.dt_sub * dx
        
        return current_state
    
    def step(self) -> bool:
        """Perform one simulation step"""
        if self.current_step_idx <= self.N_steps:
            self.state, integration_info = self.system_transition_function(self.state, self.action)
            self.integration_history.append(integration_info)
            self.current_step_idx += 1
            return True
        return False
    
    def get_sim_step_data(self) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Get current simulation data"""
        return (
            int(self.current_step_idx),
            np.copy(self.state),
            self.system.get_observation(self.state),
            np.copy(self.action)
        )
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        if not self.integration_history:
            return {}
        
        return {
            'method': self.integration_method,
            'n_substeps': self.n_substeps,
            'total_substeps': len(self.integration_history) * self.n_substeps,
            'average_dt': self.dt_sub,
            'integration_history': self.integration_history
        }


# Factory function for creating simulators
def create_simulator(
    simulator_type: str = 'casadi',
    **kwargs
) -> CasADiSimulator:
    """
    Factory function to create different types of simulators
    
    Args:
        simulator_type: Type of simulator ('casadi', 'adaptive_casadi', 'scipy')
        **kwargs: Additional arguments for simulator initialization
    
    Returns:
        Simulator instance
    """
    if simulator_type == 'casadi':
        return CasADiSimulator(**kwargs)
    elif simulator_type == 'scipy':
        # Import and return the original scipy simulator
        from src.simulator import Simulator
        return Simulator(**kwargs)
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")