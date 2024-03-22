import pandas as pd
import numpy as np

from src.simulator import Simulator
from src.policy import PDController

from datetime import datetime

import matplotlib.pyplot as plt

plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "mathptmx",
    'font.size': 14,
})

class SimulationScenario:
    
    def __init__(
        self,
        simulator: Simulator,
        controller: PDController,
        data_path:str,
        discount_factor: float = 1.0,
        dpi: int = 400,
    ):
        self.simulator = simulator
        self.controller = controller
        
        self.data_path = data_path
        
        self.discount_factor = discount_factor
        
        self.dpi = dpi
        
        self.clean_data()
        
        
    def clean_data(self):
        self.observations = []
        self.actions = []
        self.states = []
        
        self.total_objective = 0
    
    
    def compute_running_objective(
        self, observation: np.array, action: np.array
    ) -> float:
        """Computes running objective

        Args:
            observation (np.array): current observation
            action (np.array): current action

        Returns:
            float: running objective value
        """
        
        length_diff = (1 - observation[0])

        
        # # WAS
        # # If length smaller than critical, penalty for the negative velocity
        # # If length larger than critical, penalty for the positive velocity
        # return length_diff ** 2 -\
        #     np.sign(length_diff) * observation[1] * abs(observation[1]) * 1 # get relative velocity in [1/ms] # Previous miltiply was 1e2
        
        return length_diff ** 2
    
    
    def compute_total_objective(
        self,
        observations, 
        actions, 
    ):
        total_objective = 0
        
        for step_idx, (observation, action) in enumerate(
            zip(observations, actions)
        ):
            discounted_running_objective = self.discount_factor ** (
                step_idx
            ) * self.compute_running_objective(observation, action)
            # for learning curve plotting
            total_objective += discounted_running_objective
        
        return total_objective
    
    
    def get_real_actions(self, actions:list):

        A = np.zeros((len(actions)*2, 2))

        for i in range(len(actions)-1):
            A[i*2+1:i*2+3, 1] = actions[i+1]
            if i > 0:
                A[i*2:i*2+2, 0] = i

        A[-2:,0] = len(actions) - 1
        
        return A
    
    
    def run(self):
        while self.simulator.step():
            (
                step_idx,
                state,
                observation,
                action,
            ) = self.simulator.get_sim_step_data()

            # PD CONTROLLER
            new_action = self.controller.get_action(observation)  
      
            self.simulator.set_action(new_action)
            self.observations.append(observation)
            self.actions.append(action)
            self.states.append(state)

        total_obj = self.compute_total_objective(
            observations=self.observations,
            actions=self.actions,
        )
        
        print(f'Total objective: {total_obj:.5f}')
        
    
    def plot_data(self, log_data=True):
        """Plot results"""
        # TODO: MERGE WITH PLOT DATA FROM MONTECARLO
        if log_data:
            # datetime object containing current date and time
            now = datetime.now()
            # %Y-%m-%d_%H%M%S
            dt_string = now.strftime("%Y-%m-%d_%H%M%S")
            print("log date and time =", dt_string)
        
        observations = pd.DataFrame(
            data=np.array(self.observations)
        )
        
        # actions = pd.DataFrame(
        #     data=self.actions #self.last_actions.loc[0].values
        # )
        
        ax_jet_length, ax_jet_velocity = observations.plot(
            xlabel="step number $(t)$",
            title="Observations",
            legend=False,
            subplots=True,
            grid=True,
            marker='.'
        )
        
        ax_jet_length.set_ylabel(r"$x^\mathrm{rel}_\mathrm{jet}$")
        ax_jet_velocity.set_ylabel(r"$v^\mathrm{rel}_\mathrm{jet}$")
        
        plt.tight_layout()
        if log_data:
            plt.savefig(
                self.data_path+f"{dt_string}_observations.pdf", 
                dpi=self.dpi,
            )

        actions_arr = self.get_real_actions(self.actions)

        pulse_actions = pd.DataFrame(
            data=actions_arr,
            columns=['step', 'action']
        )

        ax_actions = pulse_actions.plot.line(
            x='step',
            y='action',
            xlabel="step number $(t)$",
            title="Throttle position (action)",
            label='action',
            grid=True,
        )
        ax_actions.set_ylabel(r"$x^\mathrm{act}_\mathrm{th}$ [µm]")
        
        states_arr = np.array(self.states)
        ax_actions.plot(states_arr[:, 2], marker=".", label='$x_\mathrm{th}$')
        
        ax_actions.legend()

        plt.tight_layout()
        if log_data:
            plt.savefig(
                self.data_path+f"{dt_string}_actions.pdf", 
                dpi=self.dpi
            )
            
            observations.to_csv(self.data_path+f"{dt_string}_observations.csv")
            actions = pd.DataFrame(
                data=self.actions,
            )
            actions.to_csv(self.data_path+f"{dt_string}_actions.csv")
            
        plt.show()
    