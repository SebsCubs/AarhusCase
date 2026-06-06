"""
Gym environment wrapper for T4B models, 
It implements the step and reset functions for the environment.
Each environment requires a defined model and creates a vectorized object 
that inherits from the tb.Simulator class. This class allows for the simulation to 
receive actions and return the next state, reward, done, and info every step.

Author: Sebastian Cubides @SebsCubs in github

"""
import twin4build as tb
import gymnasium as gym
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from twin4build.systems.saref4syst.system import System
from datetime import datetime, timedelta
from gymnasium import spaces
import logging
import json
import os
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

class GymSimulator(tb.Simulator):
    def __init__(self, model, enable_logging: bool = False):
        """Initialize the gym simulator with a twin4build model.
        
        Args:
            model: A twin4build model instance
            enable_logging: Whether to enable debug logging. Defaults to True.
        """
        super().__init__(model)
        self.control_inputs: Dict[str, Dict[str, Any]] = {}  # Maps component_id to {input_name: current_value}
        self.observation_outputs: Dict[str, Dict[str, Any]] = {}  # Maps component_id to {output_name: current_value}
        # Actions targeting an output port (e.g. ScheduleSystem.scheduleValue, which
        # has no input). Applied as output overrides AFTER do_step.
        self._output_override_signals: set = set()
        self.current_step = 0
        self.model = model
        # Configure logging
        self.enable_logging = enable_logging
        if enable_logging:
            logger.setLevel(logging.INFO)
            logger.info(f"Initialized gym_simulator with model: {model.__class__.__name__}")
        else:
            logger.setLevel(logging.WARNING)
        
    def initialize_simulation(self, startTime: datetime, endTime: datetime, stepSize: int) -> None:
        """Initialize the simulation parameters and model.
        
        Args:
            startTime (datetime): Start time of the simulation (must have timezone)
            endTime (datetime): End time of the simulation (must have timezone)
            stepSize (int): Step size in seconds
            
        Raises:
            AssertionError: If input parameters are invalid or missing timezone info
        """
        assert startTime.tzinfo is not None, "The argument startTime must have a timezone"
        assert endTime.tzinfo is not None, "The argument endTime must have a timezone"
        assert isinstance(stepSize, int), "The argument stepSize must be an integer"
        
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize
        self.current_step = 0

        # Initialize model and generate timesteps (dev-branch snake_case API;
        # per-period list arguments).
        self.model.initialize(start_time=[startTime], end_time=[endTime], step_size=[stepSize])
        sts, dts, max_timesteps, _ = tb.Simulator.get_simulation_timesteps(
            startTime, endTime, stepSize
        )
        # sts/dts are 2D (n_periods x n_steps); the gym uses a single period.
        self._second_time_steps_2d = sts
        self._date_time_steps_2d = dts
        self.n_timesteps = max_timesteps
        self.secondTimeSteps = list(sts[0])
        self.dateTimeSteps = list(dts[0])
        self.secondTime = self._second_time_steps_2d[:, self.current_step]
        self.dateTime = self.dateTimeSteps[self.current_step]
        
        if self.enable_logging:
            total_steps = len(self.secondTimeSteps)
            duration = endTime - startTime
            logger.info(f"Simulation initialized: {startTime} -> {endTime}")
            logger.info(f"Step size: {stepSize}s, Total steps: {total_steps}")
            logger.info(f"Simulation duration: {duration}")
        
    def _do_component_timestep(self, component: System, second_time, date_time,
                               step_size, step_index: int) -> None:
        """Step a single component for the dev-branch API.

        Inputs are assigned from connections (history-indexed), then gym control
        inputs are injected, then do_step runs, then output-override actions (e.g.
        schedule setpoints) are applied and observed outputs cached.
        """
        # 1. Assign inputs from connected components (handles vector port indices).
        tb.Simulator._assign_component_inputs(component, step_index)

        # 2. Apply input-type control actions (override the connection value).
        if component.id in self.control_inputs:
            for input_name, value in self.control_inputs[component.id].items():
                if input_name in component.input and not isinstance(value, dict) and value is not None:
                    component.input[input_name].set(float(value), i_t=step_index)

        # 3. Execute the component step.
        component.do_step(second_time, date_time, step_size, step_index)

        # 4. Apply output-override actions (ScheduleSystem setpoints, etc.) AFTER
        #    do_step so downstream components read the overridden value this step.
        if component.id in self.control_inputs:
            for signal_name, value in self.control_inputs[component.id].items():
                if (component.id, signal_name) in self._output_override_signals \
                        and not isinstance(value, dict) and value is not None:
                    out = component.output[signal_name]
                    # Schedule outputs are leaf scalars (value lives in history),
                    # so the public setter is rejected; write the tensor directly.
                    fv = float(value)
                    out._tensor[:] = fv
                    if getattr(out, "_log_history", False):
                        out._history[step_index] = fv

        # 5. Cache observed outputs.
        if component.id in self.observation_outputs:
            for output_name in self.observation_outputs[component.id]:
                if output_name in component.output:
                    self.observation_outputs[component.id][output_name] = \
                        component.output[output_name].get()

    def _do_system_time_step(self, model, second_time, date_time, step_size,
                             step_index: int) -> None:
        """Execute one timestep for all components in execution order."""
        for component_group in self.model.execution_order:
            for component in component_group:
                self._do_component_timestep(
                    component, second_time, date_time, step_size, step_index
                )

    def add_control_input(self, component_id: str, input_name: str) -> None:
        """Add a control input to the simulator.
        
        Args:
            component_id: ID of the component to control
            input_name: Name of the input parameter to control
        """
        if component_id not in self.control_inputs:
            self.control_inputs[component_id] = {}
        self.control_inputs[component_id][input_name] = {}  # Initialize with an empty dictionary
        
    def add_observation_output(self, component_id: str, output_name: str) -> None:
        """Add an observation output to monitor.
        
        Args:
            component_id: ID of the component to observe
            output_name: Name of the output parameter to monitor
        """
        if component_id not in self.observation_outputs:
            self.observation_outputs[component_id] = {}
        self.observation_outputs[component_id][output_name] = {}  # Initialize with an empty dictionary
                        
    def _populate_from_json(self, json_file):
        """Populate actions and observations from a JSON file.
        
        Args:
            json_file: Path to the JSON file containing actions and observations
        """
        with open(json_file, 'r') as f:
            config = json.load(f)

        #assert the format of the json file
        assert 'actions' in config, "The JSON file must contain an 'actions' key"
        assert 'observations' in config, "The JSON file must contain an 'observations' key"
  
        # Add control inputs from actions
        for component_id, signals in config['actions'].items():
            component = self.model.components[component_id]
            for signal_name, signal_config in signals.items():
                # A signal is controllable either as an input port or, for
                # source components like ScheduleSystem (no inputs), as an output
                # port that we override after do_step.
                if signal_name in component.input:
                    pass
                elif signal_name in component.output:
                    self._output_override_signals.add((component_id, signal_name))
                else:
                    raise ValueError(f"Signal {signal_name} not found in component {component_id}")
                self.add_control_input(component_id, signal_name)
                # Add the max and min values for the action space
                self.control_inputs[component_id][signal_name]['max'] = signal_config['max']
                self.control_inputs[component_id][signal_name]['min'] = signal_config['min']
            
        # Add observation outputs from observations
        for component_id, signals in config['observations'].items():
            for signal_name, signal_config in signals.items():
                self.add_observation_output(component_id, signal_name)
                # Add the max and min values for the observation space
                self.observation_outputs[component_id][signal_name]['max'] = signal_config['max']
                self.observation_outputs[component_id][signal_name]['min'] = signal_config['min']

    def get_observations(self) -> Dict[str, Dict[str, float]]:
        """Get current observations from monitored outputs.
        
        Returns:
            Dictionary mapping component IDs to their output values
            Format: {component_id: {output_name: value}}
        """
        observations = {}
        for component_id, outputs in self.observation_outputs.items():
            #Check that the component is in the model
            if component_id not in self.model.components:
                raise ValueError(f"Component {component_id} not found in the model")
            
    
            component = self.model.components[component_id]
            observations[component_id] = {}
            for output_name in outputs:
                #Check that the output is in the component
                if output_name not in component.output:
                    raise ValueError(f"Output {output_name} not found in component {component_id}")
                
                value = component.output[output_name].get()
                if value is None:
                    value = 0.0
                else:
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = float(np.asarray(value).reshape(-1)[-1])
                observations[component_id][output_name] = value
                self.observation_outputs[component_id][output_name] = value
        return observations
    
    def step_simulation(self, actions: Optional[np.ndarray] = None) -> Tuple[Dict[str, Dict[str, float]], bool]:
        """Perform one simulation step with the given actions.
        
        This method advances the simulation by one timestep, similar to the original
        simulate method but allowing for step-by-step control. It:
        1. Updates control input storage with new actions
        2. Updates simulation time
        3. Executes one system timestep (which applies controls via _do_component_timestep)
        4. Returns current observations and done status
        
        Args:
            actions: Control actions as a numpy array, ordered according to self.control_inputs
            
        Returns:
            Tuple containing:
            - Current observations after applying actions
            - Boolean indicating if simulation is complete (reached endTime)
        """
        # Check if we've reached the end of simulation
        if self.current_step >= len(self.secondTimeSteps):
            if self.enable_logging:
                logger.info("Simulation complete")
            return self.get_observations(), True
            
        # Store the new control actions (will be applied during component timesteps)
        if self.enable_logging:
            logger.info(f"Step {self.current_step + 1}/{len(self.secondTimeSteps)}")
            logger.debug(f"Time: {self.dateTimeSteps[self.current_step]}")
        
        if actions is not None:
            # Convert array action to dictionary using control_inputs order
            action_idx = 0
            for component_id, inputs in self.control_inputs.items():
                for input_name in inputs:
                    self.control_inputs[component_id][input_name] = actions[action_idx]
                    action_idx += 1
        elif self.enable_logging:
            logger.info("No actions provided, using default values")
        
        # Get current timestep values (dev-branch shapes: per-period columns).
        step_index = self.current_step
        second_time = self._second_time_steps_2d[:, step_index]
        date_time = self._date_time_steps_2d[:, step_index]
        self.secondTime = second_time
        self.dateTime = self.dateTimeSteps[step_index]

        # Execute one system timestep.
        self._do_system_time_step(
            self.model, second_time, date_time, [self.stepSize], step_index
        )

        # Increment step counter
        self.current_step += 1

        done = self.current_step >= len(self.secondTimeSteps)

        return done
    

class T4BGymEnv(gym.Env):
    """
    Gymnasium environment wrapper for Twin4Build models.
    
    This environment provides a standard gym interface for interacting with
    Twin4Build simulation models, allowing for reinforcement learning applications.

    Things to be defined by the user:
    - start_time (conditional on the model data available)
    - end_time (conditional on the model data available)
    - episode_length (can be smaller than the total simulation time)
    - random_start (boolean to define if the start time is random or not (only used if episode_length is smaller than the total simulation time))
    - excluding_periods (list of tuples defining periods of the simulation that should be excluded from the training data)
    - stepSize
    - io_config_file (this is defining action and observation spaces)
    - warmup_period (not implemented yet, but would be a period of the simulation that runs before the training starts)
    - reward_function (implemented by inheriting from the class and overriding the method)

    """
    
    def __init__(self, 
                 model: tb.Model, 
                 io_config_file: str, # Mandatory for now
                 start_time: datetime = None,
                 end_time: datetime = None,
                 episode_length: int = None,
                 random_start = False,
                 excluding_periods: List[Tuple[datetime, datetime]] = None,
                 forecast_horizon: int = 0,
                 step_size: int = 600,
                 warmup_period = 0):
        """Initialize the gym environment.
        
        Args:
            model: Twin4Build model instance
            io_config_file: Path to the JSON file containing actions and observations
            start_time: Start time of the simulation (must have timezone)
            end_time: End time of the simulation (must have timezone)
            episode_length: Length of each episode in steps (must be smaller than total simulation time)
            random_start: Whether to start episodes at random times within the simulation period
            excluding_periods: List of (start, end) datetime tuples defining periods to exclude from training
            step_size: Simulation step size in seconds
            warmup_period: Number of steps to run before starting the episode (not implemented yet)
        """
        super().__init__()
        self.simulator = GymSimulator(model)

        # Set simulation parameters
        self.step_size = step_size

        self.global_start_time = start_time 
        self.global_end_time = end_time
        self.episode_length = episode_length
        self.random_start = random_start
        self.excluding_periods = excluding_periods or []
        self.warmup_period = warmup_period
        self.forecast_horizon = forecast_horizon

        # Set up control inputs and observation outputs if io_config_file is provided
        assert io_config_file is not None, """io_config_file is mandatory. The JSON file should have this format:
                            {
                                "actions": {
                                    "Component_ID": {
                                        "signal_key": "input_name",
                                        "min": float,
                                        "max": float,
                                        "description": "string"
                                    }
                                },
                                "observations": {
                                    "Component_ID": {
                                        "signal_key": "output_name",
                                        "min": float,
                                        "max": float,
                                        "description": "string"
                                    }
                                }
                            }"""
        #Assert that the io_config_file has the correct format
        assert os.path.exists(io_config_file), "The io_config_file does not exist"
        assert os.path.isfile(io_config_file), "The io_config_file is not a file"   
        assert os.path.splitext(io_config_file)[1] == '.json', "The io_config_file must be a JSON file"
        with open(io_config_file) as f:
            io_dict = json.load(f)
        assert 'actions' in io_dict, "The io_config_file must contain an 'actions' key"
        assert 'observations' in io_dict, "The io_config_file must contain an 'observations' key"
        
        self.io_config_dict = io_dict

        if self.episode_length is not None:
            #Assert that the episode length is smaller than the total simulation time, raise a value error if not
            if self.episode_length > (self.global_end_time - self.global_start_time).total_seconds()/self.step_size:
                raise ValueError("Episode length must be smaller than the total simulation time")
        if self.random_start and self.excluding_periods is not None:
            #Assert that there is at least one chunk of available time >= episode_length
            if self.excluding_periods is not None and not self._check_available_time_chunks():
                raise ValueError("Excluding periods fragment the available time into chunks smaller than episode_length")
            for start, end in self.excluding_periods:
                    if start < self.global_start_time or end > self.global_end_time:
                        raise ValueError("Excluding periods must be within the total simulation time")
        
        #Populate the simulator with the actions and observations
        self.simulator._populate_from_json(io_config_file)        
        self.create_observation_space()
        self.create_action_space()

        #self.simulator.initialize_simulation(self.start_time, self.end_time, self.step_size)
    
    def _check_available_time_chunks(self):
        """Check if excluding periods fragment the available time into chunks smaller than episode length.
        
        Returns:
            bool: True if there is at least one chunk of available time >= episode_length
        """
        # Sort excluding periods by start time
        sorted_periods = sorted(self.excluding_periods, key=lambda x: x[0])
        
        # Get all time boundaries
        boundaries = [self.global_start_time] + [t for period in sorted_periods for t in period] + [self.global_end_time]
        
        # Find available chunks between excluding periods
        available_chunks = []
        for i in range(0, len(boundaries)-1, 2):
            chunk_start = boundaries[i]
            chunk_end = boundaries[i+1]
            chunk_duration = (chunk_end - chunk_start).total_seconds() / self.step_size
            available_chunks.append(chunk_duration)
        
        # Check if any chunk is large enough for the episode
        return any(chunk >= self.episode_length for chunk in available_chunks)

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility (unused)
            options: Additional options for reset (unused)
            
        Returns:
            tuple: (observations, info)
                observations: Initial state observations
                info: Additional information
        """

        if self.episode_length is not None:
            self.sim_start_time = self.global_start_time
            self.sim_end_time = self.global_start_time + timedelta(seconds=self.episode_length*self.step_size)

        if self.random_start:

            #Generate a random start time
            max_start_time = self.global_end_time - timedelta(seconds=self.episode_length*self.step_size)
            random_start_time = self.global_start_time + timedelta(seconds=random.randint(0, int((max_start_time - self.global_start_time).total_seconds())))
            
            #Check that end time is not beyond the global end time
            episode_end_time = min(random_start_time + timedelta(seconds=self.episode_length*self.step_size), self.global_end_time)
            
            #Check if any part of the episode overlaps with excluding periods
            if self.excluding_periods is not None:
                max_attempts = 1000  # Maximum number of attempts to find a valid time slot
                attempts = 0
                while any((start <= random_start_time < end) or 
                         (start < episode_end_time <= end) or
                         (random_start_time <= start and episode_end_time >= end) 
                         for start, end in self.excluding_periods):
                    random_start_time = self.global_start_time + timedelta(seconds=random.randint(0, int((max_start_time - self.global_start_time).total_seconds())))
                    episode_end_time = min(random_start_time + timedelta(seconds=self.episode_length*self.step_size), self.global_end_time)
                    attempts += 1
                    if attempts >= max_attempts:
                        raise ValueError("Could not find a valid time slot after maximum attempts. The excluding periods may leave no room for the episode length.")

            self.sim_start_time = random_start_time
            self.sim_end_time = episode_end_time
            
        if self.episode_length is None:
            self.sim_start_time = self.global_start_time
            self.sim_end_time = self.global_end_time
        
        # Reset simulator state
        self.simulator.initialize_simulation(self.sim_start_time, self.sim_end_time, self.step_size)
        
        # Get initial observations
        observations = self._get_obs()
        
        return observations, {}
    
    def step(self, action: Dict[str, Dict[str, float]]):
        """Take a step in the environment.
        
        Args:
            action: Dictionary of control actions
                   Format: {component_id: {input_name: value}}
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                observation: Current state observation
                reward: Reward from the action
                terminated: Whether episode is done
                truncated: Whether episode was artificially terminated
                info: Additional information
        """
        #Assert the format of the action
        assert isinstance(action, np.ndarray), "The action must be a numpy array"
        assert action.shape == self.action_space.shape, f"Action shape {action.shape} must match action space shape {self.action_space.shape}"
        assert np.all(action >= self.action_space.low) and np.all(action <= self.action_space.high), "Action values must be within action space bounds"

        #if action contains NaN, set it to 0
        if np.isnan(action).any():
            action = np.nan_to_num(action, 0.0)

        # Apply action and get new observations
        done = self.simulator.step_simulation(action)
        
        observations = self._get_obs()
        if np.isnan(observations).any():
            raise ValueError("Observations contain NaN")

        # Calculate reward (placeholder - should be implemented based on specific task)
        reward = self.get_reward(observations, action)
        
        # Check if episode is done (placeholder)
        terminated = done
        truncated = False
        
        # Additional info
        info = {}
        
        return observations, reward, terminated, truncated, info
    
    def get_reward(self, observations, action) -> float:
        """Calculate the reward based on the observations and action.
        
        Args:
            observations: Current state observations
            action: Control actions applied
        Returns:
            float: Reward value
        """
        #Placeholder for the reward function, meant to be implemented by the user
        return 0.0

    def create_action_space(self):
        """Create the action space based on the control inputs defined in the simulator.
        
        Returns:
            spaces.Dict: Action space
        """
        
        #Define the action and observation spaces, if io_config_file is provided, max and min values are defined in the json file
        action_keys = []
        low_bounds = []
        upper_bounds = []

        for component_id, inputs in self.simulator.control_inputs.items():
            for input_name in inputs:
                action_keys.append(input_name)
                low_bounds.append(self.simulator.control_inputs[component_id][input_name]['min'])
                upper_bounds.append(self.simulator.control_inputs[component_id][input_name]['max'])

        #Define the action space
        self.action_space = spaces.Box(low=np.array(low_bounds), high=np.array(upper_bounds), dtype=np.float32)
    
    def create_observation_space(self):
        """Create the observation space based on the observations defined in the simulator and the io_config_file.
        1. Gets the populated component outputs from the simulator
        2. Checks if config file contains "time_embeddings" or "forecasts" keys
        3. If "time_embeddings" key is present, the observation space is extended based on the time embeddings
        4. If "forecasts" key is present, the observation space is extended based on the forecasts
        5. If neither key is present, the observation space is defined based on only the component outputs
        Returns:
            spaces.Dict: Observation space
        """
        #Get the populated component outputs from the simulator
        observations = []
        low_bounds = []
        upper_bounds = []

        for component_id, outputs in self.simulator.observation_outputs.items():
            for output_name in outputs:
                observations.append(outputs[output_name])
                low_bounds.append(self.simulator.observation_outputs[component_id][output_name]['min'])
                upper_bounds.append(self.simulator.observation_outputs[component_id][output_name]['max'])

        #Check if config file contains "time_embeddings" or "forecasts" keys
        if 'time_embeddings' in self.io_config_dict:
            time_embedding_keys = list(self.io_config_dict['time_embeddings'].keys())
            for key in time_embedding_keys:
                #Using sin and cos embeddings for the time of day, day of week, month of year
                #Therefore there are two observations for each time embedding
                observations.append(self.io_config_dict['time_embeddings'][key]["signal_key"] + "_sin")
                observations.append(self.io_config_dict['time_embeddings'][key]["signal_key"] + "_cos")
                low_bounds.append(-1)
                low_bounds.append(-1)
                upper_bounds.append(1)
                upper_bounds.append(1)

        if 'forecasts' in self.io_config_dict:
            forecast_keys = list(self.io_config_dict['forecasts'].keys())
            for key in forecast_keys:
                # For each forecast component, iterate through its signals
                for signal_name, signal_config in self.io_config_dict['forecasts'][key].items():
                    # For each signal, we need forecast_horizon + 1 dimensions (current value + future values)
                    for _ in range(self.forecast_horizon + 1):
                        observations.append(signal_name)
                        low_bounds.append(signal_config['min'])
                        upper_bounds.append(signal_config['max'])

        #Define the observation space
        self.observation_space = spaces.Box(low=np.array(low_bounds), high=np.array(upper_bounds), dtype=np.float32)

    def _get_forecast(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """Helper method to get forecast window for a given column.
        
        Args:
            df: DataFrame containing the forecast data
            column_name: Name of the column to get forecast for
            
        Returns:
            pd.Series: Forecast window padded if needed
        """
        start_idx = self.simulator.current_step
        end_idx = start_idx + self.forecast_horizon + 1
        
        # Get available forecast values
        forecast = df[column_name].iloc[start_idx:min(end_idx, len(df))]
        
        if len(forecast) < self.forecast_horizon + 1:
            # If we're at the end of the episode, use the last known value for padding
            if start_idx >= len(df) - 1:
                last_value = df[column_name].iloc[-1]
            else:
                # Otherwise use the last value from our current forecast window
                last_value = forecast.iloc[-1]
                
            # Pad with last value to reach desired length
            padding_length = self.forecast_horizon + 1 - len(forecast)
            padding = pd.Series([last_value] * padding_length)
            forecast = pd.concat([forecast, padding])
            
        return forecast

    def _get_obs(self):
        """Get the current observations from the simulator.
        
        Returns:
            numpy.ndarray: Current observations
        """
        model_obs = self.simulator.get_observations()

        #Check if config file contains "time_embeddings" or "forecasts" keys
        if 'time_embeddings' in self.io_config_dict:
            current_time = self.simulator.dateTime
            time_embedding_keys = list(self.io_config_dict['time_embeddings'].keys())
            if "time_of_day" in time_embedding_keys:
                # Use sin cos embeddings for the time of day
                model_obs["time_of_day_sin"] = np.sin(2 * np.pi * current_time.hour / 24)
                model_obs["time_of_day_cos"] = np.cos(2 * np.pi * current_time.hour / 24)
            if "day_of_week" in time_embedding_keys:
                model_obs["day_of_week_sin"] = np.sin(2 * np.pi * current_time.weekday() / 7)
                model_obs["day_of_week_cos"] = np.cos(2 * np.pi * current_time.weekday() / 7)
            if "month_of_year" in time_embedding_keys:
                model_obs["month_of_year_sin"] = np.sin(2 * np.pi * current_time.month / 12)
                model_obs["month_of_year_cos"] = np.cos(2 * np.pi * current_time.month / 12)
        if 'forecasts' in self.io_config_dict:
            #Find the OutdoorEnvironment component
            outdoor_env_component = None
            for component in self.simulator.model.components:
                #TODO: Make this more robust
                if component == "outdoor_environment":
                    outdoor_env_component = self.simulator.model.components[component]
                    break
            if outdoor_env_component is None:
                raise ValueError("OutdoorEnvironment component not found in the model")
            #get the df from the OutdoorEnvironment component
            df = outdoor_env_component.df
            
            # Map forecast keys to their corresponding column names
            global_forecast_columns = {
                "outdoor_temperature": "outdoorTemperature",
                "global_irradiation": "globalIrradiation",
            }
            
            # Get forecasts for each requested key
            horizon = self.forecast_horizon + 1
            for component_id, signals in self.io_config_dict['forecasts'].items():
                if component_id in global_forecast_columns:
                    col = global_forecast_columns[component_id]
                    for signal_name, signal_config in signals.items():
                        if df is not None and col in getattr(df, "columns", []):
                            model_obs[signal_name] = self._get_forecast(df, col).values
                        else:
                            # Dev-branch OutdoorEnvironmentSystem stores per-signal
                            # time series (no combined .df). Fall back to a flat
                            # forecast from the component's current output value.
                            try:
                                cur = float(outdoor_env_component.output[col].get())
                            except Exception:
                                cur = 0.0
                            model_obs[signal_name] = np.full(horizon, cur)


            #Occupancy values vary from model to model, they are provided as a list of schedule components
            #Go through all the remaining keys in the forecasts dictionary in the io_config_file
            for component_id, signals in self.io_config_dict['forecasts'].items():
                if component_id not in global_forecast_columns:
                    for signal_name, signal_config in signals.items():
                        #Get the schedule component
                        component = self.simulator.model.components[component_id]
                        if hasattr(component, 'do_step_instance'):
                            df = component.do_step_instance.df
                            # Create a new DataFrame with the values and proper column name
                            df = pd.DataFrame(df.values, index=df.index, columns=[signal_name])
                            
                            forecast = self._get_forecast(df, signal_name)
                            column_name = component_id + "_" + signal_name
                            model_obs[column_name] = forecast.values
                        else:
                            #Component has a schedule ruleset, not a dataframe, so we need to get the schedule values
                            current_time = self.simulator.dateTime
                            forecast_datetimes = [current_time + timedelta(seconds=i*self.step_size) for i in range(self.forecast_horizon + 1)]
                            #Get the schedule values
                            schedule_values = [component.get_schedule_value(dt) for dt in forecast_datetimes]
                            
                            # For schedule components, we already have the exact forecast array we need
                            column_name = component_id + "_" + signal_name
                            model_obs[column_name] = np.array(schedule_values)

        #Return the observations as a numpy array
        obs = []
        for key in model_obs.keys():
            if isinstance(model_obs[key], dict):
                for output_name in model_obs[key].keys():
                    if isinstance(model_obs[key][output_name], np.ndarray):
                        # If it's a forecast array, append each value
                        obs.extend(model_obs[key][output_name])
                    else:
                        # If it's a single value, append it directly
                        obs.append(model_obs[key][output_name])
            else:
                if isinstance(model_obs[key], np.ndarray):
                    # If it's a forecast array, append each value
                    obs.extend(model_obs[key])
                else:
                    # If it's a single value, append it directly
                    obs.append(model_obs[key])

        return np.array(obs)

class NormalizedObservationWrapper(gym.ObservationWrapper):
    '''This wrapper normalizes the values of the observation space to lie
    between -1 and 1. Normalization can significantly help with convergence
    speed. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = NormalizedObservationWrapper(env)`
    
    '''
    
    def __init__(self, env):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
    def observation(self, observation):
        '''
        This method accepts a single parameter (the 
        observation to be modified) and returns the modified observation.
        
        Parameters
        ----------
        observation: 
            Observation in the original environment observation space format 
            to be modified. 
        
        Returns
        -------
            Modified observation returned by the wrapped environment. 
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ObservationWrapper` parent class is doing in `gym.core`:
        
        '''
        
        # Check for NaN values
        if np.isnan(observation).any():
            raise ValueError("NaN values detected in observation before normalization")
            
        # Check if observation is within bounds
        if not np.all(observation >= self.observation_space.low) or not np.all(observation <= self.observation_space.high):
            #Check which values are outside the bounds
            outside_bounds = np.where(observation < self.observation_space.low)[0]
            outside_bounds = np.where(observation > self.observation_space.high)[0]
            #print(f"Observation values outside of observation space bounds: {outside_bounds}")
            #If the outside bounds are more than 10% of the total observations, raise an error
            if len(outside_bounds) > 0.1*len(observation):
                raise ValueError("Observation values outside of observation space bounds")
            else:
                logging.warning(f"Observation values outside of observation space bounds: {outside_bounds}")
        
        # Convert to one number for the wrapped environment
        observation_wrapper = 2*(observation - self.observation_space.low)/\
            (self.observation_space.high-self.observation_space.low)-1
            
        # Check for NaN values after normalization
        if np.isnan(observation_wrapper).any():
            raise ValueError("NaN values detected in observation after normalization")
            
        # Check if normalized observation is within [-1, 1] bounds
        if not np.all(observation_wrapper >= -1) or not np.all(observation_wrapper <= 1):
            #If values are bigger than 1, set them to 1
            observation_wrapper[observation_wrapper > 1] = 1
            #If values are smaller than -1, set them to -1
            observation_wrapper[observation_wrapper < -1] = -1

        
        return observation_wrapper
     
class NormalizedActionWrapper(gym.ActionWrapper):
    '''This wrapper normalizes the values of the action space to lie
    between -1 and 1. Normalization can significantly help with convergence
    speed. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = NormalizedActionWrapper(env)`
    
    '''
    
    def __init__(self, env):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
        # Assert that original observation space is a Box space
        assert isinstance(self.unwrapped.action_space, spaces.Box), 'This wrapper only works with continuous action space (spaces.Box)'
        
        # Store low and high bounds of action space
        self.low    = self.unwrapped.action_space.low
        self.high   = self.unwrapped.action_space.high
        
        # Check for invalid action space bounds
        if np.any(self.low >= self.high):
            raise ValueError("Action space bounds are invalid: low >= high")
            
        # Check for NaN values in action space bounds
        if np.isnan(self.low).any() or np.isnan(self.high).any():
            raise ValueError("NaN values detected in action space bounds")
        
        # Redefine action space to lie between [-1,1]
        self.action_space = spaces.Box(low = -1, 
                                       high = 1,
                                       shape=self.unwrapped.action_space.shape, 
                                       dtype= np.float32)        
        
    def action(self, action_wrapper):
        '''This method accepts a single parameter (the modified action
        in the wrapper format) and returns the action to be passed to the 
        original environment. Thus, this method basically rescales the  
        action inside the environment.
        
        Parameters
        ----------
        action_wrapper: 
            Action in the modified environment action space format 
            to be reformulated back to the original environment format.
        
        Returns
        -------
            Action in the original environment format.  
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ActionWrapper` parent class is doing in `gym.core`:
        
        '''
        # Check for NaN values in input action
        if np.isnan(action_wrapper).any():
            raise ValueError("NaN values detected in input action")
            
        # Check if input action is within [-1, 1] bounds
        if not np.all(action_wrapper >= -1) or not np.all(action_wrapper <= 1):
            raise ValueError("Input action values outside of [-1, 1] bounds")
            
        action = self.low + (0.5*(action_wrapper+1.0)*(self.high-self.low))
        
        # Check for NaN values in output action
        if np.isnan(action).any():
            raise ValueError("NaN values detected in output action")
            
        # Check if output action is within original bounds
        if not np.all(action >= self.low) or not np.all(action <= self.high):
            raise ValueError("Output action values outside of original action space bounds")
            
        return action