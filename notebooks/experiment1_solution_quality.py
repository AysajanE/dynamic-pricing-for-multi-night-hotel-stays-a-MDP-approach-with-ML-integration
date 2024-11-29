import itertools
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
import time
from data_generator import TestConfiguration, create_test_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DPState:
    """State representation for the Dynamic Programming solution.
    
    Attributes:
        capacity: Vector of remaining room capacity for each day in service horizon
        time: Current time period in the booking horizon
    """
    capacity: np.ndarray  # Capacity vector for each day
    time: int            # Current time period
    
    def __post_init__(self):
        """Ensure capacity is numpy array with integer type."""
        if not isinstance(self.capacity, np.ndarray):
            self.capacity = np.array(self.capacity, dtype=np.int32)
    
    def __hash__(self):
        return hash((tuple(self.capacity.tolist()), self.time))
    
    def __eq__(self, other):
        if not isinstance(other, DPState):
            return False
        return (np.array_equal(self.capacity, other.capacity) and 
                self.time == other.time)

class DynamicProgramming:
    """Dynamic Programming solution for hotel room pricing optimization.
    
    This implementation follows the exact mathematical formulation from the paper,
    solving for optimal pricing policies that maximize expected revenue.
    """
    
    def __init__(self, T: int, N: int, C: int, booking_classes: List[Tuple[int, int]], 
                 arrival_probs: Dict[int, Dict[Tuple[int, int], float]], 
                 price_sensitivity: Dict[Tuple[int, int], float],
                 price_bounds: Tuple[float, float]):
        """Initialize the DP solver with problem parameters.
        
        Args:
            T: Number of booking periods
            N: Number of days in service horizon
            C: Room capacity
            booking_classes: List of (arrival_day, departure_day) tuples
            arrival_probs: Dict mapping period to {booking_class: probability}
            price_sensitivity: Dict mapping booking class to epsilon value
            price_bounds: Tuple of (minimum_price, maximum_price)
        """
        self.T = T
        self.N = N
        self.C = C
        self.booking_classes = booking_classes
        self.arrival_probs = arrival_probs
        self.price_sensitivity = price_sensitivity
        self.price_min, self.price_max = price_bounds
        
        self.value_function = {}  # State -> Value mapping
        logger.info(f"Initialized DP solver with T={T}, N={N}, C={C}")
    
    def solve(self) -> Tuple[float, Dict[DPState, float]]:
        """Solve the dynamic programming problem.
        
        Returns:
            Tuple of (optimal_value, value_function)
        """
        logger.info("Starting DP solution")
        
        # Initialize boundary conditions
        self._initialize_boundary_conditions()
        
        # Backward induction
        for t in range(self.T, 0, -1):
            logger.info(f"Processing time period {t}")
            for capacity in self._generate_capacity_vectors():
                state = DPState(capacity=capacity, time=t)
                if np.all(state.capacity == 0):
                    continue  # Skip computation for sold-out states
                self.value_function[state] = self._compute_value_function(state)
        
        # Return optimal value and full value function
        initial_state = DPState(
            capacity=np.full(self.N, self.C, dtype=np.int32),
            time=1
        )
        return self.value_function[initial_state], self.value_function
    
    def _initialize_boundary_conditions(self):
        """Initialize boundary conditions for the value function.
        
        Sets V(x,T+1) = 0 for all states and V(0,t) = 0 for all periods.
        """
        # Terminal period conditions: V(x,T+1) = 0 for all x
        for capacity in self._generate_capacity_vectors():
            terminal_state = DPState(capacity=capacity, time=self.T + 1)
            self.value_function[terminal_state] = 0.0
        
        # Zero capacity conditions: V(0,t) = 0 for all t
        zero_capacity = np.zeros(self.N, dtype=np.int32)
        for t in range(1, self.T + 2):
            sold_out_state = DPState(capacity=zero_capacity, time=t)
            self.value_function[sold_out_state] = 0.0
    
    def _compute_value_function(self, state: DPState) -> float:
        """Compute the value function for a given state.
        
        Args:
            state: Current state (capacity vector and time period)
            
        Returns:
            Value function V(x,t) for the given state
        """
        if state in self.value_function:
            return self.value_function[state]
        
        current_probs = self.arrival_probs[state.time]
        max_value = float('-inf')
        
        # Try different prices
        for price in np.linspace(self.price_min, self.price_max, num=50):
            value = 0.0
            
            # No arrival probability
            no_arrival_prob = 1.0 - sum(current_probs.values())
            if no_arrival_prob > 0:
                next_state = DPState(
                    capacity=state.capacity.copy(),
                    time=state.time + 1
                )
                value += no_arrival_prob * self.value_function[next_state]
            
            # For each possible booking class arrival
            for (arrival, departure), arrival_prob in current_probs.items():
                # Skip if probability is zero
                if arrival_prob <= 0:
                    continue
                    
                # Check capacity availability
                has_capacity = True
                for day in range(arrival - 1, departure):
                    if state.capacity[day] < 1:
                        has_capacity = False
                        break
                
                # Compute acceptance probability
                eps = self.price_sensitivity[(arrival, departure)]
                accept_prob = max(0, 1 - eps * price)
                
                if has_capacity and accept_prob > 0:
                    # State transition for accepted booking
                    next_capacity = state.capacity.copy()
                    for day in range(arrival - 1, departure):
                        next_capacity[day] -= 1
                    next_state = DPState(
                        capacity=next_capacity,
                        time=state.time + 1
                    )
                    
                    # Revenue calculation
                    stay_length = departure - arrival + 1
                    immediate_revenue = stay_length * price
                    
                    # Add to value function
                    value += arrival_prob * accept_prob * (
                        immediate_revenue + self.value_function[next_state]
                    )
                
                # Handle rejection case
                reject_prob = 1 - accept_prob
                next_state = DPState(
                    capacity=state.capacity.copy(),
                    time=state.time + 1
                )
                value += arrival_prob * reject_prob * self.value_function[next_state]
            
            max_value = max(max_value, value)
        
        return max_value
    
    def _generate_capacity_vectors(self) -> List[np.ndarray]:
        """Generate all possible capacity vectors for the service horizon.
        
        Returns:
            List of all possible capacity vectors from 0 to C for each day
        """
        capacities = []
        for values in itertools.product(range(self.C + 1), repeat=self.N):
            capacity = np.array(values, dtype=np.int32)
            capacities.append(capacity)
        return capacities

class StochasticApproximation:
    """Implementation of the Stochastic Approximation Algorithm."""
    
    def __init__(self, instance: Dict, learning_rate: float = 0.1, 
                 num_iterations: int = 1000):
        self.instance = instance
        self.params = instance['parameters']
        self.booking_classes = instance['booking_classes']
        self.arrival_probs = instance['arrival_probabilities']
        self.epsilon = instance['reservation_price_params']
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.prices = self._initialize_prices()
        
    def _initialize_prices(self) -> Dict[int, np.ndarray]:
        """Initialize price vectors for each time period."""
        return {t: np.full(self.params.N, 
                         (self.params.price_min + self.params.price_max) / 2)
                for t in range(1, self.params.T + 1)}
    
    def solve(self) -> Tuple[float, float]:
        """Run the SAA algorithm and return the expected revenue."""
        start_time = time.time()
        
        for iteration in range(self.num_iterations):
            if iteration % 100 == 0:
                logger.info(f"SAA iteration {iteration}")
            
            # Generate sample path
            sample_path = self._generate_sample_path()
            
            # Forward pass
            revenue, gradients = self._forward_pass(sample_path)
            
            # Update prices using gradients
            self._update_prices(gradients)
        
        # Evaluate final solution
        final_revenue = self._evaluate_solution()
        solve_time = time.time() - start_time
        
        logger.info(f"SAA solution completed in {solve_time:.2f} seconds")
        return final_revenue, solve_time
    
    def _generate_sample_path(self) -> List[Tuple]:
        """Generate a sample path of customer arrivals and reservation prices."""
        path = []
        for t in range(1, self.params.T + 1):
            # Generate arrival
            probs = self.arrival_probs[t]
            classes = list(probs.keys())
            probabilities = list(probs.values())
            
            if np.random.random() < sum(probabilities):
                booking_class = np.random.choice(len(classes), p=probabilities/sum(probabilities))
                arrival, departure = classes[booking_class]
                
                # Generate reservation price
                eps = self.epsilon[(arrival, departure)]
                max_price = 1/eps  # Price where acceptance probability becomes 0
                reservation_price = np.random.uniform(self.params.price_min, max_price)
                
                path.append((t, arrival, departure, reservation_price))
            else:
                path.append((t, None, None, None))
        
        return path
    
    def _forward_pass(self, sample_path: List[Tuple]) -> Tuple[float, Dict]:
        """Perform forward pass through the sample path and compute gradients."""
        total_revenue = 0
        gradients = {t: np.zeros(self.params.N) for t in range(1, self.params.T + 1)}
        capacity = np.full(self.params.N, self.params.C)
        
        for t, arrival, departure, reservation_price in sample_path:
            if arrival is not None:
                # Check capacity
                has_capacity = True
                for day in range(arrival - 1, departure):
                    if capacity[day] < 1:
                        has_capacity = False
                        break
                
                if has_capacity:
                    # Calculate average price for the stay
                    stay_prices = self.prices[t][arrival-1:departure]
                    avg_price = np.mean(stay_prices)
                    
                    # Check if customer accepts price
                    if reservation_price >= avg_price:
                        # Accept booking
                        revenue = (departure - arrival + 1) * avg_price
                        total_revenue += revenue
                        
                        # Update capacity
                        for day in range(arrival - 1, departure):
                            capacity[day] -= 1
                        
                        # Compute price gradients
                        for day in range(arrival - 1, departure):
                            gradients[t][day] += 1  # Simplified gradient
        
        return total_revenue, gradients
    
    def _update_prices(self, gradients: Dict):
        """Update prices using computed gradients."""
        for t in range(1, self.params.T + 1):
            self.prices[t] += self.learning_rate * gradients[t]
            # Project prices to feasible range
            self.prices[t] = np.clip(self.prices[t], 
                                   self.params.price_min, 
                                   self.params.price_max)
    
    def _evaluate_solution(self, num_samples: int = 1000) -> float:
        """Evaluate the current solution using Monte Carlo simulation."""
        total_revenue = 0
        
        for _ in range(num_samples):
            sample_path = self._generate_sample_path()
            revenue, _ = self._forward_pass(sample_path)
            total_revenue += revenue
        
        return total_revenue / num_samples