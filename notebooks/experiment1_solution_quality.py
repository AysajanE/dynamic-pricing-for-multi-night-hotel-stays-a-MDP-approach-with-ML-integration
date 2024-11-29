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

@dataclass(frozen=True)
class DPState:
    """Immutable state representation for Dynamic Programming solution."""
    capacity: Tuple[int, ...]  # (capacity_day1, capacity_day2, capacity_day3)
    time: int

class DynamicProgramming:
    """Dynamic Programming solution for hotel revenue optimization."""
    
    def __init__(self, instance: Dict):
        """Initialize using test instance from data_generator."""
        self.params = instance['parameters']
        self.T = self.params.T
        self.N = self.params.N
        self.C = self.params.C
        self.price_min = self.params.price_min
        self.price_max = self.params.price_max
        
        self.booking_classes = instance['booking_classes']
        self.arrival_probs = instance['arrival_probabilities']
        self.price_sensitivity = instance['reservation_price_params']
        
        # Use exactly 3 price levels as specified
        self.price_levels = [self.price_min, (self.price_min + self.price_max)/2, self.price_max]
        self.price_combinations = list(itertools.product(self.price_levels, repeat=self.N))
        
        # Pre-compute stay patterns
        self.class_stays = {
            (arr, dep): set(range(arr - 1, dep)) 
            for arr, dep in self.booking_classes
        }
        self.stay_lengths = {
            (arr, dep): dep - arr + 1 
            for arr, dep in self.booking_classes
        }
        
        # Initialize value function and policy
        self.value_function: Dict[DPState, float] = {}
        self.optimal_policy: Dict[DPState, Dict[int, float]] = {}
        
    def solve(self) -> Tuple[Dict[Tuple[int, ...], Dict[int, float]], float]:
        """Solve the DP problem and return value functions and optimal policy."""
        logger.info("Starting DP solution")
        
        # Initialize boundary conditions
        self._initialize_boundary_conditions()
        
        # Backward induction
        for t in range(self.T, 0, -1):
            logger.info(f"Processing time period {t}")
            for capacity in self._generate_capacity_vectors():
                state = DPState(capacity=capacity, time=t)
                optimal_value, optimal_prices = self._compute_optimal_decision(state)
                self.value_function[state] = optimal_value
                
                # Store policy as dictionary mapping day to price
                self.optimal_policy[state] = {
                    i+1: price for i, price in enumerate(optimal_prices)
                }
        
        # Extract and format results for capacity = 5
        results = {}
        for state, prices in self.optimal_policy.items():
            if state.capacity[0] == 5:  # Only for first day capacity = 5
                results[state.capacity] = prices
        
        # Get optimal value for initial state
        initial_state = DPState(
            capacity=tuple(self.C for _ in range(self.N)),
            time=1
        )
        optimal_value = self.value_function[initial_state]
        
        return results, optimal_value
    
    def _initialize_boundary_conditions(self):
        """Initialize boundary conditions."""
        for capacity in self._generate_capacity_vectors():
            # Terminal period conditions
            terminal_state = DPState(capacity=capacity, time=self.T + 1)
            self.value_function[terminal_state] = 0.0
            
            # Zero capacity conditions for all periods
            if sum(capacity) == 0:
                for t in range(1, self.T + 2):
                    state = DPState(capacity=capacity, time=t)
                    self.value_function[state] = 0.0
    
    def _compute_optimal_decision(self, state: DPState) -> Tuple[float, Tuple[float, ...]]:
        """Compute optimal value and prices for a state."""
        max_value = float('-inf')
        optimal_prices = self.price_levels[0:self.N]  # Default to minimum prices
        
        for prices in self.price_combinations:
            value = self._compute_expected_value(state, prices)
            if value > max_value:
                max_value = value
                optimal_prices = prices
        
        return max_value, optimal_prices
    
    def _compute_expected_value(self, state: DPState, prices: Tuple[float, ...]) -> float:
        """Compute expected value for state-prices pair."""
        value = 0.0
        current_probs = self.arrival_probs[state.time]
        
        # No arrival case
        no_arrival_prob = 1.0 - sum(current_probs.values())
        if no_arrival_prob > 0:
            next_state = DPState(capacity=state.capacity, time=state.time + 1)
            value += no_arrival_prob * self.value_function[next_state]
        
        # For each possible booking request
        for (arrival, departure), arrival_prob in current_probs.items():
            if arrival_prob <= 0:
                continue
                
            stay_nights = self.class_stays[(arrival, departure)]
            has_capacity = all(state.capacity[day] > 0 for day in stay_nights)
            
            if has_capacity:
                # Calculate average price and acceptance probability
                stay_prices = [prices[day] for day in stay_nights]
                avg_price = sum(stay_prices) / self.stay_lengths[(arrival, departure)]
                
                eps = self.price_sensitivity[(arrival, departure)]
                accept_prob = max(0, 1 - eps * avg_price)
                
                if accept_prob > 0:
                    # Acceptance case
                    next_capacity = list(state.capacity)
                    for day in stay_nights:
                        next_capacity[day] -= 1
                    
                    next_state = DPState(capacity=tuple(next_capacity), time=state.time + 1)
                    immediate_revenue = sum(stay_prices)
                    future_value = self.value_function[next_state]
                    
                    value += arrival_prob * accept_prob * (immediate_revenue + future_value)
                
                # Rejection case
                if accept_prob < 1:
                    reject_prob = 1 - accept_prob
                    next_state = DPState(capacity=state.capacity, time=state.time + 1)
                    value += arrival_prob * reject_prob * self.value_function[next_state]
            else:
                # No capacity case
                next_state = DPState(capacity=state.capacity, time=state.time + 1)
                value += arrival_prob * self.value_function[next_state]
        
        return value
    
    def _generate_capacity_vectors(self) -> List[Tuple[int, ...]]:
        """Generate all possible capacity vectors."""
        return [tuple(cap) for cap in itertools.product(range(self.C + 1), repeat=self.N)]

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