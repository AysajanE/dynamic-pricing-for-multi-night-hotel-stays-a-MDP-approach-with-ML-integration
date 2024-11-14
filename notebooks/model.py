# model.py
import itertools
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProblemParameters:
    """
    Parameters for the hotel dynamic pricing problem.
    
    Attributes:
        N (int): Number of days in service horizon
        T (int): Number of booking periods
        C (int): Total room capacity
        price_min (float): Minimum price
        price_max (float): Maximum price
        price_levels (int): Number of discrete price levels
        arrival_prob_base (float): Base arrival probability for each booking class
    """
    N: int
    T: int
    C: int
    price_min: float
    price_max: float
    price_levels: int
    arrival_prob_base: float

    def validate(self):
        """Validate parameter values."""
        assert self.N > 0, "Service horizon must be positive"
        assert self.T > 0, "Booking periods must be positive"
        assert self.C > 0, "Capacity must be positive"
        assert self.price_min > 0, "Minimum price must be positive"
        assert self.price_max > self.price_min, "Maximum price must be greater than minimum"
        assert self.price_levels > 1, "Need at least 2 price levels"
        assert 0 < self.arrival_prob_base < 1, "Arrival probability must be between 0 and 1"
        
@dataclass
class BookingClass:
    """
    Booking class representation with additional functionality.
    """
    id: int
    arrival: int
    departure: int
    length: int
    stay_days: List[int]
    consumption_vector: Tuple[int, ...]

    @classmethod
    def create(cls, id: int, arrival: int, departure: int, N: int) -> 'BookingClass':
        """Factory method to create a booking class."""
        if not (1 <= arrival <= departure):
            raise ValueError("Invalid arrival/departure combination")
        
        length = departure - arrival + 1
        stay_days = list(range(arrival, departure + 1))
        consumption_vector = tuple(1 if i+1 in stay_days else 0 for i in range(N))
        
        return cls(id, arrival, departure, length, stay_days, consumption_vector)
    
class ReservationPriceModel:
    """
    Models customer reservation price distribution and related functions.
    """
    def __init__(self, price_min: float, price_max: float, distribution: str = 'uniform'):
        self.price_min = price_min
        self.price_max = price_max
        self.distribution = distribution
        
    def cdf(self, p: float) -> float:
        """Cumulative distribution function of reservation price."""
        if self.distribution == 'uniform':
            if p < self.price_min:
                return 0.0
            elif p > self.price_max:
                return 1.0
            else:
                return (p - self.price_min) / (self.price_max - self.price_min)
        elif self.distribution == 'exponential':
            # Example of alternative distribution
            lambda_param = 1 / (self.price_max - self.price_min)
            return 1 - np.exp(-lambda_param * (p - self.price_min))
        
    def survival(self, p: float) -> float:
        """Survival function (1 - CDF)."""
        return 1 - self.cdf(p)

    def expected_revenue(self, p: float) -> float:
        """Expected revenue at price p."""
        return p * self.survival(p)

class DynamicPricingModel:
    """
    Core implementation of the dynamic pricing model.
    """
    def __init__(self, params: ProblemParameters):
        self.params = params
        params.validate()
        
        # Initialize components
        self.initialize_booking_classes()
        self.initialize_state_space()
        self.initialize_price_vectors()
        self.initialize_arrival_probabilities()
        self.price_model = ReservationPriceModel(params.price_min, params.price_max)
        
        # Initialize value function and policy
        self.V = {}
        self.policy = {}
        self.reset_value_function()
        
    def initialize_booking_classes(self):
        """Initialize all possible booking classes."""
        self.booking_classes = []
        id_counter = 1
        
        for a, d in itertools.product(range(1, self.params.N + 1), repeat=2):
            if a <= d:
                try:
                    bc = BookingClass.create(id_counter, a, d, self.params.N)
                    self.booking_classes.append(bc)
                    id_counter += 1
                except ValueError as e:
                    logger.warning(f"Skipping invalid booking class: {e}")
                    
    def initialize_state_space(self):
        """Generate all possible states."""
        self.states = list(itertools.product(
            range(self.params.C + 1), 
            repeat=self.params.N
        ))
        
    def initialize_price_vectors(self):
        """Generate price vectors with specified levels."""
        prices = np.linspace(
            self.params.price_min,
            self.params.price_max,
            self.params.price_levels
        )
        self.price_vectors = list(itertools.product(prices, repeat=self.params.N))
        
    def initialize_arrival_probabilities(self):
        """Initialize arrival probabilities for each booking class and time period."""
        self.pi = {t: {} for t in range(1, self.params.T + 1)}
        
        # Adjust probabilities to ensure sum <= 1 for each period
        base_prob = self.params.arrival_prob_base
        num_classes = len(self.booking_classes)
        adjusted_prob = min(base_prob, 1.0 / num_classes)
        
        for t in range(1, self.params.T + 1):
            for bc in self.booking_classes:
                self.pi[t][bc.id] = adjusted_prob
                
    def reset_value_function(self):
        """Reset value function and policy to initial state."""
        self.V = {state: {t: 0.0 for t in range(1, self.params.T + 2)} 
                 for state in self.states}
        self.policy = {state: {t: None for t in range(1, self.params.T + 1)} 
                      for state in self.states}
        
    def is_available(self, state: Tuple[int, ...], bc: BookingClass) -> bool:
        """Check if booking can be accommodated."""
        return all(state[i] >= bc.consumption_vector[i] for i in range(self.params.N))
    
    def update_state(self, state: Tuple[int, ...], bc: BookingClass) -> Tuple[int, ...]:
        """Update state after accepting a booking."""
        return tuple(s - c for s, c in zip(state, bc.consumption_vector))
    
    def calculate_expected_revenue(self, state: Tuple[int, ...], price_vector: Tuple[float, ...], t: int) -> float:
        """Calculate expected revenue for a given state and price vector."""
        total_expected_value = 0.0
        base_future_value = self.V[state][t+1]  # V(x_t, t+1)

        for bc in self.booking_classes:
            if not self.is_available(state, bc):
                continue

            # Calculate average price for booking class
            stay_prices = [price_vector[day - 1] for day in bc.stay_days]
            avg_price = np.mean(stay_prices)

            # Calculate acceptance probability and expected demand
            acceptance_prob = self.price_model.survival(avg_price)
            # lambda_tb = self.pi[t][bc.id] * acceptance_prob
            u_tb = self.pi[t][bc.id] * acceptance_prob
            
            # Calculate new state value
            new_state = self.update_state(state, bc)
            
            # Calculate marginal value: L^b * p̄_t^b + V(x_t - e^b, t+1) - V(x_t, t+1)
            marginal_value = (bc.length * avg_price + 
                             self.V[new_state][t+1] - 
                             base_future_value)

            total_expected_value += u_tb * marginal_value

        return total_expected_value + base_future_value
        
    def solve(self, use_parallel: bool = False):
        """
        Solve the dynamic pricing problem.
        
        Args:
            use_parallel (bool): Whether to use parallel processing
        """
        start_time = time.time()
        logger.info("Starting solution process...")

        try:
            if use_parallel and __name__ == '__main__':
                self._solve_parallel()
            else:
                self._solve_sequential()
        except Exception as e:
            logger.error(f"Error during solving: {str(e)}")
            logger.info("Falling back to sequential processing")
            self._solve_sequential()

        elapsed_time = time.time() - start_time
        logger.info(f"Solution completed in {elapsed_time:.2f} seconds")
                
    def _solve_sequential(self):
        """Sequential solution method."""
        for t in range(self.params.T, 0, -1):
            logger.info(f"Processing time period {t}")
            for state in tqdm(self.states, desc=f"Processing states for t={t}"):
                result = self._process_state(state, t)
                # Unpack the result correctly based on what _process_state returns
                max_revenue, optimal_price = result[1], result[2]
                self.V[state][t] = max_revenue
                self.policy[state][t] = optimal_price
    
    def _process_state(self, state: Tuple[int, ...], t: int) -> Tuple[Tuple[int, ...], float, Tuple[float, ...]]:
        """
        Process a single state.
        
        Returns:
            Tuple containing (state, max_revenue, optimal_price_vector)
        """
        max_revenue = float('-inf')
        optimal_price_vector = None

        for price_vector in self.price_vectors:
            total_expected_revenue = self.calculate_expected_revenue(state, price_vector, t)
            
            if total_expected_revenue > max_revenue:
                max_revenue = total_expected_revenue
                optimal_price_vector = price_vector

        return state, max_revenue, optimal_price_vector
    
    
    
    def _solve_parallel(self):
        """Parallel solution method using ProcessPoolExecutor."""
        if __name__ != '__main__':
            logger.warning("Parallel processing not available in interactive mode")
            return self._solve_sequential()

        for t in range(self.params.T, 0, -1):
            logger.info(f"Processing time period {t}")
            try:
                with ProcessPoolExecutor() as executor:
                    futures = []
                    for state in self.states:
                        futures.append(
                            executor.submit(self._process_state, state, t)
                        )
                    
                    for state, future in tqdm(zip(self.states, futures), 
                                           total=len(self.states),
                                           desc=f"Processing states for t={t}"):
                        try:
                            result = future.result()
                            # Unpack the result correctly
                            _, max_revenue, optimal_price = result
                            self.V[state][t] = max_revenue
                            self.policy[state][t] = optimal_price
                        except Exception as e:
                            logger.error(f"Error processing state {state}: {str(e)}")
                            
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                logger.info("Falling back to sequential processing for this time period")
                for state in tqdm(self.states, desc=f"Processing states for t={t}"):
                    result = self._process_state(state, t)
                    max_revenue, optimal_price = result[1], result[2]
                    self.V[state][t] = max_revenue
                    self.policy[state][t] = optimal_price
    
    def get_solution_data(self) -> Dict:
        """
        Prepare solution data for visualization.
        
        Returns:
            Dictionary containing all necessary data for visualization
        """
        return {
            'params': self.params,
            'states': self.states,
            'policy': self.policy,
            'value_function': self.V,
            'booking_classes': self.booking_classes,
            'price_vectors': self.price_vectors,
            'arrival_probabilities': self.pi
        }