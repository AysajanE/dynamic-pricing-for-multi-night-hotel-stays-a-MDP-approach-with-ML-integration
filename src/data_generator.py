import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StudyParameters:
    """Class to store and validate study parameters."""
    T: int  # Booking horizon
    N: int  # Service horizon
    C: int  # Room capacity
    price_min: float  # Minimum price
    price_max: float  # Maximum price
    alpha: float  # Smoothing parameter for price acceptance
    beta: float  # Smoothing parameter for capacity
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.T <= 0:
            raise ValueError("Booking horizon T must be positive")
        if self.N <= 0:
            raise ValueError("Service horizon N must be positive")
        if self.C <= 0:
            raise ValueError("Capacity C must be positive")
        if self.price_min >= self.price_max:
            raise ValueError("Minimum price must be less than maximum price")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Smoothing parameters must be positive")
            
class DataGenerator:
    """Generate data for hotel dynamic pricing study."""
    
    def __init__(self, params: StudyParameters, seed: int = None):
        """
        Initialize the data generator.
        
        Args:
            params: StudyParameters object containing study parameters
            seed: Random seed for reproducibility
        """
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.booking_classes = self._generate_booking_classes()
        logger.info(f"Initialized DataGenerator with {len(self.booking_classes)} booking classes")
    
    def _generate_booking_classes(self) -> List[Tuple[int, int]]:
        """
        Generate booking classes (arrival day, departure day pairs) with a maximum
        length of stay of 7 nights.
        
        Returns:
            List of tuples (arrival_day, departure_day)
        """
        MAX_LOS = 7  # Maximum length of stay
        booking_classes = []
        for arrival in range(1, self.params.N + 1):
            for los in range(1, min(MAX_LOS + 1, self.params.N - arrival + 2)):
                departure = arrival + los - 1
                booking_classes.append((arrival, departure))
        
        logger.info(f"Generated {len(booking_classes)} booking classes with max LOS = {MAX_LOS}")
        return booking_classes
    
    def generate_arrival_probabilities(self, demand_scenario: str = 'base') -> Dict[int, Dict[Tuple[int, int], float]]:
        """
        Generate arrival probabilities incorporating different demand scenarios.
        
        Args:
            demand_scenario: Type of demand pattern to generate:
                           'base': Normal demand pattern
                           'high': Increased arrival probabilities
                           'low': Decreased arrival probabilities
                           'peak': Time-dependent demand spikes
                           'fluctuating': Higher demand variability
        """
        if demand_scenario not in ['base', 'high', 'low', 'peak', 'fluctuating']:
            raise ValueError(f"Invalid demand scenario: {demand_scenario}")

        # Define scenario-specific parameters
        scenario_params = {
            'base': {
                'demand_multiplier': 1.0,
                'variability': 0.2,
                'peak_periods': []
            },
            'high': {
                'demand_multiplier': 1.5,
                'variability': 0.2,
                'peak_periods': []
            },
            'low': {
                'demand_multiplier': 0.5,
                'variability': 0.2,
                'peak_periods': []
            },
            'peak': {
                'demand_multiplier': 1.0,
                'variability': 0.2,
                'peak_periods': [(self.params.T // 2, 2.0)]  # (period, multiplier)
            },
            'fluctuating': {
                'demand_multiplier': 1.0,
                'variability': 0.4,
                'peak_periods': []
            }
        }

        # Get scenario parameters
        scenario = scenario_params[demand_scenario]
        
        # Define day-of-week seasonality factors
        dow_factors = {
            0: 0.8,  # Sunday
            1: 0.5,  # Monday
            2: 0.5,  # Tuesday
            3: 0.5,  # Wednesday
            4: 0.6,  # Thursday
            5: 1.0,  # Friday
            6: 1.0   # Saturday
        }
        
        arrival_probs = {}
        for t in range(1, self.params.T + 1):
            base_probs = []
            for arrival, departure in self.booking_classes:
                # Calculate basic factors
                los = departure - arrival + 1
                days_until_arrival = arrival + (self.params.T - t)
                
                # Calculate day-of-week effect
                stay_days = range(arrival, departure + 1)
                stay_dow_factors = [dow_factors[((d-1) % 7)] for d in stay_days]
                avg_dow_factor = np.mean(stay_dow_factors)
                
                # Calculate base probability
                los_factor = np.exp(-0.2 * (los - 1))
                time_factor = np.exp(-0.1 * days_until_arrival)
                base_prob = los_factor * time_factor * avg_dow_factor
                base_probs.append(base_prob)
            
            # Apply scenario adjustments
            base_probs = np.array(base_probs) * scenario['demand_multiplier']
            
            # Apply peak period effects if any
            for peak_t, peak_multiplier in scenario['peak_periods']:
                distance = abs(t - peak_t)
                if distance <= 2:  # Effect spans 5 periods
                    peak_effect = peak_multiplier * (1 - 0.3 * distance)
                    base_probs *= peak_effect
            
            # Add random variation
            raw_probs = base_probs * (1 + self.rng.uniform(
                -scenario['variability'],
                scenario['variability'],
                len(base_probs)
            ))
            
            # Normalize to ensure sum < 1
            total_prob = np.sum(raw_probs)
            if total_prob > 0:
                scale_factor = 0.95 / max(total_prob, 1)  # Ensure sum is at most 0.95
                scaled_probs = raw_probs * scale_factor
            else:
                scaled_probs = raw_probs
            
            arrival_probs[t] = {
                class_: prob for class_, prob in zip(self.booking_classes, scaled_probs)
            }

        # Log scenario statistics
        self._log_arrival_probability_stats(arrival_probs, demand_scenario)
        return arrival_probs
    
    def _log_arrival_probability_stats(self, arrival_probs: Dict, demand_scenario: str):
        """Log detailed statistics for arrival probabilities."""
        daily_arrival_probs = [sum(probs.values()) for probs in arrival_probs.values()]
        logger.info(f"\nDemand Scenario: {demand_scenario}")
        logger.info(f"Average daily arrival probability: {np.mean(daily_arrival_probs):.3f}")
        logger.info(f"Maximum daily arrival probability: {np.max(daily_arrival_probs):.3f}")
        logger.info(f"Minimum daily arrival probability: {np.min(daily_arrival_probs):.3f}")
        
        # Log day-of-week statistics
        dow_probs = {dow: [] for dow in range(7)}
        for t, probs in arrival_probs.items():
            for (arrival, departure), prob in probs.items():
                dow = (arrival - 1) % 7
                dow_probs[dow].append(prob)
        
        logger.info("\nAverage arrival probabilities by day of week:")
        for dow in range(7):
            day_name = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][dow]
            avg_prob = np.mean(dow_probs[dow]) if dow_probs[dow] else 0
            logger.info(f"{day_name}: {avg_prob:.3f}")

    def generate_reservation_price_params(self, market_condition: str = 'standard') -> Dict[Tuple[int, int], float]:
        """
        Generate epsilon parameters for the linear reservation price survival function.
        F̄_b(p_t^b) = 1 - ε_b * p_t^b for each booking class.
        """
        if market_condition not in ['standard', 'luxury', 'budget', 'peak_season', 'competitive']:
            raise ValueError(f"Invalid market condition: {market_condition}")

        # Calculate maximum allowable epsilon to ensure F̄_b(p_max) ≥ 0
        max_epsilon = 1 / self.params.price_max

        # Define market-specific parameters
        market_params = {
            'standard': {
                'min_epsilon_factor': 0.5,    # 1/(2*price_max)
                'base_sensitivity': 0.5,      # Start at 50% of range
                'los_factor': 0.10,          # 10% increase per night
                'weekend_premium': 0.20,      # 20% higher for weekend stays
                'random_variation': 0.10      # ±10% random variation
            },
            'luxury': {
                'min_epsilon_factor': 0.25,   # 1/(4*price_max)
                'base_sensitivity': 0.4,      # Start at 40% of range
                'los_factor': 0.05,          # 5% increase per night
                'weekend_premium': 0.10,      # 10% higher for weekend stays
                'random_variation': 0.08      # ±8% random variation
            },
            'budget': {
                'min_epsilon_factor': 0.75,   # 3/(4*price_max)
                'base_sensitivity': 0.6,      # Start at 60% of range
                'los_factor': 0.15,          # 15% increase per night
                'weekend_premium': 0.30,      # 30% higher for weekend stays
                'random_variation': 0.12      # ±12% random variation
            },
            'peak_season': {
                'min_epsilon_factor': 0.4,    # 2/(5*price_max)
                'base_sensitivity': 0.3,      # Start at 30% of range
                'los_factor': 0.08,          # 8% increase per night
                'weekend_premium': 0.10,      # 10% higher for weekend stays
                'random_variation': 0.10      # ±10% random variation
            },
            'competitive': {
                'min_epsilon_factor': 0.6,    # 3/(5*price_max)
                'base_sensitivity': 0.7,      # Start at 70% of range
                'los_factor': 0.12,          # 12% increase per night
                'weekend_premium': 0.25,      # 25% higher for weekend stays
                'random_variation': 0.15      # ±15% random variation
            }
        }

        params = market_params[market_condition]
        min_epsilon = max_epsilon * params['min_epsilon_factor']
        
        epsilons = {}
        for arrival, departure in self.booking_classes:
            # Calculate length of stay
            los = departure - arrival + 1
            
            # Calculate weekend ratio
            stay_days = range(arrival, departure + 1)
            weekend_days = sum(1 for d in stay_days if (d-1) % 7 in [5, 6])
            weekend_ratio = weekend_days / los
            
            # Calculate base epsilon
            base_epsilon = min_epsilon + (max_epsilon - min_epsilon) * (
                params['base_sensitivity'] +
                params['los_factor'] * (los - 1) +
                params['weekend_premium'] * weekend_ratio
            )
            
            # Add random variation
            epsilon = base_epsilon * (1 + self.rng.uniform(
                -params['random_variation'],
                params['random_variation']
            ))
            
            # Ensure epsilon stays within valid bounds
            epsilon = min(max_epsilon, max(min_epsilon, epsilon))
            epsilons[(arrival, departure)] = epsilon
        
        # Log statistics
        self._log_reservation_price_stats(epsilons, market_condition, params)
        return epsilons
    
    def _log_reservation_price_stats(self, epsilons: Dict[Tuple[int, int], float], market_condition: str, market_params: dict):
        """Log detailed statistics for reservation price parameters."""
        logger.info(f"\nReservation Price Parameters for {market_condition.upper()} Market:")
        logger.info(f"Market Characteristics:")
        logger.info(f"- Base Price Sensitivity: {market_params['base_sensitivity']:.2f}")
        logger.info(f"- Length of Stay Factor: {market_params['los_factor']:.3f}")
        logger.info(f"- Weekend Premium: {market_params['weekend_premium']:.2f}")
        logger.info(f"- Random Variation: ±{market_params['random_variation']:.2f}")
        
        logger.info("\nEpsilon Statistics:")
        logger.info(f"Average epsilon: {np.mean(list(epsilons.values())):.6f}")
        logger.info(f"Maximum epsilon: {np.max(list(epsilons.values())):.6f}")
        logger.info(f"Minimum epsilon: {np.min(list(epsilons.values())):.6f}")
        
        # Log average epsilon by length of stay
        los_epsilons = {}
        for (arrival, departure), epsilon in epsilons.items():
            los = departure - arrival + 1
            if los not in los_epsilons:
                los_epsilons[los] = []
            los_epsilons[los].append(epsilon)
        
        logger.info("\nAverage epsilon by length of stay:")
        for los in sorted(los_epsilons.keys()):
            avg_epsilon = np.mean(los_epsilons[los])
            logger.info(f"LOS {los}: {avg_epsilon:.6f}")
            
        # Log average epsilon by day of week
        dow_epsilons = {i: [] for i in range(7)}
        for (arrival, departure), epsilon in epsilons.items():
            stay_days = range(arrival, departure + 1)
            for day in stay_days:
                dow = (day - 1) % 7
                dow_epsilons[dow].append(epsilon)
        
        logger.info("\nAverage epsilon by day of week:")
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for dow in range(7):
            avg_epsilon = np.mean(dow_epsilons[dow]) if dow_epsilons[dow] else 0
            logger.info(f"{days[dow]}: {avg_epsilon:.6f}")
            
    def generate_initial_prices(
        self, 
        market_condition: str = 'standard',
        initialization_strategy: str = 'market_based',
        strategy_params: Dict = None
        ) -> Dict[int, Dict[int, float]]:
        """
        Generate initial prices using various initialization strategies to study SAA convergence.

        Initialization Strategies:
        1. Market-based: Prices based on market conditions and industry patterns
        2. Uniform: Constant price across all periods and days
        3. Random: Completely randomized prices within bounds
        4. Extreme: All prices set to either minimum or maximum bounds
        5. Gradient-based: Smooth price transitions based on temporal patterns
        6. Demand-driven: Prices initialized based on arrival probabilities

        Args:
            market_condition: Type of market condition to consider
            initialization_strategy: Strategy to use for price initialization
            strategy_params: Additional parameters for specific strategies
                - 'uniform': {'price_level': float} - Single price level to use
                - 'extreme': {'high': bool} - True for max prices, False for min
                - 'gradient': {'slope': float} - Steepness of price changes

        Returns:
            Dictionary mapping time period to {day: price}
        """
        if strategy_params is None:
            strategy_params = {}

        # Define initialization strategies
        if initialization_strategy == 'market_based':
            return self._generate_market_based_prices(market_condition)

        elif initialization_strategy == 'uniform':
            price_level = strategy_params.get('price_level', 0.5)
            price = self.params.price_min + price_level * (self.params.price_max - self.params.price_min)
            return {t: {day: price for day in range(1, self.params.N + 1)}
                    for t in range(1, self.params.T + 1)}

        elif initialization_strategy == 'random':
            return {t: {day: self.rng.uniform(self.params.price_min, self.params.price_max)
                        for day in range(1, self.params.N + 1)}
                    for t in range(1, self.params.T + 1)}

        elif initialization_strategy == 'extreme':
            price = self.params.price_max if strategy_params.get('high', True) else self.params.price_min
            return {t: {day: price for day in range(1, self.params.N + 1)}
                    for t in range(1, self.params.T + 1)}

        elif initialization_strategy == 'gradient':
            slope = strategy_params.get('slope', 0.1)
            prices = {}
            for t in range(1, self.params.T + 1):
                daily_prices = {}
                for day in range(1, self.params.N + 1):
                    # Create smooth price transitions based on time and day
                    progress = (t + day) / (self.params.T + self.params.N)
                    base_price = self.params.price_min + progress * (self.params.price_max - self.params.price_min)
                    daily_prices[day] = base_price + slope * self.rng.normal(0, 1)
                    # Ensure prices stay within bounds
                    daily_prices[day] = min(max(daily_prices[day], self.params.price_min), self.params.price_max)
                prices[t] = daily_prices
            return prices

        elif initialization_strategy == 'demand_driven':
            # Get arrival probabilities for initialization
            arrival_probs = self.generate_arrival_probabilities(demand_scenario='base')
            prices = {}
            for t in range(1, self.params.T + 1):
                daily_prices = {}
                for day in range(1, self.params.N + 1):
                    # Calculate average arrival probability for this day
                    day_probs = [prob for (arr, dep), prob in arrival_probs[t].items() 
                               if arr <= day <= dep]
                    avg_prob = np.mean(day_probs) if day_probs else 0.5
                    # Higher prices for higher demand periods
                    price_level = 0.3 + 0.7 * avg_prob  # Maps [0,1] to [0.3,1] range
                    daily_prices[day] = self.params.price_min + price_level * (self.params.price_max - self.params.price_min)
                prices[t] = daily_prices
            return prices

        else:
            raise ValueError(f"Unknown initialization strategy: {initialization_strategy}")

    def _generate_market_based_prices(self, market_condition: str) -> Dict[int, Dict[int, float]]:
        """
        Generate thoughtfully structured initial prices for the SAA algorithm.

        This function generates initial prices considering multiple factors:
        1. Day-of-week patterns (weekend vs. weekday pricing)
        2. Market conditions (luxury vs. budget segments)
        3. Advance booking effects (prices typically decrease closer to stay dates)
        4. Capacity utilization targets

        Args:
            market_condition: Type of market condition to consider for base price levels

        Returns:
            Dictionary mapping time period to {day: price}
        """
        # Define base price levels by market condition
        market_base_prices = {
            'standard': 0.5,     # 50% of price range
            'luxury': 0.7,       # 70% of price range
            'budget': 0.3,       # 30% of price range
            'peak_season': 0.8,  # 80% of price range
            'competitive': 0.4   # 40% of price range
        }

        # Define day-of-week factors (relative to base price)
        dow_factors = {
            0: 1.1,    # Sunday
            1: 0.8,    # Monday
            2: 0.8,    # Tuesday
            3: 0.8,    # Wednesday
            4: 0.9,    # Thursday
            5: 1.2,    # Friday
            6: 1.2     # Saturday
        }

        # Calculate base price for the given market condition
        price_range = self.params.price_max - self.params.price_min
        base_price = self.params.price_min + price_range * market_base_prices.get(market_condition, 0.5)

        prices = {}
        for t in range(1, self.params.T + 1):
            # Calculate advance booking factor (prices slightly higher for early bookings)
            advance_factor = 1.0 + 0.2 * (1 - t / self.params.T)

            daily_prices = {}
            for day in range(1, self.params.N + 1):
                # Get day of week factor
                dow = (day - 1) % 7
                dow_factor = dow_factors[dow]

                # Calculate initial price with controlled randomness
                base = base_price * dow_factor * advance_factor
                random_factor = 1.0 + self.rng.uniform(-0.1, 0.1)  # ±10% random variation
                price = base * random_factor

                # Ensure price stays within bounds
                price = min(max(price, self.params.price_min), self.params.price_max)
                daily_prices[day] = price

            prices[t] = daily_prices

        # Log statistics about generated prices
        self._log_initial_price_stats(prices, market_condition)
        return prices

    def analyze_convergence_patterns(
            self,
            market_condition: str,
            initialization_strategies: List[str],
            num_replications: int = 30
        ) -> pd.DataFrame:
        """
        Analyze SAA convergence patterns for different initialization strategies.

        Args:
            market_condition: Market condition to analyze
            initialization_strategies: List of strategies to test
            num_replications: Number of replications per strategy

        Returns:
            DataFrame containing convergence analysis results
        """
        results = []
        for strategy in initialization_strategies:
            for rep in range(num_replications):
                initial_prices = self.generate_initial_prices(
                    market_condition=market_condition,
                    initialization_strategy=strategy
                )

                # Record key statistics about initial prices
                prices_array = np.array([list(p.values()) for p in initial_prices.values()])
                results.append({
                    'strategy': strategy,
                    'replication': rep,
                    'initial_mean': np.mean(prices_array),
                    'initial_std': np.std(prices_array),
                    'price_range': np.max(prices_array) - np.min(prices_array),
                    'temporal_variation': np.std([np.mean(list(p.values())) 
                                               for p in initial_prices.values()]),
                    'spatial_variation': np.mean([np.std(list(p.values())) 
                                               for p in initial_prices.values()])
                })

        return pd.DataFrame(results)
    
#     def _log_initial_price_stats(
#         self, 
#         prices: Dict[int, Dict[int, float]], 
#         market_condition: str,
#         initialization_strategy: str = 'market_based',
#         strategy_params: Dict = None
#         ):
#         """
#         Log detailed statistics for initial prices.

#         Args:
#             prices: Dictionary mapping booking periods to daily prices
#             market_condition: Type of market segment
#             initialization_strategy: Method used for price initialization
#             strategy_params: Additional parameters used in initialization
#         """
#         logger.info(f"\nInitial Price Generation Analysis for {market_condition.upper()} Market")
#         logger.info(f"Initialization Strategy: {initialization_strategy.upper()}")

#         if strategy_params:
#             logger.info("Strategy-Specific Parameters:")
#             for param, value in strategy_params.items():
#                 logger.info(f"- {param}: {value}")

#         # Calculate overall price statistics
#         all_prices = list(np.concatenate([list(period_prices.values()) 
#                                         for period_prices in prices.values()]))

#         logger.info("\nOverall Price Statistics:")
#         logger.info(f"Average Price: ${np.mean(all_prices):.2f}")
#         logger.info(f"Median Price: ${np.median(all_prices):.2f}")
#         logger.info(f"Price Range: ${np.min(all_prices):.2f} - ${np.max(all_prices):.2f}")
#         logger.info(f"Standard Deviation: ${np.std(all_prices):.2f}")

#         # Analyze day-of-week patterns
#         dow_prices = {i: [] for i in range(7)}
#         for period in prices.values():
#             for day, price in period.items():
#                 dow = (day - 1) % 7
#                 dow_prices[dow].append(price)

#         logger.info("\nDay-of-Week Price Analysis:")
#         days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
#         for dow, day_name in enumerate(days):
#             avg_price = np.mean(dow_prices[dow])
#             std_price = np.std(dow_prices[dow])
#             logger.info(f"{day_name}: ${avg_price:.2f} (±${std_price:.2f})")

#         # Analyze price trends across booking horizon
#         period_averages = {t: np.mean(list(prices[t].values())) for t in prices}
#         logger.info("\nBooking Horizon Price Trends:")
#         logger.info(f"Early Period Average: ${np.mean(list(period_averages.values())[:len(period_averages)//3]):.2f}")
#         logger.info(f"Middle Period Average: ${np.mean(list(period_averages.values())[len(period_averages)//3:2*len(period_averages)//3]):.2f}")
#         logger.info(f"Late Period Average: ${np.mean(list(period_averages.values())[2*len(period_averages)//3:]):.2f}")
    
    def _log_initial_price_stats(
            self, 
            prices: Dict[int, Dict[int, float]], 
            market_condition: str,
            initialization_strategy: str = 'market_based',
            strategy_params: Dict = None
            ):
        """
        Log detailed statistics for initial prices with proper handling of multi-week horizons.
        """
        logger.info(f"\nInitial Price Generation Analysis for {market_condition.upper()} Market")
        logger.info(f"Initialization Strategy: {initialization_strategy.upper()}")

        if strategy_params:
            logger.info("Strategy-Specific Parameters:")
            for param, value in strategy_params.items():
                logger.info(f"- {param}: {value}")

        # Calculate overall price statistics
        all_prices = []
        for period_prices in prices.values():
            for day in range(1, self.params.N + 1):  # Explicitly iterate through valid days
                if day in period_prices:
                    all_prices.append(period_prices[day])

        if all_prices:  # Ensure we have prices to analyze
            logger.info("\nOverall Price Statistics:")
            logger.info(f"Average Price: ${np.mean(all_prices):.2f}")
            logger.info(f"Median Price: ${np.median(all_prices):.2f}")
            logger.info(f"Price Range: ${np.min(all_prices):.2f} - ${np.max(all_prices):.2f}")
            logger.info(f"Standard Deviation: ${np.std(all_prices):.2f}")

            # Initialize day-of-week price collections
            dow_prices = {i: [] for i in range(7)}

            # Collect prices by day of week
            for period in prices.values():
                for day in range(1, self.params.N + 1):
                    if day in period:
                        # Map the day number to day of week (0-6)
                        dow = ((day - 1) % 7)  # Ensure proper modulo arithmetic
                        dow_prices[dow].append(float(period[day]))  # Explicit float conversion

            # Report day-of-week statistics
            logger.info("\nDay-of-Week Price Analysis:")
            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            for dow, day_name in enumerate(days):
                if dow_prices[dow]:  # Check if we have prices for this day
                    avg_price = np.mean(dow_prices[dow])
                    std_price = np.std(dow_prices[dow])
                    logger.info(f"{day_name}: ${avg_price:.2f} (±${std_price:.2f})")
                else:
                    logger.info(f"{day_name}: $0.00 (±$0.00)")  # Handle empty cases explicitly

            # Analyze booking horizon trends
            valid_period_prices = {}
            for t in prices:
                period_values = [float(v) for v in prices[t].values() if v is not None]
                if period_values:
                    valid_period_prices[t] = np.mean(period_values)

            if valid_period_prices:
                num_periods = len(valid_period_prices)
                period_values = list(valid_period_prices.values())

                # Split into thirds for trend analysis
                early = period_values[:num_periods//3]
                middle = period_values[num_periods//3:2*num_periods//3]
                late = period_values[2*num_periods//3:]

                logger.info("\nBooking Horizon Price Trends:")
                if early:
                    logger.info(f"Early Period Average: ${np.mean(early):.2f}")
                if middle:
                    logger.info(f"Middle Period Average: ${np.mean(middle):.2f}")
                if late:
                    logger.info(f"Late Period Average: ${np.mean(late):.2f}")
        else:
            logger.warning("No valid prices found for analysis")
            
    def generate_study_instance(
            self, 
            demand_scenario: str = 'base', 
            market_condition: str = 'standard',
            price_initialization: Dict = None
        ) -> Dict:
        """
        Generate a complete instance for the hotel dynamic pricing computational study.

        This function creates a comprehensive study instance by combining basic parameters,
        booking classes, demand patterns, and price sensitivity characteristics. The function
        now supports sophisticated price initialization strategies for studying algorithm
        convergence patterns.

        Args:
            demand_scenario: Type of demand pattern to generate. Options include:
                - 'base': Standard demand patterns
                - 'high': Increased arrival probabilities (1.5x base)
                - 'low': Decreased arrival probabilities (0.5x base)
                - 'peak': Time-dependent demand spikes
                - 'fluctuating': Higher demand variability

            market_condition: Type of market condition to model. Options include:
                - 'standard': Balanced price sensitivity
                - 'luxury': Lower price sensitivity
                - 'budget': Higher price sensitivity
                - 'peak_season': Temporarily reduced sensitivity
                - 'competitive': Market-driven high sensitivity

            price_initialization: Optional dictionary containing price initialization parameters:
                - 'strategy': Initialization strategy to use
                    Options: 'market_based', 'uniform', 'random', 'extreme', 
                            'gradient', 'demand_driven'
                - 'params': Additional parameters specific to the chosen strategy
                Default: {'strategy': 'market_based', 'params': None}

        Returns:
            Dictionary containing:
            - parameters: StudyParameters object with basic configuration
            - booking_classes: List of (arrival, departure) tuples
            - arrival_probabilities: Dict mapping periods to booking class probabilities
            - reservation_price_params: Dict mapping booking classes to epsilon values
            - initial_prices: Dict mapping periods to daily prices
            - scenario_info: Dict containing scenario configuration details
            - generation_timestamp: Creation time of the instance
            - initialization_info: Dict containing price initialization details
        """
        # Set default price initialization if not provided
        if price_initialization is None:
            price_initialization = {
                'strategy': 'market_based',
                'params': None
            }

        # Generate initial prices using specified strategy
        initial_prices = self.generate_initial_prices(
            market_condition=market_condition,
            initialization_strategy=price_initialization['strategy'],
            strategy_params=price_initialization['params']
        )

        # Create the study instance
        instance = {
            'parameters': self.params,
            'booking_classes': self.booking_classes,
            'arrival_probabilities': self.generate_arrival_probabilities(demand_scenario),
            'reservation_price_params': self.generate_reservation_price_params(market_condition),
            'initial_prices': initial_prices,
            'scenario_info': {
                'demand_scenario': demand_scenario,
                'market_condition': market_condition,
                'description': f"{market_condition.capitalize()} market with {demand_scenario} demand"
            },
            'initialization_info': {
                'strategy': price_initialization['strategy'],
                'params': price_initialization['params'],
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        # Log comprehensive instance information
        logger.info("\nStudy Instance Generation Summary")
        logger.info("================================")
        logger.info(f"Demand Scenario: {demand_scenario}")
        logger.info(f"Market Condition: {market_condition}")
        logger.info(f"Price Initialization Strategy: {price_initialization['strategy']}")

        logger.info("\nInstance Dimensions:")
        logger.info(f"Booking Horizon (T): {self.params.T} periods")
        logger.info(f"Service Horizon (N): {self.params.N} days")
        logger.info(f"Room Capacity (C): {self.params.C} rooms")
        logger.info(f"Number of Booking Classes: {len(instance['booking_classes'])}")

        if price_initialization['params']:
            logger.info("\nInitialization Parameters:")
            for param, value in price_initialization['params'].items():
                logger.info(f"- {param}: {value}")

        return instance
    
class TestConfiguration:
    """
    Configuration system for hotel dynamic pricing test instances.
    
    This class generates test parameters based on meaningful business and operational
    considerations rather than arbitrary values. It ensures that service horizons align
    with weekly patterns and that prices reflect realistic market relationships.
    """
    
    def __init__(self):
        # Base parameters for standard mid-scale hotel
        self.base_price = 200.0  # Standard nightly rate
        self.base_capacity = 100  # Standard hotel room capacity
        
        # Price multipliers for different market segments
        self.price_multipliers = {
            'standard': {'min': 0.6, 'max': 1.8},   # $120 - $360
            'luxury': {'min': 0.8, 'max': 2.5},     # $160 - $500
            'budget': {'min': 0.4, 'max': 1.4},     # $80 - $280
            'peak_season': {'min': 0.7, 'max': 2.2}, # $140 - $440
            'competitive': {'min': 0.5, 'max': 1.6}  # $100 - $320
        }
        
        # Service horizon options (in weeks)
        self.horizon_weeks = {
            'minimal': 1,    # One week
            'standard': 2,   # Two weeks
            'extended': 4,   # Four weeks
            'seasonal': 13   # Quarter/Season
        }
        
        # Booking period discretization (number of periods per day)
        self.periods_per_day = {
            'coarse': 2,     # Two periods per day
            'standard': 4,   # Four periods per day
            'fine': 8        # Eight periods per day
        }

    def get_config(
            self,
            test_type: str,
            market_condition: str = 'standard',
            discretization: str = 'standard',
            capacity_scale: float = 1.0
        ) -> dict:
        """
        Generate test configuration parameters based on specified conditions.
        
        Args:
            test_type: Type of test instance ('minimal', 'standard', 'extended', 'seasonal')
            market_condition: Market segment for pricing ('standard', 'luxury', etc.)
            discretization: Granularity of booking periods ('coarse', 'standard', 'fine')
            capacity_scale: Scaling factor for hotel capacity (default 1.0)
            
        Returns:
            Dictionary containing all necessary test parameters
        """
        # Calculate service horizon in days
        N = self.horizon_weeks[test_type] * 7
        
        # Calculate number of booking periods
        # For standard case: 4 weeks advance booking with 4 periods per day
        advance_booking_weeks = min(4, self.horizon_weeks[test_type] * 2)
        T = advance_booking_weeks * 7 * self.periods_per_day[discretization]
        
        # Calculate price bounds based on market condition
        price_mult = self.price_multipliers[market_condition]
        price_min = self.base_price * price_mult['min']
        price_max = self.base_price * price_mult['max']
        
        # Calculate capacity
        C = int(self.base_capacity * capacity_scale)
        
        return {
            'T': T,
            'N': N,
            'C': C,
            'price_min': price_min,
            'price_max': price_max,
            'alpha': 0.1,  # Smoothing parameters could also be configuration-dependent
            'beta': 0.1,
            'test_type': test_type,
            'market_condition': market_condition,
            'discretization': discretization,
            'periods_per_day': self.periods_per_day[discretization]
        }
    
def create_test_instance(
        demand_scenario: str = 'base', 
        market_condition: str = 'standard',
        test_configuration: Dict = None,
        price_initialization: Dict = None,
        seed: int = 42
    ) -> Dict:
    """
    Create a test instance for the hotel dynamic pricing computational study.
    
    Args:
        demand_scenario: Type of demand pattern to generate
        market_condition: Type of market segment to model
        test_configuration: Configuration parameters from TestConfiguration
        price_initialization: Optional price initialization strategy
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing complete test instance data
    """
    if test_configuration is None:
        # If no configuration provided, use default from TestConfiguration
        config = TestConfiguration()
        test_configuration = config.get_config(
            test_type='standard',
            market_condition=market_condition,
            discretization='standard',
            capacity_scale=1.0
        )
    
    # Create parameters from configuration
    params = StudyParameters(
        T=test_configuration['T'],
        N=test_configuration['N'],
        C=test_configuration['C'],
        price_min=test_configuration['price_min'],
        price_max=test_configuration['price_max'],
        alpha=test_configuration['alpha'],
        beta=test_configuration['beta']
    )
    
    # Generate test instance
    generator = DataGenerator(params, seed=seed)
    test_instance = generator.generate_study_instance(
        demand_scenario=demand_scenario,
        market_condition=market_condition,
        price_initialization=price_initialization
    )
    
    # Add test configuration information
    test_instance['test_config'] = {
        'parameters': test_configuration,
        'discretization': test_configuration.get('discretization', 'standard'),
        'periods_per_day': test_configuration.get('periods_per_day', 4),
        'seed': seed,
        'generation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return test_instance

if __name__ == "__main__":
    print("\nHotel Dynamic Pricing Study - Test Instance Analysis")
    print("=" * 60)
    
    # Initialize configuration system
    config = TestConfiguration()
    
    # Generate test instances for different scenarios
    test_cases = [
        {
            'test_type': 'standard',
            'market_condition': 'standard',
            'demand_scenario': 'base',
            'discretization': 'standard',
            'capacity_scale': 1.0
        },
        {
            'test_type': 'minimal',
            'market_condition': 'luxury',
            'demand_scenario': 'high',
            'discretization': 'fine',
            'capacity_scale': 0.5
        }
    ]
    
    for case in test_cases:
        print(f"\nGenerating Test Instance: {case['test_type'].upper()} Configuration")
        print("-" * 60)
        
        # Get configuration parameters
        params = config.get_config(
            test_type=case['test_type'],
            market_condition=case['market_condition'],
            discretization=case['discretization'],
            capacity_scale=case['capacity_scale']
        )
        
        # Create test instance
        test_instance = create_test_instance(
            demand_scenario=case['demand_scenario'],
            market_condition=case['market_condition'],
            test_configuration=params
        )
        
        print("\nConfiguration Parameters")
        print(f"Service Horizon: {params['N']} days ({params['N']//7} weeks)")
        print(f"Booking Horizon: {params['T']} periods ({params['T']/params['periods_per_day']:.1f} days)")
        print(f"Time Discretization: {params['periods_per_day']} periods per day")
        print(f"Room Capacity: {params['C']} rooms")
        print(f"Price Range: ${params['price_min']:.2f} - ${params['price_max']:.2f}")
        
        print("\nBooking Class Analysis")
        booking_classes = test_instance['booking_classes']
        los_distribution = {}
        for arrival, departure in booking_classes:
            los = departure - arrival + 1
            los_distribution[los] = los_distribution.get(los, 0) + 1
        
        print(f"Total Booking Classes: {len(booking_classes)}")
        print("Length of Stay Distribution:")
        total_classes = len(booking_classes)
        for los, count in sorted(los_distribution.items()):
            percentage = (count / total_classes) * 100
            print(f"  {los} night(s): {count} classes ({percentage:.1f}%)")
        
        print("\nArrival Pattern Analysis")
        arrival_probs = test_instance['arrival_probabilities']
        daily_probs = []
        
        # Analyze arrival probabilities by day
        for t in range(1, params['T'] + 1):
            total_prob = sum(arrival_probs[t].values())
            daily_probs.append(total_prob)
        
        print(f"Average Daily Arrival Probability: {np.mean(daily_probs):.3f}")
        print(f"Peak Arrival Probability: {np.max(daily_probs):.3f}")
        print(f"Minimum Arrival Probability: {np.min(daily_probs):.3f}")
        
        # Analyze arrival patterns by time of day
        if params['periods_per_day'] > 1:
            periods_analysis = []
            for period in range(params['periods_per_day']):
                period_probs = [daily_probs[i] for i in range(len(daily_probs)) 
                              if i % params['periods_per_day'] == period]
                periods_analysis.append(np.mean(period_probs))
            
            print("\nArrival Patterns by Time of Day:")
            for period, avg_prob in enumerate(periods_analysis):
                print(f"  Period {period + 1}: {avg_prob:.3f}")
        
        print("\nPrice Sensitivity Analysis")
        eps_params = test_instance['reservation_price_params']
        epsilon_values = list(eps_params.values())
        
        print(f"Epsilon Statistics:")
        print(f"  Average: {np.mean(epsilon_values):.3f}")
        print(f"  Range: [{np.min(epsilon_values):.3f}, {np.max(epsilon_values):.3f}]")
        print(f"  Standard Deviation: {np.std(epsilon_values):.3f}")
        
        # Analyze pricing by day of week
        print("\nInitial Price Analysis by Day of Week")
        initial_prices = test_instance['initial_prices']
        dow_prices = {i: [] for i in range(7)}
        
        for t in initial_prices:
            for day, price in initial_prices[t].items():
                dow = (day - 1) % 7
                dow_prices[dow].append(price)
        
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for dow, day_name in enumerate(days):
            avg_price = np.mean(dow_prices[dow])
            std_price = np.std(dow_prices[dow])
            print(f"  {day_name}: ${avg_price:.2f} (±${std_price:.2f})")
        
        print("\nInstance Metadata")
        print(f"Market Condition: {case['market_condition']}")
        print(f"Demand Scenario: {case['demand_scenario']}")
        print(f"Generated at: {test_instance['test_config']['generation_timestamp']}")
        print("\n" + "=" * 60)