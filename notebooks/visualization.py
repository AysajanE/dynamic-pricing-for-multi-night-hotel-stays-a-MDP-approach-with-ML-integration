# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, List, Any
import random

logger = logging.getLogger(__name__)

class HotelPricingVisualizer:
    """
    A separate class for visualizing hotel pricing model results.
    """
    def __init__(self, solution_data: Dict):
        """
        Initialize visualizer with solution data.
        
        Args:
            solution_data: Dictionary containing model solution data
        """
        self.data = solution_data
        self.params = solution_data['params']
        self.setup_style()
        
    def setup_style(self):
        """Configure visualization style."""
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
    def _plot_pricing_policy_3d(self):
        """
        Plot 3D visualization of pricing policies for different states.
        Shows how optimal prices vary with available capacity for each day.
        """
        try:
            # Create subplots for each time period
            for t in range(1, self.params.T + 1):
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Prepare data
                x1, x2, x3, prices = [], [], [], []
                for state in self.states:
                    if self.policy[state][t] is not None:  # Check if policy exists
                        x1.append(state[0])  # Day 1 capacity
                        x2.append(state[1])  # Day 2 capacity
                        x3.append(state[2])  # Day 3 capacity
                        avg_price = np.mean(self.policy[state][t])
                        prices.append(avg_price)

                # Convert to numpy arrays for better compatibility
                x1 = np.array(x1)
                x2 = np.array(x2)
                x3 = np.array(x3)
                prices = np.array(prices)

                # Check if we have data to plot
                if len(prices) == 0:
                    logger.warning(f"No data to plot for time period {t}")
                    continue

                # Create color normalization
                norm = plt.Normalize(prices.min(), prices.max())
                colors = plt.cm.viridis(norm(prices))

                # Create scatter plot with explicit sizes
                scatter = ax.scatter(x1, x2, x3, 
                                   c=colors,
                                   s=50,  # explicit point size
                                   alpha=0.6)

                # Labels and title
                ax.set_xlabel('Available Rooms Day 1')
                ax.set_ylabel('Available Rooms Day 2')
                ax.set_zlabel('Available Rooms Day 3')
                ax.set_title(f'Optimal Average Pricing Policy (t={t})')

                # Create custom colorbar
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm)
                cbar.set_label('Average Price ($)')

                # Optional: Add grid
                ax.grid(True)

                # Optional: Set viewing angle
                ax.view_init(elev=20, azim=45)

                # Optional: Set axis limits
                ax.set_xlim(0, self.params.C)
                ax.set_ylim(0, self.params.C)
                ax.set_zlim(0, self.params.C)

                plt.tight_layout()

        except Exception as e:
            logger.error(f"Error in 3D plotting: {str(e)}")
            # Fallback to 2D visualization
            self._plot_pricing_policy_2d(t)

    def _plot_pricing_policy_2d(self, t):
        """
        Fallback 2D visualization of pricing policies.
        Creates a series of 2D plots for different capacity levels of day 3.
        """
        # Create a grid of subplots for different day 3 capacities
        n_plots = self.params.C + 1
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5*n_cols, 4*n_rows),
                                squeeze=False)
        fig.suptitle(f'Pricing Policy Slices (t={t})')

        for day3_capacity in range(self.params.C + 1):
            row = day3_capacity // n_cols
            col = day3_capacity % n_cols
            ax = axes[row, col]

            # Prepare data for this slice
            x1, x2, prices = [], [], []
            for state in self.states:
                if (self.policy[state][t] is not None and 
                    state[2] == day3_capacity):
                    x1.append(state[0])
                    x2.append(state[1])
                    prices.append(np.mean(self.policy[state][t]))

            if len(prices) > 0:
                # Create heatmap for this slice
                x1_unique = sorted(set(x1))
                x2_unique = sorted(set(x2))
                price_grid = np.full((len(x1_unique), len(x2_unique)), np.nan)

                for i, j, p in zip(x1, x2, prices):
                    i_idx = x1_unique.index(i)
                    j_idx = x2_unique.index(j)
                    price_grid[i_idx, j_idx] = p

                im = ax.imshow(price_grid.T, origin='lower', 
                              aspect='equal', cmap='viridis')

                # Labels
                ax.set_xlabel('Day 1 Capacity')
                ax.set_ylabel('Day 2 Capacity')
                ax.set_title(f'Day 3 Capacity = {day3_capacity}')

                # Add colorbar
                plt.colorbar(im, ax=ax)

            else:
                ax.text(0.5, 0.5, 'No Data',
                       horizontalalignment='center',
                       verticalalignment='center')

            # Set ticks
            ax.set_xticks(range(len(x1_unique)))
            ax.set_yticks(range(len(x2_unique)))
            ax.set_xticklabels(x1_unique)
            ax.set_yticklabels(x2_unique)

        # Remove empty subplots if any
        for idx in range(day3_capacity + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
    
    def _plot_value_function(self):
        """
        Visualize the value function across different states and time periods.
        Shows expected revenue potential for different states.
        """
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Value function vs total capacity
        plt.subplot(131)
        for t in range(1, self.params.T + 1):
            x = [sum(state) for state in self.states]
            y = [self.V[state][t] for state in self.states]
            plt.scatter(x, y, alpha=0.5, label=f't={t}')
        
        plt.xlabel('Total Available Rooms')
        plt.ylabel('Expected Revenue ($)')
        plt.title('Value Function vs. Total Capacity')
        plt.legend()
        
        # Plot 2: Value function heatmap for t=1
        plt.subplot(132)
        state_values_t1 = np.zeros((self.params.C + 1, self.params.C + 1))
        for state in self.states:
            if state[2] == self.params.C // 2:  # Fix third dimension
                state_values_t1[state[0], state[1]] = self.V[state][1]
        
        plt.imshow(state_values_t1, cmap='viridis', origin='lower')
        plt.colorbar(label='Expected Revenue ($)')
        plt.xlabel('Day 1 Capacity')
        plt.ylabel('Day 2 Capacity')
        plt.title('Value Function Heatmap (t=1, Day 3 fixed)')
        
        # Plot 3: Value function change over time
        plt.subplot(133)
        for state in random.sample(self.states, min(10, len(self.states))):
            values = [self.V[state][t] for t in range(1, self.params.T + 1)]
            plt.plot(range(1, self.params.T + 1), values, 
                    marker='o', label=f'State {state}')
        
        plt.xlabel('Time Period')
        plt.ylabel('Expected Revenue ($)')
        plt.title('Value Function Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_price_distributions(self):
        """
        Analyze and visualize the distribution of optimal prices.
        Shows pricing patterns and relationships.
        """
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Price distribution histogram
        plt.subplot(131)
        all_prices = []
        for t in range(1, self.params.T + 1):
            prices = [np.mean(self.policy[state][t]) 
                     for state in self.states 
                     if self.policy[state][t] is not None]
            all_prices.extend(prices)
        
        plt.hist(all_prices, bins=30, alpha=0.7)
        plt.xlabel('Average Price ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Optimal Prices')
        
        # Plot 2: Price vs Remaining Capacity
        plt.subplot(132)
        total_capacity = [sum(state) for state in self.states]
        avg_prices = [np.mean(self.policy[state][1]) 
                     for state in self.states 
                     if self.policy[state][1] is not None]
        
        plt.scatter(total_capacity, avg_prices, alpha=0.5)
        plt.xlabel('Total Available Rooms')
        plt.ylabel('Average Price ($)')
        plt.title('Price vs. Available Capacity')
        
        # Plot 3: Price correlation between days
        plt.subplot(133)
        day1_prices = []
        day2_prices = []
        for state in self.states:
            if self.policy[state][1] is not None:
                day1_prices.append(self.policy[state][1][0])
                day2_prices.append(self.policy[state][1][1])
        
        plt.scatter(day1_prices, day2_prices, alpha=0.5)
        plt.xlabel('Day 1 Price ($)')
        plt.ylabel('Day 2 Price ($)')
        plt.title('Price Correlation Between Days')
        
    def _plot_booking_class_analysis(self):
        """
        Analyze and visualize booking class characteristics and their impact.
        Shows patterns in multi-day bookings and pricing.
        """
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Length of stay distribution
        plt.subplot(131)
        lengths = [bc.length for bc in self.booking_classes]
        plt.hist(lengths, bins=range(1, self.params.N + 2), 
                alpha=0.7, rwidth=0.8)
        plt.xlabel('Length of Stay')
        plt.ylabel('Number of Booking Classes')
        plt.title('Distribution of Stay Lengths')
        
        # Plot 2: Average price by length of stay
        plt.subplot(132)
        avg_prices_by_length = {}
        for bc in self.booking_classes:
            prices = []
            for state in self.states:
                if self.policy[state][1] is not None:
                    stay_prices = [self.policy[state][1][day-1] 
                                 for day in bc.stay_days]
                    prices.append(np.mean(stay_prices))
            avg_prices_by_length[bc.length] = np.mean(prices)
        
        plt.bar(avg_prices_by_length.keys(), 
               avg_prices_by_length.values(), alpha=0.7)
        plt.xlabel('Length of Stay')
        plt.ylabel('Average Price ($)')
        plt.title('Average Price by Stay Length')
        
        # Plot 3: Booking class demand analysis
        plt.subplot(133)
        demands = []
        for bc in self.booking_classes:
            avg_demand = np.mean([self.pi[t][bc.id] 
                                for t in range(1, self.params.T + 1)])
            demands.append((bc.id, avg_demand))
        
        ids, demand_values = zip(*demands)
        plt.bar(ids, demand_values, alpha=0.7)
        plt.xlabel('Booking Class ID')
        plt.ylabel('Average Demand Probability')
        plt.title('Demand by Booking Class')
        
    def _plot_state_transitions(self):
        """
        Visualize state transition patterns and probabilities.
        Shows how the system evolves over time.
        """
        plt.figure(figsize=(15, 5))
        
        # Plot 1: State transition network for selected states
        plt.subplot(131)
        # Select a subset of states for visualization
        sample_states = random.sample(self.states, 
                                    min(5, len(self.states)))
        
        # Create a simple directed graph
        for state in sample_states:
            if self.policy[state][1] is not None:
                for bc in self.booking_classes:
                    if self.is_available(state, bc):
                        new_state = self.update_state(state, bc)
                        plt.plot([sum(state)], [sum(new_state)], 
                               'b->', alpha=0.3)
        
        plt.xlabel('Current Total Capacity')
        plt.ylabel('Next Total Capacity')
        plt.title('State Transition Patterns')
        
        # Plot 2: Capacity utilization over time
        plt.subplot(132)
        avg_capacity = {t: [] for t in range(1, self.params.T + 1)}
        for state in self.states:
            for t in range(1, self.params.T + 1):
                avg_capacity[t].append(sum(state) / (self.params.C * self.params.N))
        
        for t, caps in avg_capacity.items():
            plt.boxplot(caps, positions=[t])
        
        plt.xlabel('Time Period')
        plt.ylabel('Capacity Utilization Rate')
        plt.title('Capacity Utilization Distribution')
        
        # Plot 3: Expected revenue by capacity utilization
        plt.subplot(133)
        utilization_rates = []
        revenues = []
        for state in self.states:
            util_rate = sum(state) / (self.params.C * self.params.N)
            revenue = self.V[state][1]
            utilization_rates.append(util_rate)
            revenues.append(revenue)
        
        plt.scatter(utilization_rates, revenues, alpha=0.5)
        plt.xlabel('Capacity Utilization Rate')
        plt.ylabel('Expected Revenue ($)')
        plt.title('Revenue vs. Utilization')

        
    def visualize_all(self):
        """Generate all visualizations."""
        self.plot_pricing_policy_3d()
        self.plot_value_function()
        self.plot_price_distributions()
        self.plot_booking_class_analysis()
        self.plot_state_transitions()
        
    def plot_pricing_policy_3d(self):
        """Plot 3D visualization of pricing policies."""
        try:
            for t in range(1, self.params.T + 1):
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Prepare data
                x1, x2, x3, prices = self._prepare_3d_data(t)
                
                if len(prices) == 0:
                    logger.warning(f"No data to plot for time period {t}")
                    continue
                    
                # Create visualization
                self._create_3d_scatter(ax, x1, x2, x3, prices, t)
                plt.tight_layout()
                
        except Exception as e:
            logger.error(f"Error in 3D plotting: {str(e)}")
            self.plot_pricing_policy_2d(t)

    def _prepare_3d_data(self, t):
        """Prepare data for 3D visualization."""
        x1, x2, x3, prices = [], [], [], []
        for state in self.model.states:
            if self.model.policy[state][t] is not None:
                x1.append(state[0])
                x2.append(state[1])
                x3.append(state[2])
                avg_price = np.mean(self.model.policy[state][t])
                prices.append(avg_price)
        return map(np.array, (x1, x2, x3, prices))

    def _create_3d_scatter(self, ax, x1, x2, x3, prices, t):
        """Create 3D scatter plot."""
        norm = plt.Normalize(prices.min(), prices.max())
        colors = plt.cm.viridis(norm(prices))
        
        scatter = ax.scatter(x1, x2, x3, c=colors, s=50, alpha=0.6)
        
        ax.set_xlabel('Available Rooms Day 1')
        ax.set_ylabel('Available Rooms Day 2')
        ax.set_zlabel('Available Rooms Day 3')
        ax.set_title(f'Optimal Average Pricing Policy (t={t})')
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Average Price ($)')
        
        ax.grid(True)
        ax.view_init(elev=20, azim=45)
        self._set_axis_limits(ax)

    def _set_axis_limits(self, ax):
        """Set axis limits for 3D plot."""
        ax.set_xlim(0, self.params.C)
        ax.set_ylim(0, self.params.C)
        ax.set_zlim(0, self.params.C)

    def plot_value_function(self):
        """Plot value function analysis."""
        plt.figure(figsize=(15, 5))
        
        # Value vs total capacity
        self._plot_value_vs_capacity()
        
        # Value function heatmap
        self._plot_value_heatmap()
        
        # Value function evolution
        self._plot_value_evolution()
        
        plt.tight_layout()
        plt.show()

    def _plot_value_vs_capacity(self):
        """Plot value function vs total capacity."""
        plt.subplot(131)
        for t in range(1, self.params.T + 1):
            x = [sum(state) for state in self.model.states]
            y = [self.model.V[state][t] for state in self.model.states]
            plt.scatter(x, y, alpha=0.5, label=f't={t}')
        
        plt.xlabel('Total Available Rooms')
        plt.ylabel('Expected Revenue ($)')
        plt.title('Value Function vs. Total Capacity')
        plt.legend()

    def plot_price_distributions(self):
        """Plot price distribution analysis."""
        plt.figure(figsize=(15, 5))
        
        self._plot_price_histogram()
        self._plot_price_vs_capacity()
        self._plot_price_correlation()
        
        plt.tight_layout()
        plt.show()

    def plot_booking_class_analysis(self):
        """Plot booking class analysis."""
        plt.figure(figsize=(15, 5))
        
        self._plot_length_distribution()
        self._plot_price_by_length()
        self._plot_demand_analysis()
        
        plt.tight_layout()
        plt.show()

    def plot_state_transitions(self):
        """Plot state transition analysis."""
        plt.figure(figsize=(15, 5))
        
        self._plot_transition_network()
        self._plot_capacity_utilization()
        self._plot_revenue_utilization()
        
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        """Generate all visualizations."""
        self.plot_pricing_policy_3d()
        self.plot_value_function()
        self.plot_price_distributions()
        self.plot_booking_class_analysis()
        self.plot_state_transitions()

class AnimatedPricingVisualizer(HotelPricingVisualizer):
    """
    Extended visualizer with animation capabilities.
    """
    def animate_3d_pricing(self, save_path=None):
        """Create animated rotation of 3D pricing plot."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x1, x2, x3, prices = self._prepare_3d_data(1)
        self._create_3d_scatter(ax, x1, x2, x3, prices, 1)
        
        frames = []
        for angle in range(0, 360, 5):
            ax.view_init(30, angle)
            plt.draw()
            
            if save_path:
                # Save frame
                pass
                
        return frames

class InteractivePricingVisualizer(HotelPricingVisualizer):
    """
    Extended visualizer with interactive features.
    """
    def create_interactive_dashboard(self):
        """Create interactive dashboard using plotly or other interactive library."""
        pass