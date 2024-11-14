"""
visualization.py - Comprehensive visualization module for hotel pricing model.

This module provides visualization capabilities for analyzing and presenting results
from the hotel dynamic pricing model. It includes basic static visualizations,
animated views, and interactive features.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, List, Any, Optional
import random

# Configure logging
logger = logging.getLogger(__name__)

class HotelPricingVisualizer:
    """
    Base visualizer class for hotel pricing model results.
    
    This class provides comprehensive visualization capabilities for analyzing
    pricing policies, value functions, and booking patterns.
    
    Attributes:
        data (Dict): Complete solution data
        params: Problem parameters
        states: All possible states
        policy: Computed pricing policy
        V: Value function
        booking_classes: All booking classes
    """
    def __init__(self, solution_data: Dict):
        """
        Initialize visualizer with solution data.
        
        Args:
            solution_data: Dictionary containing model solution data
        """
        self.data = solution_data
        self.params = solution_data['params']
        self.states = solution_data['states']
        self.policy = solution_data['policy']
        self.V = solution_data['value_function']
        self.booking_classes = solution_data['booking_classes']
        if 'arrival_probabilities' in solution_data:
            self.pi = solution_data['arrival_probabilities']
        self.setup_style()
        
    def setup_style(self):
        """Configure visualization style settings."""
        sns.set_theme()
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': [12, 8],
            'figure.dpi': 100,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def plot_all(self):
        """Generate all available visualizations."""
        self.plot_pricing_policy_3d()
        self.plot_value_function()
        self.plot_price_distributions()
        self.plot_booking_class_analysis()
        self.plot_state_transitions()
        plt.tight_layout()
        plt.show()

    def plot_pricing_policy_3d(self):
        """Plot 3D visualization of pricing policies."""
        try:
            for t in range(1, self.params.T + 1):
                self._create_3d_policy_plot(t)
        except Exception as e:
            logger.error(f"Error in 3D plotting: {str(e)}")
            self.plot_pricing_policy_2d(t)

    def _create_3d_policy_plot(self, t: int):
        """Create a single 3D policy plot for a given time period."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        x1, x2, x3, prices = [], [], [], []
        for state in self.states:
            if self.policy[state][t] is not None:
                x1.append(state[0])
                x2.append(state[1])
                x3.append(state[2])
                avg_price = np.mean(self.policy[state][t])
                prices.append(avg_price)

        if not prices:
            logger.warning(f"No data to plot for time period {t}")
            return

        x1, x2, x3, prices = map(np.array, [x1, x2, x3, prices])
        
        # Create visualization
        norm = plt.Normalize(prices.min(), prices.max())
        scatter = ax.scatter(x1, x2, x3, 
                           c=prices,
                           cmap='viridis',
                           s=50,
                           alpha=0.6)

        # Labels and formatting
        ax.set_xlabel('Available Rooms Day 1')
        ax.set_ylabel('Available Rooms Day 2')
        ax.set_zlabel('Available Rooms Day 3')
        ax.set_title(f'Optimal Average Pricing Policy (t={t})')
        
        # Colorbar
        plt.colorbar(scatter, label='Average Price ($)')
        
        # View and limits
        ax.view_init(elev=20, azim=45)
        ax.set_xlim(0, self.params.C)
        ax.set_ylim(0, self.params.C)
        ax.set_zlim(0, self.params.C)
        
        plt.tight_layout()

    def plot_pricing_policy_2d(self, t: int):
        """
        Create 2D slice views of the pricing policy.
        
        Args:
            t: Time period to visualize
        """
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
            
            self._create_2d_slice(ax, t, day3_capacity)
        
        # Remove empty subplots
        for idx in range(day3_capacity + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()

    def _create_2d_slice(self, ax, t: int, day3_capacity: int):
        """Create a 2D slice plot for a specific day 3 capacity."""
        # Collect data for this slice
        x1, x2, prices = [], [], []
        for state in self.states:
            if (self.policy[state][t] is not None and 
                state[2] == day3_capacity):
                x1.append(state[0])
                x2.append(state[1])
                prices.append(np.mean(self.policy[state][t]))
        
        if prices:
            # Create heatmap
            x1_unique = sorted(set(x1))
            x2_unique = sorted(set(x2))
            price_grid = np.full((len(x1_unique), len(x2_unique)), np.nan)
            
            for i, j, p in zip(x1, x2, prices):
                i_idx = x1_unique.index(i)
                j_idx = x2_unique.index(j)
                price_grid[i_idx, j_idx] = p
            
            im = ax.imshow(price_grid.T, origin='lower', 
                          aspect='equal', cmap='viridis')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No Data',
                   horizontalalignment='center',
                   verticalalignment='center')
        
        # Labels
        ax.set_xlabel('Day 1 Capacity')
        ax.set_ylabel('Day 2 Capacity')
        ax.set_title(f'Day 3 Capacity = {day3_capacity}')
        
        # Ticks
        if prices:
            ax.set_xticks(range(len(x1_unique)))
            ax.set_yticks(range(len(x2_unique)))
            ax.set_xticklabels(x1_unique)
            ax.set_yticklabels(x2_unique)

    def plot_value_function(self):
        """Plot comprehensive value function analysis."""
        plt.figure(figsize=(15, 5))
        
        # Value vs capacity
        plt.subplot(131)
        self._plot_value_vs_capacity()
        
        # Value heatmap
        plt.subplot(132)
        self._plot_value_heatmap()
        
        # Value evolution
        plt.subplot(133)
        self._plot_value_evolution()
        
        plt.tight_layout()
        plt.show()

    def _plot_value_vs_capacity(self):
        """Plot value function against total capacity."""
        for t in range(1, self.params.T + 1):
            x = [sum(state) for state in self.states]
            y = [self.V[state][t] for state in self.states]
            plt.scatter(x, y, alpha=0.5, label=f't={t}')
        
        plt.xlabel('Total Available Rooms')
        plt.ylabel('Expected Revenue ($)')
        plt.title('Value Function vs. Total Capacity')
        plt.legend()

    def _plot_value_heatmap(self):
        """Create heatmap of value function."""
        state_values_t1 = np.zeros((self.params.C + 1, self.params.C + 1))
        for state in self.states:
            if state[2] == self.params.C // 2:
                state_values_t1[state[0], state[1]] = self.V[state][1]
        
        plt.imshow(state_values_t1, cmap='viridis', origin='lower')
        plt.colorbar(label='Expected Revenue ($)')
        plt.xlabel('Day 1 Capacity')
        plt.ylabel('Day 2 Capacity')
        plt.title('Value Function Heatmap (t=1, Day 3 fixed)')

    def _plot_value_evolution(self):
        """Plot value function evolution over time."""
        sample_states = random.sample(self.states, min(10, len(self.states)))
        for state in sample_states:
            values = [self.V[state][t] for t in range(1, self.params.T + 1)]
            plt.plot(range(1, self.params.T + 1), values, 
                    marker='o', label=f'State {state}')
        
        plt.xlabel('Time Period')
        plt.ylabel('Expected Revenue ($)')
        plt.title('Value Function Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_price_distributions(self):
        """Plot analysis of price distributions."""
        plt.figure(figsize=(15, 5))
        
        # Distribution histogram
        plt.subplot(131)
        self._plot_price_histogram()
        
        # Price vs capacity
        plt.subplot(132)
        self._plot_price_capacity_relationship()
        
        # Price correlations
        plt.subplot(133)
        self._plot_price_correlations()
        
        plt.tight_layout()
        plt.show()

    def _plot_price_histogram(self):
        """Create histogram of optimal prices."""
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

    def _plot_price_capacity_relationship(self):
        """Plot relationship between prices and capacity."""
        total_capacity = [sum(state) for state in self.states]
        avg_prices = [np.mean(self.policy[state][1]) 
                     for state in self.states 
                     if self.policy[state][1] is not None]
        
        plt.scatter(total_capacity, avg_prices, alpha=0.5)
        plt.xlabel('Total Available Rooms')
        plt.ylabel('Average Price ($)')
        plt.title('Price vs. Available Capacity')

    def _plot_price_correlations(self):
        """Plot price correlations between days."""
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

    def plot_booking_class_analysis(self):
        """Plot analysis of booking classes."""
        plt.figure(figsize=(15, 5))
        
        # Stay length distribution
        plt.subplot(131)
        self._plot_stay_length_distribution()
        
        # Price by stay length
        plt.subplot(132)
        self._plot_price_by_stay_length()
        
        # Demand analysis
        plt.subplot(133)
        self._plot_booking_demand()
        
        plt.tight_layout()
        plt.show()

    def _plot_stay_length_distribution(self):
        """Plot distribution of stay lengths."""
        lengths = [bc.length for bc in self.booking_classes]
        plt.hist(lengths, bins=range(1, self.params.N + 2), 
                alpha=0.7, rwidth=0.8)
        plt.xlabel('Length of Stay')
        plt.ylabel('Number of Booking Classes')
        plt.title('Distribution of Stay Lengths')

    def _plot_price_by_stay_length(self):
        """Plot average prices by length of stay."""
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

    def _plot_booking_demand(self):
        """Plot booking demand analysis."""
        demands = []
        for bc in self.booking_classes:
            if hasattr(self, 'pi'):
                avg_demand = np.mean([self.pi[t][bc.id] 
                                    for t in range(1, self.params.T + 1)])
                demands.append((bc.id, avg_demand))
        
        if demands:
            ids, demand_values = zip(*demands)
            plt.bar(ids, demand_values, alpha=0.7)
            plt.xlabel('Booking Class ID')
            plt.ylabel('Average Demand Probability')
            plt.title('Demand by Booking Class')
        else:
            plt.text(0.5, 0.5, 'No Demand Data Available',
                    horizontalalignment='center')

    def plot_state_transitions(self):
        """Plot state transition analysis."""
        plt.figure(figsize=(15, 5))
        
        # Transition network
        plt.subplot(131)
        self._plot_transition_network()
        
        # Capacity utilization
        plt.subplot(132)
        self._plot_capacity_utilization()
        
        # Revenue utilization
        plt.subplot(133)
        self._plot_revenue_utilization()
        
    def _plot_transition_network(self):
        """Plot network of state transitions."""
        # Select representative states
        sample_states = random.sample(self.states, min(5, len(self.states)))
        
        # Plot transitions
        for state in sample_states:
            if self.policy[state][1] is not None:
                for bc in self.booking_classes:
                    if self._check_state_availability(state, bc):
                        new_state = self._calculate_new_state(state, bc)
                        plt.plot([sum(state)], [sum(new_state)], 
                               'b->', alpha=0.3)
        
        plt.xlabel('Current Total Capacity')
        plt.ylabel('Next Total Capacity')
        plt.title('State Transition Patterns')

    def _plot_capacity_utilization(self):
        """Plot capacity utilization patterns."""
        avg_capacity = {t: [] for t in range(1, self.params.T + 1)}
        for state in self.states:
            for t in range(1, self.params.T + 1):
                util_rate = sum(state) / (self.params.C * self.params.N)
                avg_capacity[t].append(util_rate)
        
        for t, caps in avg_capacity.items():
            plt.boxplot(caps, positions=[t])
        
        plt.xlabel('Time Period')
        plt.ylabel('Capacity Utilization Rate')
        plt.title('Capacity Utilization Distribution')

    def _plot_revenue_utilization(self):
        """Plot relationship between revenue and utilization."""
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

    def _check_state_availability(self, state: Tuple[int, ...], bc) -> bool:
        """
        Check if a booking class can be accommodated in a given state.
        
        Args:
            state: Current state tuple
            bc: BookingClass instance
        
        Returns:
            bool: True if booking can be accommodated
        """
        return all(state[i] >= bc.consumption_vector[i] 
                  for i in range(self.params.N))

    def _calculate_new_state(self, state: Tuple[int, ...], bc) -> Tuple[int, ...]:
        """
        Calculate new state after accepting a booking.
        
        Args:
            state: Current state tuple
            bc: BookingClass instance
            
        Returns:
            Tuple[int, ...]: New state after booking
        """
        return tuple(s - c for s, c in zip(state, bc.consumption_vector))


class AnimatedHotelPricingVisualizer(HotelPricingVisualizer):
    """
    Extended visualizer with animation capabilities.
    
    This class adds animation features to the base visualizer, allowing for
    dynamic visualization of pricing policies and state transitions.
    """
    
    def animate_3d_policy(self, save_path: Optional[str] = None,
                         rotation_speed: int = 5):
        """
        Create animated rotation of 3D pricing visualization.
        
        Args:
            save_path: Optional path to save animation frames
            rotation_speed: Degrees to rotate per frame (default: 5)
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data for time period 1
        x1, x2, x3, prices = [], [], [], []
        for state in self.states:
            if self.policy[state][1] is not None:
                x1.append(state[0])
                x2.append(state[1])
                x3.append(state[2])
                avg_price = np.mean(self.policy[state][1])
                prices.append(avg_price)
        
        # Convert to numpy arrays
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        prices = np.array(prices)
        
        # Create scatter plot
        scatter = ax.scatter(x1, x2, x3, c=prices, cmap='viridis', alpha=0.6)
        
        # Labels
        ax.set_xlabel('Available Rooms Day 1')
        ax.set_ylabel('Available Rooms Day 2')
        ax.set_zlabel('Available Rooms Day 3')
        plt.colorbar(scatter, label='Average Price ($)')
        
        # Animate
        for angle in range(0, 360, rotation_speed):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.1)
            
            if save_path:
                plt.savefig(f'{save_path}/frame_{angle:03d}.png')
        
        plt.show()

    def animate_value_evolution(self, save_path: Optional[str] = None):
        """
        Create animation showing evolution of value function over time.
        
        Args:
            save_path: Optional path to save animation frames
        """
        fig = plt.figure(figsize=(10, 7))
        
        # Sample some states for visualization
        sample_states = random.sample(self.states, min(5, len(self.states)))
        
        # Plot evolution
        for t in range(1, self.params.T + 1):
            plt.clf()
            for state in sample_states:
                values = [self.V[state][i] for i in range(1, t + 1)]
                plt.plot(range(1, t + 1), values, 
                        marker='o', label=f'State {state}')
            
            plt.xlabel('Time Period')
            plt.ylabel('Expected Revenue ($)')
            plt.title(f'Value Function Evolution (t={t})')
            plt.legend()
            
            if save_path:
                plt.savefig(f'{save_path}/value_evolution_{t:03d}.png')
            
            plt.pause(0.5)


class InteractiveHotelPricingVisualizer(HotelPricingVisualizer):
    """
    Extended visualizer with interactive features.
    
    This class adds interactive visualization capabilities using plotly
    for dynamic exploration of the pricing model results.
    """
    
    def create_interactive_dashboard(self):
        """Create interactive dashboard for data exploration."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'scatter3d'}],
                      [{'type': 'heatmap'}, {'type': 'scatter'}]],
                subplot_titles=('Pricing Surface', 'State Space',
                              'Value Function', 'Revenue vs. Capacity')
            )
            
            # Add pricing surface
            self._add_pricing_surface(fig, row=1, col=1)
            
            # Add state space scatter
            self._add_state_space_scatter(fig, row=1, col=2)
            
            # Add value function heatmap
            self._add_value_heatmap(fig, row=2, col=1)
            
            # Add revenue scatter
            self._add_revenue_scatter(fig, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                title_text="Hotel Pricing Analysis Dashboard"
            )
            
            fig.show()
            
        except ImportError:
            logger.warning("Plotly is required for interactive dashboard. "
                         "Please install with: pip install plotly")
            return None

    def _add_pricing_surface(self, fig, row: int, col: int):
        """Add pricing surface plot to dashboard."""
        # Implementation for pricing surface
        pass

    def _add_state_space_scatter(self, fig, row: int, col: int):
        """Add state space scatter plot to dashboard."""
        # Implementation for state space scatter
        pass

    def _add_value_heatmap(self, fig, row: int, col: int):
        """Add value function heatmap to dashboard."""
        # Implementation for value heatmap
        pass

    def _add_revenue_scatter(self, fig, row: int, col: int):
        """Add revenue scatter plot to dashboard."""
        # Implementation for revenue scatter
        pass


# Optional: Helper functions for common visualization tasks
def create_colormap(values: np.ndarray, cmap: str = 'viridis') -> np.ndarray:
    """
    Create normalized colors for values using specified colormap.
    
    Args:
        values: Array of values to map to colors
        cmap: Name of colormap to use
        
    Returns:
        Array of RGBA colors
    """
    norm = plt.Normalize(values.min(), values.max())
    return plt.cm.get_cmap(cmap)(norm(values))

def save_visualization(fig: plt.Figure, filename: str):
    """
    Save visualization figure with standard settings.
    
    Args:
        fig: Figure to save
        filename: Output filename
    """
    fig.savefig(filename, dpi=300, bbox_inches='tight')