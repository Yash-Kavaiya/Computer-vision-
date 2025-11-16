"""
Visualization Module for Cricket Batting Analysis
Creates charts, graphs, and visual reports for batting statistics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import seaborn as sns
from datetime import datetime
import os


class BattingVisualization:
    """
    Create visualizations for cricket batting analysis
    """

    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualization settings

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        sns.set_palette("husl")
        self.output_dir = 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_batting_metrics(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Create a comprehensive batting metrics dashboard

        Args:
            metrics: Dictionary containing batting statistics
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cricket Batting Performance Dashboard', fontsize=16, fontweight='bold')

        # 1. Key Statistics Bar Chart
        ax1 = axes[0, 0]
        stats_to_plot = ['runs', 'boundaries', 'sixes', 'dot_balls']
        stats_values = [metrics.get(stat, 0) for stat in stats_to_plot]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

        bars = ax1.bar(range(len(stats_to_plot)), stats_values, color=colors)
        ax1.set_xticks(range(len(stats_to_plot)))
        ax1.set_xticklabels(['Runs', 'Fours', 'Sixes', 'Dots'])
        ax1.set_ylabel('Count')
        ax1.set_title('Batting Statistics', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        # 2. Strike Rate and Percentages
        ax2 = axes[0, 1]
        percentages = {
            'Strike Rate': metrics.get('strike_rate', 0),
            'Dot %': metrics.get('dot_percentage', 0),
            'Boundary %': metrics.get('boundary_percentage', 0)
        }

        y_pos = np.arange(len(percentages))
        values = list(percentages.values())
        colors_p = ['#9b59b6', '#e67e22', '#1abc9c']

        bars = ax2.barh(y_pos, values, color=colors_p)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(percentages.keys())
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Batting Percentages', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{values[i]:.1f}%', ha='left', va='center', fontweight='bold')

        # 3. Boundary Contribution Pie Chart
        ax3 = axes[1, 0]
        boundary_runs = metrics.get('runs_from_boundaries', 0)
        total_runs = metrics.get('runs', 1)
        other_runs = max(0, total_runs - boundary_runs)

        sizes = [boundary_runs, other_runs]
        labels = ['From Boundaries', 'From Singles/Doubles']
        colors_pie = ['#e74c3c', '#3498db']
        explode = (0.1, 0)

        if total_runs > 0:
            ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax3.set_title('Run Distribution', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)

        # 4. Performance Gauge
        ax4 = axes[1, 1]
        self._create_performance_gauge(ax4, metrics)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'batting_metrics.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def _create_performance_gauge(self, ax, metrics: Dict):
        """Create a performance gauge visualization"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Calculate overall performance score
        strike_rate = metrics.get('strike_rate', 0)
        consistency = metrics.get('consistency_index', 50)

        # Normalize to 0-100 scale
        sr_score = min(100, (strike_rate / 150) * 100)  # 150 SR = 100%
        overall_score = (sr_score + consistency) / 2

        # Determine performance level
        if overall_score >= 75:
            performance = 'Excellent'
            color = '#2ecc71'
        elif overall_score >= 60:
            performance = 'Good'
            color = '#3498db'
        elif overall_score >= 40:
            performance = 'Average'
            color = '#f39c12'
        else:
            performance = 'Needs Improvement'
            color = '#e74c3c'

        # Draw gauge
        circle = plt.Circle((5, 5), 3, color=color, alpha=0.3)
        ax.add_patch(circle)

        # Add text
        ax.text(5, 6, 'Performance', ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.text(5, 5, performance, ha='center', va='center',
               fontsize=16, fontweight='bold', color=color)
        ax.text(5, 3.5, f'Score: {overall_score:.1f}/100', ha='center', va='center',
               fontsize=12)

    def plot_shot_distribution(self, shot_analysis: Dict, save_path: Optional[str] = None):
        """
        Visualize shot distribution

        Args:
            shot_analysis: Shot analysis data
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Shot Selection Analysis', fontsize=16, fontweight='bold')

        # Shot Type Distribution
        ax1 = axes[0]
        shot_dist = shot_analysis.get('shot_distribution', {})

        if shot_dist:
            shots = list(shot_dist.keys())
            counts = list(shot_dist.values())

            # Sort by count
            sorted_pairs = sorted(zip(shots, counts), key=lambda x: x[1], reverse=True)
            shots, counts = zip(*sorted_pairs)

            colors = plt.cm.Set3(np.linspace(0, 1, len(shots)))
            bars = ax1.barh(range(len(shots)), counts, color=colors)
            ax1.set_yticks(range(len(shots)))
            ax1.set_yticklabels(shots)
            ax1.set_xlabel('Number of Shots')
            ax1.set_title('Shot Type Distribution', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(counts[i])}', ha='left', va='center', fontweight='bold')

        # Zone Distribution
        ax2 = axes[1]
        zone_dist = shot_analysis.get('zone_distribution', {})

        if zone_dist:
            zones = list(zone_dist.keys())
            zone_counts = list(zone_dist.values())

            colors_zone = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            ax2.pie(zone_counts, labels=zones, colors=colors_zone[:len(zones)],
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.set_title('Scoring Zone Distribution', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'shot_distribution.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_performance_trend(self, innings_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot performance trends over time

        Args:
            innings_df: DataFrame with innings data
            save_path: Optional path to save the figure
        """
        if innings_df.empty:
            print("No data available for trend analysis")
            return

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Performance Trends Over Time', fontsize=16, fontweight='bold')

        # Runs trend
        ax1 = axes[0]
        if 'runs' in innings_df.columns:
            ax1.plot(innings_df.index, innings_df['runs'], marker='o',
                    linewidth=2, markersize=8, label='Runs', color='#3498db')

            if 'avg_runs_5' in innings_df.columns:
                ax1.plot(innings_df.index, innings_df['avg_runs_5'],
                        linestyle='--', linewidth=2, label='5-Innings Avg',
                        color='#e74c3c', alpha=0.7)

            ax1.set_ylabel('Runs')
            ax1.set_title('Runs per Innings', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Strike Rate trend
        ax2 = axes[1]
        if 'strike_rate' in innings_df.columns:
            ax2.plot(innings_df.index, innings_df['strike_rate'], marker='s',
                    linewidth=2, markersize=8, label='Strike Rate', color='#2ecc71')

            if 'avg_sr_5' in innings_df.columns:
                ax2.plot(innings_df.index, innings_df['avg_sr_5'],
                        linestyle='--', linewidth=2, label='5-Innings Avg',
                        color='#e67e22', alpha=0.7)

            # Add reference line for good strike rate
            ax2.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='SR 100')

            ax2.set_xlabel('Innings')
            ax2.set_ylabel('Strike Rate')
            ax2.set_title('Strike Rate Trend', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'performance_trend.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def create_wagon_wheel(self, shots: List[Dict], save_path: Optional[str] = None):
        """
        Create a wagon wheel visualization showing shot directions

        Args:
            shots: List of shot dictionaries with angle and runs
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Convert shot zones to angles (in radians)
        zone_angles = {
            'V': 0,                    # Straight
            'Off Side': np.pi/2,       # Square on off
            'Leg Side': -np.pi/2,      # Square on leg
            'Behind': np.pi            # Behind wicket
        }

        angles = []
        distances = []
        colors_map = {0: 'blue', 1: 'green', 2: 'orange', 4: 'red', 6: 'purple'}
        colors = []

        for shot in shots:
            zone = shot.get('zone', 'V')
            runs = shot.get('runs', 0)

            # Add some randomness to avoid overlap
            base_angle = zone_angles.get(zone, 0)
            angle = base_angle + np.random.uniform(-0.3, 0.3)

            angles.append(angle)
            distances.append(1 + runs * 0.2)  # Scale by runs
            colors.append(colors_map.get(runs, 'gray'))

        # Plot shots
        ax.scatter(angles, distances, c=colors, s=100, alpha=0.6)

        # Configure plot
        ax.set_ylim(0, 3)
        ax.set_theta_zero_location('N')  # 0 degrees at top (straight)
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title('Wagon Wheel - Shot Distribution\n', fontsize=14, fontweight='bold', pad=20)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Dot Ball'),
            Patch(facecolor='green', label='1-2 Runs'),
            Patch(facecolor='red', label='Four'),
            Patch(facecolor='purple', label='Six')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'wagon_wheel.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def create_heatmap(self, shot_data: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create a heatmap showing batting effectiveness in different zones

        Args:
            shot_data: DataFrame with shot information
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a matrix for the cricket field
        # Simplified: rows = line (yorker to bouncer), cols = line (off to leg)
        zones_matrix = np.zeros((5, 5))

        # Populate with random data for demonstration
        # In real implementation, this would be based on actual shot data
        zones_matrix = np.random.randint(0, 10, size=(5, 5))

        # Create heatmap
        sns.heatmap(zones_matrix, annot=True, fmt='d', cmap='YlOrRd',
                   cbar_kws={'label': 'Runs Scored'}, ax=ax)

        ax.set_title('Batting Effectiveness Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Horizontal Position (Off → Leg)')
        ax.set_ylabel('Vertical Position (Full → Short)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'effectiveness_heatmap.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def create_comparison_chart(self, players_data: List[Dict], save_path: Optional[str] = None):
        """
        Create a radar chart comparing multiple players

        Args:
            players_data: List of player statistics dictionaries
            save_path: Optional path to save the figure
        """
        if not players_data:
            return

        categories = ['Strike Rate', 'Boundary %', 'Consistency', 'Shot Variety', 'Risk Management']
        num_vars = len(categories)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        for i, player in enumerate(players_data[:4]):  # Max 4 players
            values = [
                min(100, player.get('strike_rate', 0) / 1.5),  # Normalize SR
                player.get('boundary_percentage', 0),
                player.get('consistency_index', 50),
                player.get('shot_variety_score', 50),
                100 - player.get('risk_score', 50)  # Invert risk
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=player.get('name', f'Player {i+1}'),
                   color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Player Comparison - Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'player_comparison.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def generate_visual_report(self, complete_data: Dict, output_dir: Optional[str] = None):
        """
        Generate a complete visual report with all charts

        Args:
            complete_data: Dictionary with all analysis data
            output_dir: Optional output directory
        """
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

        print("Generating comprehensive visual report...")

        # Generate all visualizations
        if 'metrics' in complete_data:
            self.plot_batting_metrics(complete_data['metrics'])
            print("  ✓ Batting metrics dashboard created")

        if 'shot_analysis' in complete_data:
            self.plot_shot_distribution(complete_data['shot_analysis'])
            print("  ✓ Shot distribution charts created")

        if 'innings_data' in complete_data:
            df = pd.DataFrame(complete_data['innings_data'])
            self.plot_performance_trend(df)
            print("  ✓ Performance trend charts created")

        if 'shots' in complete_data:
            self.create_wagon_wheel(complete_data['shots'])
            print("  ✓ Wagon wheel visualization created")

        print(f"\nAll visualizations saved to: {self.output_dir}/")


def main():
    """Example usage of visualization module"""
    viz = BattingVisualization()

    print("Cricket Batting Visualization Module")
    print("=" * 60)

    # Example data
    sample_metrics = {
        'runs': 75,
        'balls_faced': 58,
        'strike_rate': 129.31,
        'boundaries': 8,
        'sixes': 2,
        'dot_balls': 15,
        'dot_percentage': 25.86,
        'boundary_percentage': 17.24,
        'runs_from_boundaries': 44,
        'boundary_contribution': 58.67,
        'consistency_index': 65.5
    }

    sample_shot_analysis = {
        'total_shots': 58,
        'shot_distribution': {
            'Cover Drive': 12,
            'Pull Shot': 8,
            'Defense': 15,
            'Flick Shot': 10,
            'Cut Shot': 7,
            'Straight Drive': 6
        },
        'zone_distribution': {
            'V': 18,
            'Off Side': 22,
            'Leg Side': 16,
            'Behind': 2
        }
    }

    # Generate visualizations
    print("\nGenerating sample visualizations...")
    viz.plot_batting_metrics(sample_metrics)
    viz.plot_shot_distribution(sample_shot_analysis)

    print(f"\nVisualizations saved to: {viz.output_dir}/")


if __name__ == "__main__":
    main()
