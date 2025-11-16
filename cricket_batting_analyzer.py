"""
Cricket Batting Analysis System
Comprehensive analysis tool for cricket batting using computer vision and statistical analysis
"""

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os


class CricketBattingAnalyzer:
    """
    Main class for cricket batting analysis using computer vision
    """

    def __init__(self):
        """Initialize the batting analyzer"""
        self.shot_types = [
            'Cover Drive', 'Straight Drive', 'Pull Shot', 'Cut Shot',
            'Hook Shot', 'Sweep Shot', 'Reverse Sweep', 'Flick Shot',
            'Square Drive', 'Late Cut', 'Defense', 'Leave'
        ]

        self.batting_zones = {
            'V': (0, 45),           # Straight drives
            'Off Side': (45, 135),  # Off side shots
            'Leg Side': (-135, -45), # Leg side shots
            'Behind': (135, 180)     # Behind wicket
        }

        self.performance_metrics = []
        self.session_data = {}

    def analyze_batting_stance(self, frame: np.ndarray) -> Dict:
        """
        Analyze batting stance from a video frame

        Args:
            frame: Video frame (numpy array)

        Returns:
            Dictionary with stance analysis
        """
        stance_analysis = {
            'timestamp': datetime.now().isoformat(),
            'balance': 'good',
            'head_position': 'aligned',
            'feet_position': 'shoulder_width',
            'bat_angle': 0,
            'weight_distribution': 50  # Percentage on front foot
        }

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect key points (simplified - would use pose estimation in real implementation)
        height, width = gray.shape

        # Analyze frame properties
        brightness = np.mean(gray)
        contrast = np.std(gray)

        stance_analysis['frame_quality'] = {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'resolution': f"{width}x{height}"
        }

        return stance_analysis

    def detect_shot_type(self, frames: List[np.ndarray], ball_trajectory: Optional[Dict] = None) -> Dict:
        """
        Detect and classify the type of cricket shot played

        Args:
            frames: List of video frames showing the shot
            ball_trajectory: Optional ball trajectory data

        Returns:
            Shot classification with confidence scores
        """
        if not frames:
            return {'shot_type': 'Unknown', 'confidence': 0.0}

        # Analyze shot characteristics
        shot_features = self._extract_shot_features(frames)

        # Classify shot based on features
        shot_classification = {
            'shot_type': 'Cover Drive',  # Placeholder - would use ML model
            'confidence': 0.85,
            'bat_speed': shot_features['bat_speed'],
            'follow_through': shot_features['follow_through'],
            'footwork': shot_features['footwork'],
            'timing': shot_features['timing'],
            'shot_zone': self._determine_shot_zone(shot_features)
        }

        return shot_classification

    def _extract_shot_features(self, frames: List[np.ndarray]) -> Dict:
        """Extract features from shot frames for classification"""
        features = {
            'bat_speed': np.random.uniform(60, 120),  # km/h (placeholder)
            'follow_through': 'complete',
            'footwork': 'front_foot',
            'timing': 'good',
            'contact_point': 'front_of_body',
            'elbow_position': 'high',
            'head_movement': 'minimal'
        }

        # Analyze frame sequence
        if len(frames) > 1:
            # Calculate motion between frames
            motion_diff = cv2.absdiff(frames[0], frames[-1])
            motion_intensity = np.mean(motion_diff)
            features['motion_intensity'] = float(motion_intensity)

        return features

    def _determine_shot_zone(self, features: Dict) -> str:
        """Determine which zone the shot was played to"""
        # Simplified zone detection
        zones = ['V', 'Off Side', 'Leg Side', 'Behind']
        return np.random.choice(zones)

    def calculate_batting_metrics(self, runs: int, balls_faced: int,
                                 boundaries: int, sixes: int,
                                 dots: int) -> Dict:
        """
        Calculate comprehensive batting statistics

        Args:
            runs: Total runs scored
            balls_faced: Number of balls faced
            boundaries: Number of 4s
            sixes: Number of 6s
            dots: Number of dot balls

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            'runs': runs,
            'balls_faced': balls_faced,
            'strike_rate': (runs / balls_faced * 100) if balls_faced > 0 else 0,
            'boundaries': boundaries,
            'sixes': sixes,
            'dot_balls': dots,
            'dot_percentage': (dots / balls_faced * 100) if balls_faced > 0 else 0,
            'boundary_percentage': ((boundaries + sixes) / balls_faced * 100) if balls_faced > 0 else 0,
            'runs_from_boundaries': (boundaries * 4) + (sixes * 6),
            'boundary_contribution': 0
        }

        if runs > 0:
            metrics['boundary_contribution'] = (metrics['runs_from_boundaries'] / runs * 100)

        # Calculate scoring patterns
        metrics['scoring_rate'] = self._calculate_scoring_rate(runs, balls_faced)
        metrics['consistency_index'] = self._calculate_consistency(metrics)

        return metrics

    def _calculate_scoring_rate(self, runs: int, balls: int) -> str:
        """Categorize scoring rate"""
        if balls == 0:
            return 'N/A'

        strike_rate = (runs / balls) * 100

        if strike_rate < 60:
            return 'Slow'
        elif strike_rate < 90:
            return 'Moderate'
        elif strike_rate < 130:
            return 'Aggressive'
        else:
            return 'Explosive'

    def _calculate_consistency(self, metrics: Dict) -> float:
        """Calculate batting consistency index (0-100)"""
        # Higher consistency means more balanced scoring
        boundary_contrib = metrics['boundary_contribution']
        dot_percentage = metrics['dot_percentage']

        # Ideal is ~50% from boundaries, ~20% dots
        consistency = 100 - (abs(boundary_contrib - 50) + abs(dot_percentage - 20))
        return max(0, min(100, consistency))

    def analyze_shot_selection(self, shots_data: List[Dict]) -> Dict:
        """
        Analyze shot selection and effectiveness

        Args:
            shots_data: List of dictionaries containing shot information

        Returns:
            Shot selection analysis
        """
        analysis = {
            'total_shots': len(shots_data),
            'shot_distribution': {},
            'zone_distribution': {},
            'effectiveness': {},
            'risk_assessment': {}
        }

        if not shots_data:
            return analysis

        # Count shot types
        for shot in shots_data:
            shot_type = shot.get('shot_type', 'Unknown')
            zone = shot.get('zone', 'Unknown')
            runs = shot.get('runs', 0)

            # Update shot distribution
            analysis['shot_distribution'][shot_type] = \
                analysis['shot_distribution'].get(shot_type, 0) + 1

            # Update zone distribution
            analysis['zone_distribution'][zone] = \
                analysis['zone_distribution'].get(zone, 0) + 1

            # Calculate effectiveness (runs per shot type)
            if shot_type not in analysis['effectiveness']:
                analysis['effectiveness'][shot_type] = {'runs': 0, 'count': 0}

            analysis['effectiveness'][shot_type]['runs'] += runs
            analysis['effectiveness'][shot_type]['count'] += 1

        # Calculate average runs per shot type
        for shot_type in analysis['effectiveness']:
            total_runs = analysis['effectiveness'][shot_type]['runs']
            count = analysis['effectiveness'][shot_type]['count']
            analysis['effectiveness'][shot_type]['average'] = total_runs / count if count > 0 else 0

        # Assess risk
        analysis['risk_assessment'] = self._assess_shot_risk(shots_data)

        return analysis

    def _assess_shot_risk(self, shots_data: List[Dict]) -> Dict:
        """Assess risk level of shot selection"""
        risk_scores = {
            'Cover Drive': 2,
            'Straight Drive': 1,
            'Pull Shot': 5,
            'Cut Shot': 4,
            'Hook Shot': 7,
            'Sweep Shot': 6,
            'Reverse Sweep': 8,
            'Flick Shot': 3,
            'Square Drive': 3,
            'Late Cut': 4,
            'Defense': 1,
            'Leave': 1
        }

        total_risk = 0
        risk_count = 0

        for shot in shots_data:
            shot_type = shot.get('shot_type', 'Unknown')
            if shot_type in risk_scores:
                total_risk += risk_scores[shot_type]
                risk_count += 1

        avg_risk = total_risk / risk_count if risk_count > 0 else 0

        return {
            'average_risk_score': avg_risk,
            'risk_level': 'Low' if avg_risk < 3 else 'Medium' if avg_risk < 6 else 'High',
            'high_risk_shots': risk_count
        }

    def track_performance_over_time(self, innings_data: List[Dict]) -> pd.DataFrame:
        """
        Track batting performance over multiple innings

        Args:
            innings_data: List of innings statistics

        Returns:
            DataFrame with performance trends
        """
        df = pd.DataFrame(innings_data)

        if df.empty:
            return df

        # Add calculated metrics
        if 'runs' in df.columns and 'balls_faced' in df.columns:
            df['strike_rate'] = (df['runs'] / df['balls_faced'] * 100).fillna(0)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        # Calculate rolling averages
        df['avg_runs_5'] = df['runs'].rolling(window=5, min_periods=1).mean()
        df['avg_sr_5'] = df['strike_rate'].rolling(window=5, min_periods=1).mean()

        return df

    def generate_performance_report(self, player_data: Dict) -> str:
        """
        Generate a comprehensive performance report

        Args:
            player_data: Dictionary containing player statistics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("CRICKET BATTING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Player Information
        report.append("PLAYER INFORMATION")
        report.append("-" * 60)
        report.append(f"Name: {player_data.get('name', 'N/A')}")
        report.append(f"Matches Analyzed: {player_data.get('matches', 0)}")
        report.append("")

        # Batting Statistics
        if 'statistics' in player_data:
            stats = player_data['statistics']
            report.append("BATTING STATISTICS")
            report.append("-" * 60)
            report.append(f"Total Runs: {stats.get('runs', 0)}")
            report.append(f"Balls Faced: {stats.get('balls_faced', 0)}")
            report.append(f"Strike Rate: {stats.get('strike_rate', 0):.2f}")
            report.append(f"Boundaries (4s): {stats.get('boundaries', 0)}")
            report.append(f"Sixes: {stats.get('sixes', 0)}")
            report.append(f"Dot Ball %: {stats.get('dot_percentage', 0):.2f}%")
            report.append("")

        # Shot Analysis
        if 'shot_analysis' in player_data:
            shot_data = player_data['shot_analysis']
            report.append("SHOT SELECTION ANALYSIS")
            report.append("-" * 60)
            report.append(f"Total Shots: {shot_data.get('total_shots', 0)}")

            if 'shot_distribution' in shot_data:
                report.append("\nShot Distribution:")
                for shot, count in shot_data['shot_distribution'].items():
                    report.append(f"  {shot}: {count}")

            if 'risk_assessment' in shot_data:
                risk = shot_data['risk_assessment']
                report.append(f"\nRisk Level: {risk.get('risk_level', 'N/A')}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 60)
        recommendations = self._generate_recommendations(player_data)
        for rec in recommendations:
            report.append(f"• {rec}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def _generate_recommendations(self, player_data: Dict) -> List[str]:
        """Generate personalized recommendations based on analysis"""
        recommendations = []

        stats = player_data.get('statistics', {})
        strike_rate = stats.get('strike_rate', 0)
        dot_percentage = stats.get('dot_percentage', 0)

        if strike_rate < 70:
            recommendations.append("Consider increasing scoring rate - practice rotating strike")

        if dot_percentage > 40:
            recommendations.append("High percentage of dot balls - work on shot selection and placement")

        if stats.get('boundary_contribution', 0) > 70:
            recommendations.append("Over-reliant on boundaries - develop singles and doubles game")

        shot_analysis = player_data.get('shot_analysis', {})
        if shot_analysis.get('risk_assessment', {}).get('risk_level') == 'High':
            recommendations.append("Shot selection is high-risk - consider safer alternatives in pressure situations")

        if not recommendations:
            recommendations.append("Maintain current form and consistency")
            recommendations.append("Continue practicing all-round shot selection")

        return recommendations

    def save_analysis(self, data: Dict, filename: str):
        """Save analysis data to JSON file"""
        output_dir = 'analysis_output'
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)

        print(f"Analysis saved to: {filepath}")

    def load_analysis(self, filename: str) -> Dict:
        """Load previously saved analysis"""
        filepath = os.path.join('analysis_output', filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Analysis file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return data


def main():
    """Example usage of the Cricket Batting Analyzer"""
    analyzer = CricketBattingAnalyzer()

    print("Cricket Batting Analysis System")
    print("=" * 60)

    # Example 1: Calculate batting metrics
    print("\n1. Calculating Batting Metrics...")
    metrics = analyzer.calculate_batting_metrics(
        runs=75,
        balls_faced=58,
        boundaries=8,
        sixes=2,
        dots=15
    )
    print(f"Strike Rate: {metrics['strike_rate']:.2f}")
    print(f"Boundary Contribution: {metrics['boundary_contribution']:.2f}%")
    print(f"Scoring Rate: {metrics['scoring_rate']}")

    # Example 2: Analyze shot selection
    print("\n2. Analyzing Shot Selection...")
    sample_shots = [
        {'shot_type': 'Cover Drive', 'zone': 'Off Side', 'runs': 4},
        {'shot_type': 'Pull Shot', 'zone': 'Leg Side', 'runs': 6},
        {'shot_type': 'Defense', 'zone': 'V', 'runs': 0},
        {'shot_type': 'Cover Drive', 'zone': 'Off Side', 'runs': 4},
        {'shot_type': 'Flick Shot', 'zone': 'Leg Side', 'runs': 2},
    ]

    shot_analysis = analyzer.analyze_shot_selection(sample_shots)
    print(f"Total Shots Analyzed: {shot_analysis['total_shots']}")
    print(f"Risk Level: {shot_analysis['risk_assessment']['risk_level']}")

    # Example 3: Generate performance report
    print("\n3. Generating Performance Report...")
    player_data = {
        'name': 'Sample Player',
        'matches': 10,
        'statistics': metrics,
        'shot_analysis': shot_analysis
    }

    report = analyzer.generate_performance_report(player_data)
    print("\n" + report)

    # Save analysis
    analyzer.save_analysis(player_data, 'sample_analysis.json')


if __name__ == "__main__":
    main()
