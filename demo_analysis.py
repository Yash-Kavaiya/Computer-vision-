"""
Demo Script for Cricket Batting Analysis System
Demonstrates how to use all modules together
"""

import numpy as np
from cricket_batting_analyzer import CricketBattingAnalyzer
from visualization import BattingVisualization
from datetime import datetime, timedelta
import random


def generate_sample_innings_data(num_innings: int = 10) -> list:
    """Generate sample innings data for demonstration"""
    innings_data = []
    base_date = datetime.now() - timedelta(days=num_innings * 7)

    for i in range(num_innings):
        runs = random.randint(20, 120)
        balls = random.randint(25, 100)
        boundaries = random.randint(2, 15)
        sixes = random.randint(0, 6)

        innings_data.append({
            'date': (base_date + timedelta(days=i*7)).strftime('%Y-%m-%d'),
            'runs': runs,
            'balls_faced': balls,
            'boundaries': boundaries,
            'sixes': sixes,
            'opponent': f'Team {chr(65 + i % 10)}',
            'venue': random.choice(['Home', 'Away'])
        })

    return innings_data


def generate_sample_shots(num_shots: int = 50) -> list:
    """Generate sample shot data for demonstration"""
    shot_types = [
        'Cover Drive', 'Straight Drive', 'Pull Shot', 'Cut Shot',
        'Hook Shot', 'Sweep Shot', 'Flick Shot', 'Defense'
    ]

    zones = ['V', 'Off Side', 'Leg Side', 'Behind']
    runs_options = [0, 0, 1, 1, 2, 2, 3, 4, 6]

    shots = []
    for _ in range(num_shots):
        shots.append({
            'shot_type': random.choice(shot_types),
            'zone': random.choice(zones),
            'runs': random.choice(runs_options),
            'confidence': random.uniform(0.7, 0.99)
        })

    return shots


def demo_basic_analysis():
    """Demonstrate basic batting analysis"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Batting Analysis")
    print("="*70)

    analyzer = CricketBattingAnalyzer()

    # Calculate metrics for a sample innings
    print("\n📊 Analyzing an innings performance...")
    metrics = analyzer.calculate_batting_metrics(
        runs=85,
        balls_faced=62,
        boundaries=10,
        sixes=3,
        dots=18
    )

    print(f"\n✓ Analysis Complete!")
    print(f"  Runs Scored: {metrics['runs']}")
    print(f"  Strike Rate: {metrics['strike_rate']:.2f}")
    print(f"  Boundary Contribution: {metrics['boundary_contribution']:.2f}%")
    print(f"  Scoring Rate: {metrics['scoring_rate']}")
    print(f"  Consistency Index: {metrics['consistency_index']:.2f}")

    return metrics


def demo_shot_analysis():
    """Demonstrate shot selection analysis"""
    print("\n" + "="*70)
    print("DEMO 2: Shot Selection Analysis")
    print("="*70)

    analyzer = CricketBattingAnalyzer()

    # Generate sample shots
    print("\n🎯 Analyzing shot selection...")
    shots = generate_sample_shots(60)

    shot_analysis = analyzer.analyze_shot_selection(shots)

    print(f"\n✓ Shot Analysis Complete!")
    print(f"  Total Shots: {shot_analysis['total_shots']}")
    print(f"\n  Shot Distribution:")
    for shot_type, count in sorted(shot_analysis['shot_distribution'].items(),
                                   key=lambda x: x[1], reverse=True)[:5]:
        print(f"    • {shot_type}: {count}")

    print(f"\n  Zone Distribution:")
    for zone, count in shot_analysis['zone_distribution'].items():
        print(f"    • {zone}: {count}")

    print(f"\n  Risk Assessment:")
    risk = shot_analysis['risk_assessment']
    print(f"    • Risk Level: {risk['risk_level']}")
    print(f"    • Average Risk Score: {risk['average_risk_score']:.2f}/10")

    return shot_analysis, shots


def demo_performance_tracking():
    """Demonstrate performance tracking over time"""
    print("\n" + "="*70)
    print("DEMO 3: Performance Tracking Over Time")
    print("="*70)

    analyzer = CricketBattingAnalyzer()

    # Generate innings data
    print("\n📈 Tracking performance across 10 innings...")
    innings_data = generate_sample_innings_data(10)

    df = analyzer.track_performance_over_time(innings_data)

    print(f"\n✓ Performance Tracking Complete!")
    print(f"\n  Recent Innings (Last 5):")
    print(df[['runs', 'balls_faced', 'strike_rate']].tail().to_string(index=False))

    print(f"\n  Statistical Summary:")
    print(f"    • Average Runs: {df['runs'].mean():.2f}")
    print(f"    • Average Strike Rate: {df['strike_rate'].mean():.2f}")
    print(f"    • Best Score: {df['runs'].max()}")
    print(f"    • Total Boundaries: {df['boundaries'].sum()}")
    print(f"    • Total Sixes: {df['sixes'].sum()}")

    return df


def demo_performance_report():
    """Demonstrate comprehensive performance report generation"""
    print("\n" + "="*70)
    print("DEMO 4: Comprehensive Performance Report")
    print("="*70)

    analyzer = CricketBattingAnalyzer()

    # Prepare complete player data
    print("\n📋 Generating detailed performance report...")

    metrics = analyzer.calculate_batting_metrics(
        runs=450,
        balls_faced=385,
        boundaries=48,
        sixes=12,
        dots=95
    )

    shots = generate_sample_shots(385)
    shot_analysis = analyzer.analyze_shot_selection(shots)

    player_data = {
        'name': 'Demo Player',
        'matches': 10,
        'statistics': metrics,
        'shot_analysis': shot_analysis
    }

    report = analyzer.generate_performance_report(player_data)
    print(report)

    # Save the analysis
    analyzer.save_analysis(player_data, 'demo_player_analysis.json')

    return player_data


def demo_visualizations():
    """Demonstrate visualization capabilities"""
    print("\n" + "="*70)
    print("DEMO 5: Creating Visualizations")
    print("="*70)

    viz = BattingVisualization()
    analyzer = CricketBattingAnalyzer()

    print("\n🎨 Generating visualizations...")

    # Prepare data
    metrics = analyzer.calculate_batting_metrics(
        runs=95,
        balls_faced=71,
        boundaries=11,
        sixes=4,
        dots=22
    )

    shots = generate_sample_shots(71)
    shot_analysis = analyzer.analyze_shot_selection(shots)

    innings_data = generate_sample_innings_data(15)
    df = analyzer.track_performance_over_time(innings_data)

    # Generate visualizations
    print("\n  Creating batting metrics dashboard...")
    viz.plot_batting_metrics(metrics)

    print("  Creating shot distribution charts...")
    viz.plot_shot_distribution(shot_analysis)

    print("  Creating performance trend analysis...")
    viz.plot_performance_trend(df)

    print("  Creating wagon wheel visualization...")
    viz.create_wagon_wheel(shots)

    print(f"\n✓ All visualizations saved to: {viz.output_dir}/")


def demo_complete_workflow():
    """Demonstrate complete analysis workflow"""
    print("\n" + "="*70)
    print("DEMO 6: Complete Analysis Workflow")
    print("="*70)

    analyzer = CricketBattingAnalyzer()
    viz = BattingVisualization()

    print("\n🔄 Running complete analysis workflow...\n")

    # Step 1: Collect data
    print("Step 1: Collecting match data...")
    innings_data = generate_sample_innings_data(10)
    print(f"  ✓ Collected data from {len(innings_data)} innings")

    # Step 2: Calculate comprehensive metrics
    print("\nStep 2: Calculating comprehensive metrics...")
    total_runs = sum(i['runs'] for i in innings_data)
    total_balls = sum(i['balls_faced'] for i in innings_data)
    total_boundaries = sum(i['boundaries'] for i in innings_data)
    total_sixes = sum(i['sixes'] for i in innings_data)

    metrics = analyzer.calculate_batting_metrics(
        runs=total_runs,
        balls_faced=total_balls,
        boundaries=total_boundaries,
        sixes=total_sixes,
        dots=int(total_balls * 0.25)  # Estimate 25% dot balls
    )
    print("  ✓ Metrics calculated")

    # Step 3: Analyze shot selection
    print("\nStep 3: Analyzing shot selection...")
    shots = generate_sample_shots(total_balls)
    shot_analysis = analyzer.analyze_shot_selection(shots)
    print("  ✓ Shot selection analyzed")

    # Step 4: Track performance trends
    print("\nStep 4: Tracking performance trends...")
    df = analyzer.track_performance_over_time(innings_data)
    print("  ✓ Performance trends identified")

    # Step 5: Generate report
    print("\nStep 5: Generating comprehensive report...")
    player_data = {
        'name': 'Complete Analysis Demo',
        'matches': len(innings_data),
        'statistics': metrics,
        'shot_analysis': shot_analysis
    }
    report = analyzer.generate_performance_report(player_data)
    print("  ✓ Report generated")

    # Step 6: Create visualizations
    print("\nStep 6: Creating visualizations...")
    complete_data = {
        'metrics': metrics,
        'shot_analysis': shot_analysis,
        'innings_data': innings_data,
        'shots': shots
    }
    viz.generate_visual_report(complete_data)
    print("  ✓ Visualizations created")

    # Step 7: Save everything
    print("\nStep 7: Saving analysis results...")
    analyzer.save_analysis(player_data, 'complete_workflow_analysis.json')
    print("  ✓ Analysis saved")

    print("\n" + "="*70)
    print("🎉 Complete workflow finished successfully!")
    print("="*70)
    print(f"\nResults saved in:")
    print(f"  • JSON: analysis_output/complete_workflow_analysis.json")
    print(f"  • Visualizations: visualizations/")


def main():
    """Main demo function"""
    print("\n" + "="*70)
    print(" "*15 + "CRICKET BATTING ANALYSIS SYSTEM")
    print(" "*20 + "Demonstration Suite")
    print("="*70)

    print("\nThis demo showcases the capabilities of the Cricket Batting")
    print("Analysis System using computer vision and statistical analysis.\n")

    demos = [
        ("Basic Batting Analysis", demo_basic_analysis),
        ("Shot Selection Analysis", demo_shot_analysis),
        ("Performance Tracking", demo_performance_tracking),
        ("Performance Report", demo_performance_report),
        ("Visualizations", demo_visualizations),
        ("Complete Workflow", demo_complete_workflow)
    ]

    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\n" + "-"*70)

    try:
        # Run all demos
        for name, demo_func in demos:
            demo_func()
            print("\n" + "-"*70)

        print("\n✅ All demonstrations completed successfully!")
        print("\nGenerated Files:")
        print("  • analysis_output/demo_player_analysis.json")
        print("  • analysis_output/complete_workflow_analysis.json")
        print("  • visualizations/batting_metrics.png")
        print("  • visualizations/shot_distribution.png")
        print("  • visualizations/performance_trend.png")
        print("  • visualizations/wagon_wheel.png")

        print("\n" + "="*70)
        print("Thank you for exploring the Cricket Batting Analysis System!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
