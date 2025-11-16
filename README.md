# Cricket Batting Analysis System 🏏

A comprehensive computer vision and statistical analysis system for analyzing cricket batting performance. This system uses advanced image processing, pose detection, and data analytics to provide detailed insights into batting technique, shot selection, and performance metrics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules Overview](#modules-overview)
- [Usage Examples](#usage-examples)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🎯 Core Capabilities

- **Video Analysis**: Real-time analysis of batting technique from video footage
- **Pose Detection**: Automated detection of batting stance and body positioning
- **Shot Classification**: Identify and classify different types of cricket shots
- **Statistical Analysis**: Comprehensive batting statistics and metrics calculation
- **Performance Tracking**: Track performance trends over multiple innings
- **Visual Reports**: Generate detailed visualizations and dashboards

### 📊 Analysis Metrics

- Strike rate and scoring patterns
- Boundary contribution analysis
- Shot selection effectiveness
- Risk assessment of shot choices
- Consistency index calculation
- Zone-wise scoring distribution
- Performance trends over time

### 🎨 Visualizations

- Batting metrics dashboard
- Shot distribution charts
- Performance trend graphs
- Wagon wheel visualization
- Effectiveness heatmaps
- Player comparison radar charts

## 🏗️ System Architecture

```
Cricket-Batting-Analysis/
├── cricket_batting_analyzer.py  # Main analysis engine
├── video_analysis.py            # Video processing & pose detection
├── visualization.py             # Chart and graph generation
├── demo_analysis.py             # Demonstration and examples
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

### Module Dependencies

```
┌─────────────────────────────┐
│   demo_analysis.py          │
│   (Orchestration Layer)     │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┬────────────────┐
    │             │                │
┌───▼────┐  ┌────▼─────┐  ┌──────▼──────┐
│ Cricket│  │  Video   │  │Visualization│
│Analyzer│  │ Analysis │  │   Module    │
└────────┘  └──────────┘  └─────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam or video files for analysis (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/cricket-batting-analysis.git
cd cricket-batting-analysis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python cricket_batting_analyzer.py
```

## 🎬 Quick Start

### Run the Demo

The easiest way to get started is to run the demonstration script:

```bash
python demo_analysis.py
```

This will:
- Generate sample batting data
- Perform comprehensive analysis
- Create visualizations
- Save results to output directories

### Basic Usage Example

```python
from cricket_batting_analyzer import CricketBattingAnalyzer

# Initialize analyzer
analyzer = CricketBattingAnalyzer()

# Calculate batting metrics
metrics = analyzer.calculate_batting_metrics(
    runs=75,
    balls_faced=58,
    boundaries=8,
    sixes=2,
    dots=15
)

print(f"Strike Rate: {metrics['strike_rate']:.2f}")
print(f"Boundary Contribution: {metrics['boundary_contribution']:.2f}%")
```

## 📚 Modules Overview

### 1. Cricket Batting Analyzer (`cricket_batting_analyzer.py`)

The main analysis engine that provides:

**Key Classes:**
- `CricketBattingAnalyzer`: Main class for batting analysis

**Key Methods:**
- `calculate_batting_metrics()`: Calculate comprehensive statistics
- `analyze_shot_selection()`: Analyze shot types and effectiveness
- `track_performance_over_time()`: Monitor trends across innings
- `generate_performance_report()`: Create detailed text reports

### 2. Video Analysis (`video_analysis.py`)

Computer vision module for video processing:

**Key Classes:**
- `BattingVideoAnalyzer`: Video processing and pose detection

**Key Methods:**
- `load_video()`: Load video file or webcam stream
- `analyze_frame()`: Analyze individual frames
- `detect_shot_event()`: Identify when shots are played
- `process_video()`: Complete video analysis pipeline

### 3. Visualization (`visualization.py`)

Creates visual representations of analysis:

**Key Classes:**
- `BattingVisualization`: Chart and graph generation

**Key Methods:**
- `plot_batting_metrics()`: Create metrics dashboard
- `plot_shot_distribution()`: Visualize shot selection
- `plot_performance_trend()`: Show performance over time
- `create_wagon_wheel()`: Generate wagon wheel chart

## 💡 Usage Examples

### Example 1: Analyze Single Innings

```python
from cricket_batting_analyzer import CricketBattingAnalyzer
from visualization import BattingVisualization

# Initialize
analyzer = CricketBattingAnalyzer()
viz = BattingVisualization()

# Analyze innings
metrics = analyzer.calculate_batting_metrics(
    runs=85,
    balls_faced=62,
    boundaries=10,
    sixes=3,
    dots=18
)

# Create visualization
viz.plot_batting_metrics(metrics)
```

### Example 2: Track Performance Over Time

```python
from cricket_batting_analyzer import CricketBattingAnalyzer

analyzer = CricketBattingAnalyzer()

# Sample innings data
innings_data = [
    {'date': '2024-01-01', 'runs': 45, 'balls_faced': 38, 'boundaries': 5, 'sixes': 1},
    {'date': '2024-01-08', 'runs': 62, 'balls_faced': 47, 'boundaries': 7, 'sixes': 2},
    {'date': '2024-01-15', 'runs': 78, 'balls_faced': 55, 'boundaries': 9, 'sixes': 3},
]

# Track performance
df = analyzer.track_performance_over_time(innings_data)
print(df)
```

### Example 3: Video Analysis

```python
from video_analysis import BattingVideoAnalyzer

analyzer = BattingVideoAnalyzer()

# Process video file
results = analyzer.process_video(
    video_path='batting_session.mp4',
    output_path='analyzed_output.mp4',
    show_preview=True
)

# Save analysis
analyzer.save_analysis_results(results, 'video_analysis.json')
```

### Example 4: Shot Selection Analysis

```python
from cricket_batting_analyzer import CricketBattingAnalyzer
from visualization import BattingVisualization

analyzer = CricketBattingAnalyzer()
viz = BattingVisualization()

# Define shots played
shots = [
    {'shot_type': 'Cover Drive', 'zone': 'Off Side', 'runs': 4},
    {'shot_type': 'Pull Shot', 'zone': 'Leg Side', 'runs': 6},
    {'shot_type': 'Defense', 'zone': 'V', 'runs': 0},
    {'shot_type': 'Flick Shot', 'zone': 'Leg Side', 'runs': 2},
]

# Analyze shots
analysis = analyzer.analyze_shot_selection(shots)

# Visualize
viz.plot_shot_distribution(analysis)
viz.create_wagon_wheel(shots)
```

### Example 5: Generate Complete Report

```python
from cricket_batting_analyzer import CricketBattingAnalyzer

analyzer = CricketBattingAnalyzer()

# Prepare player data
player_data = {
    'name': 'Player Name',
    'matches': 10,
    'statistics': {
        'runs': 450,
        'balls_faced': 385,
        'strike_rate': 116.88,
        'boundaries': 48,
        'sixes': 12
    }
}

# Generate report
report = analyzer.generate_performance_report(player_data)
print(report)
```

## 📂 Output Files

The system generates various output files:

### Analysis Output (`analysis_output/`)
- `*.json`: Detailed analysis data in JSON format
- Contains metrics, shot analysis, and recommendations

### Visualizations (`visualizations/`)
- `batting_metrics.png`: Comprehensive metrics dashboard
- `shot_distribution.png`: Shot selection charts
- `performance_trend.png`: Performance over time
- `wagon_wheel.png`: Shot direction visualization
- `effectiveness_heatmap.png`: Zone-wise effectiveness

## 🔧 Advanced Usage

### Custom Shot Classification

```python
from cricket_batting_analyzer import CricketBattingAnalyzer

analyzer = CricketBattingAnalyzer()

# Add custom shot types
analyzer.shot_types.extend(['Custom Shot', 'Special Technique'])

# Define custom zones
analyzer.batting_zones['Custom Zone'] = (45, 90)
```

### Real-time Webcam Analysis

```python
from video_analysis import BattingVideoAnalyzer

analyzer = BattingVideoAnalyzer()

# Use webcam (0 = default camera)
analyzer.load_video(0)

# Process in real-time
analyzer.process_video(0, show_preview=True)
```

### Batch Processing Multiple Videos

```python
from video_analysis import BattingVideoAnalyzer
import os

analyzer = BattingVideoAnalyzer()

video_dir = 'videos/'
output_dir = 'processed/'

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        input_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f'analyzed_{video_file}')

        results = analyzer.process_video(input_path, output_path)
        analyzer.save_analysis_results(results, f'{video_file}_analysis.json')
```

## 📊 Performance Metrics Explained

### Strike Rate
- Formula: `(Runs / Balls Faced) × 100`
- Indicates scoring speed
- Higher is generally better (context-dependent)

### Boundary Contribution
- Formula: `(Runs from Boundaries / Total Runs) × 100`
- Shows reliance on boundaries vs. rotation of strike
- Ideal range: 40-60%

### Consistency Index
- Scale: 0-100
- Measures balanced scoring approach
- Considers boundary percentage and dot ball percentage
- Higher indicates more consistent batting

### Risk Assessment
- Evaluates shot selection risk
- Scores: 1 (Low) to 10 (High)
- Based on shot types played
- Helps identify aggressive vs. conservative approach

## 🎓 Key Concepts

### Shot Types Recognized
1. **Cover Drive**: Classic off-side drive
2. **Straight Drive**: Down the ground
3. **Pull Shot**: Short ball to leg side
4. **Cut Shot**: Short ball to off side
5. **Hook Shot**: Short ball, high risk
6. **Sweep Shot**: Against spin
7. **Reverse Sweep**: Unconventional
8. **Flick Shot**: Wristy leg side
9. **Defense**: Defensive block
10. **Leave**: Leaving the ball

### Batting Zones
- **V Zone** (0-45°): Straight shots
- **Off Side** (45-135°): Off side shots
- **Leg Side** (-135 to -45°): Leg side shots
- **Behind** (135-180°): Behind wicket

## 🐛 Troubleshooting

### Common Issues

**Issue**: OpenCV not working
```bash
# Solution: Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

**Issue**: Visualization not displaying
```bash
# Solution: Install GUI backend for matplotlib
pip install PyQt5
```

**Issue**: Video file not loading
- Ensure video codec is supported (MP4, AVI recommended)
- Check file path is correct
- Try converting video with FFmpeg

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Cricket analytics community for domain knowledge
- Contributors and testers

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] ML-based shot classification using deep learning
- [ ] Ball trajectory prediction
- [ ] Bowler analysis integration
- [ ] Real-time match situation analysis
- [ ] Mobile app integration
- [ ] Cloud-based batch processing
- [ ] 3D pose estimation
- [ ] Multi-player tracking
- [ ] Automated highlight generation
- [ ] Integration with ball-tracking systems

---

**Made with ❤️ for Cricket Analytics**
