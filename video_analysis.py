"""
Video Analysis Module for Cricket Batting
Uses computer vision to analyze batting technique from video footage
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class BattingVideoAnalyzer:
    """
    Analyze cricket batting technique from video using computer vision
    """

    def __init__(self):
        """Initialize video analyzer"""
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.analysis_results = []

        # Initialize pose detection parameters
        self.pose_initialized = False

    def load_video(self, video_path: str) -> bool:
        """
        Load video file for analysis

        Args:
            video_path: Path to video file

        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Video loaded successfully:")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  FPS: {self.fps}")
        print(f"  Duration: {self.total_frames / self.fps:.2f} seconds")

        return True

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for batting technique

        Args:
            frame: Video frame as numpy array

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'frame_number': self.current_frame,
            'timestamp': self.current_frame / self.fps,
            'detections': {}
        }

        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect motion
        motion_analysis = self._detect_motion(frame)
        analysis['motion'] = motion_analysis

        # Detect bat
        bat_detection = self._detect_bat(frame, hsv)
        analysis['bat'] = bat_detection

        # Detect player (simplified contour detection)
        player_detection = self._detect_player(frame, gray)
        analysis['player'] = player_detection

        # Analyze stance/posture
        posture = self._analyze_posture(frame, player_detection)
        analysis['posture'] = posture

        return analysis

    def _detect_motion(self, frame: np.ndarray) -> Dict:
        """Detect motion in the frame"""
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return {'motion_detected': False, 'intensity': 0}

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, current_gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Calculate motion intensity
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100

        self.prev_frame = current_gray

        return {
            'motion_detected': motion_percentage > 1.0,
            'intensity': float(motion_percentage),
            'motion_level': 'High' if motion_percentage > 10 else 'Medium' if motion_percentage > 5 else 'Low'
        }

    def _detect_bat(self, frame: np.ndarray, hsv: np.ndarray) -> Dict:
        """
        Detect cricket bat in the frame

        Args:
            frame: Original frame
            hsv: HSV color space frame

        Returns:
            Bat detection results
        """
        # Bat is typically brown/wooden color or white (willow)
        # Define color ranges for bat detection
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 200])

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        # Create masks
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_or(mask_brown, mask_white)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bat_detected = False
        bat_angle = 0
        bat_position = None

        if contours:
            # Find largest contour (likely the bat)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 500:  # Minimum area threshold
                bat_detected = True

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                bat_position = (x + w // 2, y + h // 2)

                # Estimate bat angle
                if h > w:
                    bat_angle = self._calculate_angle(largest_contour)

        return {
            'detected': bat_detected,
            'position': bat_position,
            'angle': float(bat_angle),
            'orientation': self._get_bat_orientation(bat_angle)
        }

    def _calculate_angle(self, contour) -> float:
        """Calculate angle of bat from contour"""
        if len(contour) < 5:
            return 0.0

        # Fit ellipse to get orientation
        ellipse = cv2.fitEllipse(contour)
        angle = ellipse[2]
        return angle

    def _get_bat_orientation(self, angle: float) -> str:
        """Determine bat orientation from angle"""
        if 80 <= angle <= 100:
            return 'Vertical'
        elif 170 <= angle or angle <= 10:
            return 'Horizontal'
        elif 30 <= angle <= 60:
            return 'Diagonal_Down'
        elif 120 <= angle <= 150:
            return 'Diagonal_Up'
        else:
            return 'Angled'

    def _detect_player(self, frame: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect player in the frame"""
        # Use edge detection to find player outline
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        player_detected = False
        player_bbox = None
        player_centroid = None

        if contours:
            # Find largest contour (likely the player)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 5000:  # Minimum area threshold
                player_detected = True
                x, y, w, h = cv2.boundingRect(largest_contour)
                player_bbox = (x, y, w, h)
                player_centroid = (x + w // 2, y + h // 2)

        return {
            'detected': player_detected,
            'bounding_box': player_bbox,
            'centroid': player_centroid,
            'height': player_bbox[3] if player_bbox else 0
        }

    def _analyze_posture(self, frame: np.ndarray, player_detection: Dict) -> Dict:
        """Analyze batting posture"""
        posture_analysis = {
            'stance': 'Unknown',
            'balance': 'Unknown',
            'head_position': 'Unknown',
            'weight_distribution': 'Even'
        }

        if not player_detection['detected']:
            return posture_analysis

        bbox = player_detection['bounding_box']
        if bbox is None:
            return posture_analysis

        x, y, w, h = bbox

        # Analyze stance based on bounding box aspect ratio
        aspect_ratio = h / w if w > 0 else 0

        if aspect_ratio > 2.5:
            posture_analysis['stance'] = 'Upright'
        elif aspect_ratio > 2.0:
            posture_analysis['stance'] = 'Ready'
        else:
            posture_analysis['stance'] = 'Crouched'

        # Analyze balance based on centroid position
        centroid = player_detection['centroid']
        if centroid:
            frame_center_x = frame.shape[1] // 2
            offset = abs(centroid[0] - frame_center_x)

            if offset < 50:
                posture_analysis['balance'] = 'Balanced'
            elif offset < 100:
                posture_analysis['balance'] = 'Slight_Lean'
            else:
                posture_analysis['balance'] = 'Off_Balance'

        return posture_analysis

    def detect_shot_event(self, frames: List[np.ndarray]) -> Dict:
        """
        Detect when a shot is played from a sequence of frames

        Args:
            frames: List of consecutive frames

        Returns:
            Shot event detection results
        """
        if len(frames) < 2:
            return {'shot_detected': False}

        # Analyze motion intensity across frames
        motion_intensities = []

        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            intensity = np.mean(diff)
            motion_intensities.append(intensity)

        # Shot is detected when there's a sudden spike in motion
        avg_motion = np.mean(motion_intensities)
        max_motion = np.max(motion_intensities)

        shot_detected = max_motion > avg_motion * 2

        return {
            'shot_detected': shot_detected,
            'peak_motion_frame': int(np.argmax(motion_intensities)),
            'motion_intensity': float(max_motion),
            'avg_motion': float(avg_motion)
        }

    def extract_shot_sequence(self, start_frame: int, duration_frames: int = 30) -> List[np.ndarray]:
        """
        Extract a sequence of frames around a shot

        Args:
            start_frame: Starting frame number
            duration_frames: Number of frames to extract

        Returns:
            List of frames
        """
        frames = []

        # Set video to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(duration_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)

        return frames

    def annotate_frame(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """
        Annotate frame with analysis results

        Args:
            frame: Original frame
            analysis: Analysis results

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Add frame info
        cv2.putText(annotated, f"Frame: {analysis.get('frame_number', 0)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add motion info
        if 'motion' in analysis:
            motion = analysis['motion']
            color = (0, 255, 0) if motion['motion_detected'] else (0, 0, 255)
            cv2.putText(annotated, f"Motion: {motion['motion_level']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw bat detection
        if 'bat' in analysis and analysis['bat']['detected']:
            bat_pos = analysis['bat']['position']
            if bat_pos:
                cv2.circle(annotated, bat_pos, 10, (0, 255, 255), -1)
                cv2.putText(annotated, f"Bat: {analysis['bat']['orientation']}",
                            (bat_pos[0] + 15, bat_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw player detection
        if 'player' in analysis and analysis['player']['detected']:
            bbox = analysis['player']['bounding_box']
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add posture info
        if 'posture' in analysis:
            posture = analysis['posture']
            cv2.putText(annotated, f"Stance: {posture['stance']}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Balance: {posture['balance']}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_preview: bool = False) -> List[Dict]:
        """
        Process entire video and analyze batting

        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            show_preview: Whether to show live preview

        Returns:
            List of frame analysis results
        """
        if not self.load_video(video_path):
            return []

        results = []
        frame_skip = 2  # Process every nth frame for efficiency

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (frame_width, frame_height))

        print("\nProcessing video...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame += 1

            # Skip frames for efficiency
            if self.current_frame % frame_skip != 0:
                continue

            # Analyze frame
            analysis = self.analyze_frame(frame)
            results.append(analysis)

            # Annotate frame
            annotated = self.annotate_frame(frame, analysis)

            # Write to output
            if writer:
                writer.write(annotated)

            # Show preview
            if show_preview:
                cv2.imshow('Batting Analysis', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress indicator
            if self.current_frame % 30 == 0:
                progress = (self.current_frame / self.total_frames) * 100
                print(f"Progress: {progress:.1f}%", end='\r')

        print(f"\nProcessing complete. Analyzed {len(results)} frames.")

        # Cleanup
        self.cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        return results

    def save_analysis_results(self, results: List[Dict], filepath: str):
        """Save analysis results to JSON file"""
        output_data = {
            'metadata': {
                'total_frames_analyzed': len(results),
                'fps': self.fps,
                'analysis_date': datetime.now().isoformat()
            },
            'results': results
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4, default=str)

        print(f"Analysis results saved to: {filepath}")


def main():
    """Example usage of video analyzer"""
    analyzer = BattingVideoAnalyzer()

    print("Cricket Batting Video Analyzer")
    print("=" * 60)
    print("\nThis module analyzes cricket batting technique from video.")
    print("\nFeatures:")
    print("  • Motion detection and tracking")
    print("  • Bat detection and angle analysis")
    print("  • Player posture and stance analysis")
    print("  • Shot event detection")
    print("  • Frame-by-frame annotation")
    print("\nUsage:")
    print("  analyzer = BattingVideoAnalyzer()")
    print("  results = analyzer.process_video('batting.mp4', 'output.mp4')")
    print("\nFor webcam/live analysis:")
    print("  analyzer.load_video(0)  # 0 for default webcam")


if __name__ == "__main__":
    main()
