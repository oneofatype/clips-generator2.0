"""
Podcast Video to Vertical Shorts Converter (V3 - Production Ready)
===================================================================

Converts horizontal (16:9) podcast videos into vertical (9:16) shorts by:
1. Detecting scenes/cuts with PySceneDetect
2. Detecting faces with InsightFace (handles side profiles well)
3. Detecting mouth movement from facial landmarks (speaking detection)
4. Smooth camera tracking with Kalman filter
5. Instant snap on scene changes, smooth transitions otherwise

Requirements:
    pip install opencv-python insightface onnxruntime numpy ultralytics scenedetect filterpy

FFMPEG must be in PATH for final encoding.

Usage:
    python face_tracker.py input.mp4 output.mp4 [--debug] [--preview]
"""

import cv2
import numpy as np
import subprocess
import sys
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum, auto

# Third-party imports - InsightFace for robust face detection
# Works much better with side profiles than MediaPipe

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Central configuration for all processing parameters."""
    
    # === Output Format ===
    output_aspect_ratio: float = 9 / 16  # Vertical shorts format
    output_resolution: Tuple[int, int] = (1080, 1920)  # Width x Height for output
    
    # === InsightFace - Face Detection with Side Profile Support ===
    insightface_det_size: Tuple[int, int] = (640, 640)  # Detection size
    insightface_det_thresh: float = 0.3  # Detection threshold (lower = more sensitive)
    insightface_max_faces: int = 4  # Max faces to track
    
    # Face detection frequency
    detection_interval: int = 1  # Detect every N frames (1 = every frame for accuracy)
    
    # === Speaking Detection ===
    # Mouth Aspect Ratio (MAR) parameters
    mar_history_length: int = 20  # Frames to analyze for speaking
    mar_variance_threshold: float = 0.005  # Variance threshold (increased to filter head movements)
    mar_min_samples: int = 8  # Minimum samples before making speaking decision
    
    # Speaking confirmation/debounce
    speaking_confirm_frames: int = 4  # Frames of speaking before confirmed
    speaking_hold_frames: int = 60  # Hold speaking state during pauses (~2s at 30fps)
    
    # === Scene Detection ===
    scene_threshold: float = 27.0  # Content-aware threshold for scene detection
    scene_min_length: int = 15  # Minimum frames between scene cuts
    
    # Backup pixel-diff detection (in case PySceneDetect unavailable)
    pixel_diff_threshold: float = 25.0
    
    # === Camera Tracking ===
    # Kalman filter noise parameters
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 1e-2
    
    # EMA fallback (if Kalman disabled)
    ema_alpha_smooth: float = 0.08  # Slow smooth tracking
    ema_alpha_fast: float = 0.25   # Fast catch-up
    ema_alpha_instant: float = 1.0  # Instant snap
    
    # Position jump detection (camera angle change within scene)
    position_jump_threshold: float = 0.20  # 20% of frame width
    
    # === Frame Positioning ===
    # Where to position face in output frame
    face_vertical_position: float = 0.35  # Face center at 35% from top
    face_horizontal_position: float = 0.50  # Face at center horizontally
    
    # === Face Filtering ===
    min_face_size_ratio: float = 0.08  # Min face size relative to frame height
    max_face_size_ratio: float = 0.60  # Max face size (filter false positives)
    
    # === Speaker Selection ===
    # Priority weights for speaker selection
    weight_speaking: float = 15.0  # Weight for actively speaking (increased for priority)
    weight_face_size: float = 2.0  # Weight for face size (reduced to not override speaking)
    weight_center: float = 1.0     # Weight for being centered
    weight_current_speaking: float = 5.0    # Hysteresis when current speaker IS speaking
    weight_current_silent: float = 1.0      # Reduced hysteresis when current speaker is silent
    
    # Switch delay to prevent rapid speaker changes
    speaker_switch_delay: int = 8  # Frames before switching (normal)
    speaker_switch_delay_fast: int = 3  # Frames when someone else is clearly speaking
    
    # Focus persistence - keep focus on someone who was recently speaking
    recent_speaker_hold_frames: int = 90  # ~3 seconds to hold focus on recent speaker
    recent_speaker_bonus: float = 8.0  # Bonus for the person who was recently speaking
    
    # === Motion Detection (fallback when mouth tracking unavailable) ===
    motion_history_length: int = 15  # Frames to analyze for motion
    motion_variance_threshold: float = 8.0  # Pixel variance threshold for movement
    weight_motion: float = 10.0  # Weight for motion when no mouth data available
    no_mesh_penalty: float = 3.0  # Penalty for faces without mouth tracking (reduced from scoring)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def euclidean_distance_pts(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points as numpy arrays [x, y]."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_mar_insightface(landmarks_2d: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) from InsightFace 2D landmarks.
    Higher MAR = mouth more open.
    
    InsightFace 2D landmarks (106 points) mouth indices:
    - Upper lip: 86, 87, 88 (center top)
    - Lower lip: 92, 93, 94 (center bottom)  
    - Left mouth corner: 84
    - Right mouth corner: 90
    
    For 5-point landmarks, we estimate from nose/eye positions.
    """
    if landmarks_2d is None:
        return 0.0
    
    num_landmarks = len(landmarks_2d)
    
    if num_landmarks >= 106:
        # Full 106-point landmarks - precise mouth tracking
        upper_lip = landmarks_2d[87]  # Upper lip center
        lower_lip = landmarks_2d[93]  # Lower lip center
        left_corner = landmarks_2d[84]  # Left mouth corner
        right_corner = landmarks_2d[90]  # Right mouth corner
        
        vertical_dist = euclidean_distance_pts(upper_lip, lower_lip)
        horizontal_dist = euclidean_distance_pts(left_corner, right_corner)
        
        if horizontal_dist < 1.0:
            return 0.0
        
        return vertical_dist / horizontal_dist
    
    elif num_landmarks >= 68:
        # 68-point landmarks (dlib-style) 
        # Upper lip: 51, 62, 63 | Lower lip: 57, 66, 67 | Corners: 48, 54
        upper_lip = landmarks_2d[62]  # Upper lip inner
        lower_lip = landmarks_2d[66]  # Lower lip inner
        left_corner = landmarks_2d[48]
        right_corner = landmarks_2d[54]
        
        vertical_dist = euclidean_distance_pts(upper_lip, lower_lip)
        horizontal_dist = euclidean_distance_pts(left_corner, right_corner)
        
        if horizontal_dist < 1.0:
            return 0.0
        
        return vertical_dist / horizontal_dist
    
    elif num_landmarks >= 5:
        # Basic 5-point landmarks (just eyes, nose, mouth corners)
        # Indices: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
        # Can only estimate mouth width, not opening - return small value
        # We'll rely more on motion detection in this case
        return 0.05  # Small constant - face detected but no mouth detail
    
    return 0.0


def calculate_face_bbox_from_insightface(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate face center and size from InsightFace bbox.
    InsightFace returns [x1, y1, x2, y2, score]
    Returns: (center_x, center_y, width, height) in pixels
    """
    x1, y1, x2, y2 = bbox[:4]
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    return center_x, center_y, width, height


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SpeakingState(Enum):
    """Speaking state machine states."""
    SILENT = auto()
    MAYBE_SPEAKING = auto()
    SPEAKING = auto()
    MAYBE_SILENT = auto()


@dataclass
class TrackedFace:
    """Represents a tracked face with all associated data."""
    
    # Identity
    id: int
    
    # Position (pixels)
    center_x: float
    center_y: float
    width: float
    height: float
    
    # Detection source
    from_yolo: bool = False
    from_mesh: bool = False
    confidence: float = 1.0
    
    # Mouth/Speaking detection
    mar: float = 0.0
    mar_history: deque = field(default_factory=lambda: deque(maxlen=25))
    speaking_state: SpeakingState = SpeakingState.SILENT
    speaking_state_frames: int = 0  # Frames in current state
    speaking_score: float = 0.0
    frames_since_last_spoke: int = 999  # Frames since this face was speaking
    total_speaking_frames: int = 0  # Cumulative frames this face has been speaking
    
    # Tracking
    frames_since_seen: int = 0
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    motion_score: float = 0.0  # Movement detection score
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def is_moving(self) -> bool:
        """True if this face shows significant movement (head motion)."""
        return self.motion_score > 8.0  # Use threshold from config ideally
    
    def calculate_motion(self):
        """Calculate motion score based on position history variance."""
        if len(self.position_history) < 5:
            self.motion_score = 0.0
            return
        
        positions = list(self.position_history)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # Calculate variance of recent positions
        var_x = np.var(xs)
        var_y = np.var(ys)
        
        # Combined motion score
        self.motion_score = np.sqrt(var_x + var_y)
    
    @property
    def is_speaking(self) -> bool:
        return self.speaking_state in (SpeakingState.SPEAKING, SpeakingState.MAYBE_SILENT)
    
    @property
    def was_recently_speaking(self) -> bool:
        """True if this face was speaking recently (within ~3 seconds)."""
        return self.frames_since_last_spoke < 90
    
    def update_mar(self, new_mar: float):
        """Update mouth aspect ratio and history."""
        self.mar = new_mar
        self.mar_history.append(new_mar)
    
    def update_speaking_state(self, config: Config):
        """
        State machine for speaking detection.
        Uses MAR variance to detect speaking with debouncing.
        """
        if len(self.mar_history) < config.mar_min_samples:
            self.speaking_score = 0.0
            self.frames_since_last_spoke += 1
            return
        
        # Calculate variance of recent mouth openings
        variance = np.var(list(self.mar_history))
        self.speaking_score = variance
        
        is_mouth_moving = variance > config.mar_variance_threshold
        
        # Track when this face was last speaking
        if self.speaking_state == SpeakingState.SPEAKING:
            self.frames_since_last_spoke = 0
            self.total_speaking_frames += 1
        else:
            self.frames_since_last_spoke += 1
        
        # State machine transitions
        prev_state = self.speaking_state
        
        if self.speaking_state == SpeakingState.SILENT:
            if is_mouth_moving:
                self.speaking_state = SpeakingState.MAYBE_SPEAKING
                self.speaking_state_frames = 1
        
        elif self.speaking_state == SpeakingState.MAYBE_SPEAKING:
            if is_mouth_moving:
                self.speaking_state_frames += 1
                if self.speaking_state_frames >= config.speaking_confirm_frames:
                    self.speaking_state = SpeakingState.SPEAKING
                    self.speaking_state_frames = 0
            else:
                self.speaking_state = SpeakingState.SILENT
                self.speaking_state_frames = 0
        
        elif self.speaking_state == SpeakingState.SPEAKING:
            if not is_mouth_moving:
                self.speaking_state = SpeakingState.MAYBE_SILENT
                self.speaking_state_frames = 1
        
        elif self.speaking_state == SpeakingState.MAYBE_SILENT:
            if is_mouth_moving:
                self.speaking_state = SpeakingState.SPEAKING
                self.speaking_state_frames = 0
            else:
                self.speaking_state_frames += 1
                if self.speaking_state_frames >= config.speaking_hold_frames:
                    self.speaking_state = SpeakingState.SILENT
                    self.speaking_state_frames = 0
    
    def update_position(self, x: float, y: float, w: float, h: float):
        """Update position and track history."""
        self.center_x = x
        self.center_y = y
        self.width = w
        self.height = h
        self.position_history.append((x, y))
        self.frames_since_seen = 0
        self.calculate_motion()  # Update motion score


@dataclass
class CropRegion:
    """Represents the output crop region."""
    x: int
    y: int
    width: int
    height: int
    
    def clamp(self, frame_w: int, frame_h: int):
        """Ensure crop stays within frame bounds."""
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.width > frame_w:
            self.x = frame_w - self.width
        if self.y + self.height > frame_h:
            self.y = frame_h - self.height


# =============================================================================
# SCENE DETECTOR
# =============================================================================

class SceneDetector:
    """
    Detects scene changes (cuts) in video.
    Uses PySceneDetect if available, falls back to simple pixel diff.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.prev_frame_small = None
        self.frame_count = 0
        self.last_cut_frame = -100
        
        # Check if PySceneDetect is available (for future use)
        self.use_pyscenedetect = False
        try:
            from scenedetect import SceneManager, ContentDetector
            self.use_pyscenedetect = True
        except ImportError:
            pass  # Will use pixel diff detection
    
    def detect_cut(self, frame: np.ndarray) -> bool:
        """
        Detect if current frame is a scene cut.
        Returns True if scene change detected.
        """
        self.frame_count += 1
        
        # Enforce minimum scene length
        if self.frame_count - self.last_cut_frame < self.config.scene_min_length:
            # Update prev frame but don't detect cut
            self._update_prev_frame(frame)
            return False
        
        # Simple pixel difference detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (80, 45))  # Downscale for speed
        
        is_cut = False
        
        if self.prev_frame_small is not None:
            diff = cv2.absdiff(small, self.prev_frame_small)
            mean_diff = np.mean(diff)
            
            if mean_diff > self.config.pixel_diff_threshold:
                is_cut = True
                self.last_cut_frame = self.frame_count
        
        self.prev_frame_small = small
        return is_cut
    
    def _update_prev_frame(self, frame: np.ndarray):
        """Update previous frame for comparison."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame_small = cv2.resize(gray, (80, 45))
    
    def reset(self):
        """Reset detector state."""
        self.prev_frame_small = None
        self.last_cut_frame = -100


# =============================================================================
# FACE DETECTOR (InsightFace - Better Side Profile Support)
# =============================================================================

class FaceDetector:
    """
    Face detector using InsightFace for robust detection including side profiles.
    InsightFace provides:
    1. Reliable face detection at various angles (frontal to 90° profile)
    2. 2D/3D facial landmarks for mouth tracking
    3. Face embeddings (useful for identity tracking)
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize InsightFace
        self.face_analyzer = None
        self._init_insightface()
        
        # Face tracking state
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.next_face_id = 0
        
        # Debug counters
        self._frame_count = 0
        self._face_detect_count = 0
    
    def _init_insightface(self):
        """Initialize InsightFace face analyzer."""
        try:
            from insightface.app import FaceAnalysis
            
            # Use buffalo_l for best accuracy, or buffalo_sc for speed
            # Models are downloaded automatically on first use
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',  # Best model for accuracy
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(
                ctx_id=0,  # GPU ID (0 for first GPU, -1 for CPU)
                det_size=self.config.insightface_det_size,
                det_thresh=self.config.insightface_det_thresh
            )
            print("[INFO] InsightFace initialized (buffalo_l model)")
            print("[INFO] InsightFace handles side profiles much better than MediaPipe!")
        except ImportError:
            print("[ERROR] InsightFace not installed!")
            print("[TIP] Install with: pip install insightface onnxruntime")
            print("[TIP] For GPU support: pip install onnxruntime-gpu")
        except Exception as e:
            print(f"[ERROR] InsightFace init failed: {e}")
            import traceback
            traceback.print_exc()
    
    def detect(self, frame: np.ndarray) -> Dict[int, TrackedFace]:
        """
        Detect and track faces in frame using InsightFace.
        
        InsightFace provides:
        - bbox: [x1, y1, x2, y2, score]
        - landmark_2d_106: 106 facial landmarks (if available)
        - landmark_3d_68: 68 3D landmarks
        - pose: [pitch, yaw, roll] head pose
        """
        self._frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        min_face_size = frame_h * self.config.min_face_size_ratio
        max_face_size = frame_h * self.config.max_face_size_ratio
        
        detected_faces = []
        
        if self.face_analyzer is not None:
            try:
                # InsightFace expects BGR (OpenCV default)
                faces = self.face_analyzer.get(frame, max_num=self.config.insightface_max_faces)
                
                for face in faces:
                    # Get bounding box
                    cx, cy, w, h = calculate_face_bbox_from_insightface(face.bbox)
                    
                    # Filter by size
                    if h < min_face_size or h > max_face_size:
                        continue
                    
                    # Calculate MAR from landmarks
                    mar = 0.0
                    has_landmarks = False
                    
                    # Try 106-point landmarks first (best for mouth)
                    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                        mar = calculate_mar_insightface(face.landmark_2d_106)
                        has_landmarks = True
                    # Fall back to 3D 68-point landmarks
                    elif hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                        # Use only x,y from 3D landmarks
                        landmarks_2d = face.landmark_3d_68[:, :2]
                        mar = calculate_mar_insightface(landmarks_2d)
                        has_landmarks = True
                    # Fall back to 2D 5-point landmarks (basic)
                    elif hasattr(face, 'landmark_2d') and face.landmark_2d is not None:
                        mar = calculate_mar_insightface(face.landmark_2d)
                        has_landmarks = len(face.landmark_2d) >= 5
                    elif hasattr(face, 'kps') and face.kps is not None:
                        # kps = keypoints (5 points: 2 eyes, nose, 2 mouth corners)
                        mar = calculate_mar_insightface(face.kps)
                        has_landmarks = True
                    
                    # Get confidence from bbox (5th element)
                    conf = float(face.bbox[4]) if len(face.bbox) > 4 else 0.9
                    
                    # Get head pose if available (useful for debugging)
                    pose = None
                    if hasattr(face, 'pose') and face.pose is not None:
                        pose = face.pose  # [pitch, yaw, roll]
                    
                    detected_faces.append({
                        'cx': cx,
                        'cy': cy,
                        'w': w,
                        'h': h,
                        'mar': mar,
                        'conf': conf,
                        'from_yolo': False,  # Not using YOLO anymore
                        'from_mesh': has_landmarks,  # True if we have landmarks
                        'pose': pose
                    })
                
                # Debug output every 100 frames
                if len(detected_faces) > 0:
                    self._face_detect_count += 1
                
                if self._frame_count % 100 == 0:
                    rate = (self._face_detect_count / self._frame_count * 100) if self._frame_count > 0 else 0
                    print(f"[DEBUG] Frame {self._frame_count}: InsightFace detected {len(detected_faces)} faces, "
                          f"Detection rate: {rate:.1f}%")
                    for i, f in enumerate(detected_faces):
                        pose_str = ""
                        if f.get('pose') is not None:
                            yaw = f['pose'][1]  # Yaw = left/right rotation
                            pose_str = f", Yaw: {yaw:.1f}°"
                        print(f"  Face {i+1}: MAR={f['mar']:.3f}, Conf={f['conf']:.2f}{pose_str}")
                        
            except Exception as e:
                print(f"[WARNING] InsightFace detection error: {e}")
                import traceback
                traceback.print_exc()
        
        # Track faces across frames (ID persistence)
        return self._track_faces(detected_faces)
    
    def _track_faces(self, detections: List[dict]) -> Dict[int, TrackedFace]:
        """Match detections to existing tracked faces and update IDs."""
        
        # Increment frames_since_seen for all
        for face in self.tracked_faces.values():
            face.frames_since_seen += 1
        
        matched_tracked = set()
        matched_detected = set()
        
        # Greedy matching by distance
        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_dist = float('inf')
            
            for track_id, tracked in self.tracked_faces.items():
                if track_id in matched_tracked:
                    continue
                
                dist = np.hypot(
                    det['cx'] - tracked.center_x,
                    det['cy'] - tracked.center_y
                )
                
                # Max distance based on face size
                max_dist = max(det['w'], det['h']) * 1.5
                
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing face
                tracked = self.tracked_faces[best_track_id]
                tracked.update_position(det['cx'], det['cy'], det['w'], det['h'])
                tracked.from_yolo = det.get('from_yolo', False)
                tracked.from_mesh = det.get('from_mesh', True)
                tracked.confidence = det['conf']
                
                if det['mar'] > 0:
                    tracked.update_mar(det['mar'])
                
                tracked.update_speaking_state(self.config)
                
                matched_tracked.add(best_track_id)
                matched_detected.add(det_idx)
        
        # Create new faces for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detected:
                continue
            
            new_face = TrackedFace(
                id=self.next_face_id,
                center_x=det['cx'],
                center_y=det['cy'],
                width=det['w'],
                height=det['h'],
                from_yolo=det.get('from_yolo', False),
                from_mesh=det.get('from_mesh', True),
                confidence=det['conf'],
                mar=det['mar']
            )
            new_face.position_history.append((det['cx'], det['cy']))
            if det['mar'] > 0:
                new_face.mar_history.append(det['mar'])
            
            self.tracked_faces[self.next_face_id] = new_face
            self.next_face_id += 1
        
        # Remove stale faces
        stale_ids = [
            fid for fid, face in self.tracked_faces.items()
            if face.frames_since_seen > 15  # ~0.5s at 30fps
        ]
        for fid in stale_ids:
            del self.tracked_faces[fid]
        
        return self.tracked_faces
    
    def reset(self):
        """Reset tracker (call on scene change)."""
        self.tracked_faces.clear()
    
    def close(self):
        """Release InsightFace resources."""
        pass  # InsightFace doesn't need explicit cleanup


# =============================================================================
# SPEAKER SELECTOR
# =============================================================================

class SpeakerSelector:
    """
    Selects which face is the active speaker using:
    1. Mouth movement (primary signal - highest priority)
    2. Face size (larger = more prominent)
    3. Position (centered faces preferred)
    4. Conditional hysteresis (only if current speaker is still speaking)
    """
    
    def __init__(self, config: Config, frame_w: int, frame_h: int):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h
        
        self.current_speaker_id: Optional[int] = None
        self.switch_candidate_id: Optional[int] = None
        self.switch_candidate_frames: int = 0
        self.frames_current_silent: int = 0  # Track how long current speaker has been silent
    
    def select(self, faces: Dict[int, TrackedFace]) -> Optional[int]:
        """
        Select the active speaker from tracked faces.
        Returns the speaker's face ID.
        """
        if not faces:
            return self.current_speaker_id  # Keep last speaker during brief gaps
        
        # Check if current speaker is still speaking
        current_is_speaking = False
        if self.current_speaker_id is not None and self.current_speaker_id in faces:
            current_is_speaking = faces[self.current_speaker_id].is_speaking
        
        # Track how long current speaker has been silent
        if current_is_speaking:
            self.frames_current_silent = 0
        else:
            self.frames_current_silent += 1
        
        # Check if anyone else is speaking
        others_speaking = []
        for fid, face in faces.items():
            if fid != self.current_speaker_id and face.is_speaking:
                others_speaking.append((fid, face))
        
        # Score each face
        scores = {}
        max_area = max(f.area for f in faces.values()) if faces else 1
        
        # Find who has been speaking the most recently
        most_recent_speaker_id = None
        min_frames_since_spoke = 999
        for fid, face in faces.items():
            if face.frames_since_last_spoke < min_frames_since_spoke:
                min_frames_since_spoke = face.frames_since_last_spoke
                most_recent_speaker_id = fid
        
        for fid, face in faces.items():
            score = 0.0
            has_mouth_data = face.from_mesh and len(face.mar_history) > 0
            
            # Speaking score (highest priority when we have mouth data)
            if face.is_speaking:
                # Base speaking score + bonus based on how strong the speaking signal is
                score += self.config.weight_speaking * (1 + face.speaking_score * 50)
            
            # MOTION DETECTION - fallback when no mouth tracking available
            # This helps detect who's active when MediaPipe fails to track mouth
            if not has_mouth_data:
                # Use motion as a proxy for speaking/activity
                if face.is_moving:
                    # Moving person is likely speaking or reacting
                    score += self.config.weight_motion * (1 + face.motion_score / 50.0)
                # Still apply penalty for no mouth data, but reduced if moving
                penalty = self.config.no_mesh_penalty if not face.is_moving else (self.config.no_mesh_penalty * 0.3)
                score -= penalty
            elif len(face.mar_history) == 0:
                # Has mesh but no MAR yet - small penalty, will resolve soon
                score -= 2.0
            
            # RECENT SPEAKER BONUS - key for handling pauses
            # Give significant bonus to whoever was speaking recently (within ~3 seconds)
            if face.was_recently_speaking and face.total_speaking_frames > 10:
                # Decaying bonus based on how recently they spoke
                recency = 1.0 - (face.frames_since_last_spoke / self.config.recent_speaker_hold_frames)
                recency = max(0, recency)
                score += self.config.recent_speaker_bonus * recency
            
            # Face size score (larger = more important, but reduced weight)
            size_ratio = face.area / max_area
            score += self.config.weight_face_size * size_ratio
            
            # Center position score
            center_dist = abs(face.center_x - self.frame_w / 2) / (self.frame_w / 2)
            center_score = 1.0 - center_dist  # Higher score if more centered
            score += self.config.weight_center * center_score
            
            # CONDITIONAL Hysteresis: bonus for current speaker depends on their state
            if fid == self.current_speaker_id:
                if current_is_speaking:
                    # Full hysteresis - they're still speaking
                    score += self.config.weight_current_speaking
                elif self.frames_current_silent < 45:  # ~1.5 second grace period
                    # Reduced hysteresis - they recently stopped
                    decay = 1.0 - (self.frames_current_silent / 45.0)
                    score += self.config.weight_current_silent * decay
                # If silent for too long, no hysteresis bonus
            
            # Confidence bonus
            score *= face.confidence
            
            scores[fid] = score
        
        # Find best candidate
        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]
        
        # Determine switch delay based on context
        # Determine if we should use fast switching
        # Fast switch ONLY if: 
        # 1. Someone else has been speaking for a while (not just a brief moment)
        # 2. Current speaker has been silent for a significant time
        # 3. The other person has accumulated real speaking time
        others_speaking_sustained = [
            (fid, face) for fid, face in others_speaking 
            if face.total_speaking_frames > 30  # Must have spoken for ~1 second total
        ]
        
        use_fast_switch = (len(others_speaking_sustained) > 0 and 
                          not current_is_speaking and 
                          self.frames_current_silent > 30)  # Current silent for ~1 second
        
        switch_delay = (self.config.speaker_switch_delay_fast if use_fast_switch 
                       else self.config.speaker_switch_delay)
        
        # Apply switch delay
        if best_id != self.current_speaker_id:
            # Additional check: don't switch if current was recently speaking
            # unless the other person has been speaking significantly longer
            current_face = faces.get(self.current_speaker_id)
            best_face = faces.get(best_id)
            
            should_consider_switch = True
            if current_face and best_face:
                # If current speaker was speaking recently, require strong evidence
                if current_face.was_recently_speaking and current_face.frames_since_last_spoke < 60:
                    # Only switch if best candidate is actively speaking now
                    if not best_face.is_speaking:
                        should_consider_switch = False
            
            if should_consider_switch:
                if self.switch_candidate_id == best_id:
                    self.switch_candidate_frames += 1
                else:
                    self.switch_candidate_id = best_id
                    self.switch_candidate_frames = 1
                
                # Only switch after delay
                if self.switch_candidate_frames >= switch_delay:
                    self.current_speaker_id = best_id
                    self.switch_candidate_id = None
                    self.switch_candidate_frames = 0
                    self.frames_current_silent = 0  # Reset for new speaker
            else:
                # Reset switch candidate if we're not considering a switch
                self.switch_candidate_frames = max(0, self.switch_candidate_frames - 1)
        else:
            self.switch_candidate_id = None
            self.switch_candidate_frames = 0
        
        # If no current speaker yet, just pick best
        if self.current_speaker_id is None:
            self.current_speaker_id = best_id
        
        return self.current_speaker_id
    
    def force_speaker(self, speaker_id: Optional[int]):
        """Force switch to specific speaker (used on scene change)."""
        self.current_speaker_id = speaker_id
        self.switch_candidate_id = None
        self.switch_candidate_frames = 0
    
    def reset(self):
        """Reset selector state."""
        self.current_speaker_id = None
        self.switch_candidate_id = None
        self.switch_candidate_frames = 0
        self.frames_current_silent = 0
    
    def on_camera_angle_change(self):
        """Called when camera angle changes significantly (not a full scene cut)."""
        # Reduce hysteresis but don't fully reset - let speaking detection take over
        self.frames_current_silent = 20  # Pretend current speaker has been silent
        self.switch_candidate_frames = 0


# =============================================================================
# CAMERA CONTROLLER (Kalman Filter)
# =============================================================================

class CameraController:
    """
    Controls the virtual camera position using Kalman filtering
    for smooth, natural movement.
    """
    
    def __init__(self, config: Config, frame_w: int, frame_h: int):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h
        
        # Calculate crop dimensions
        self.crop_h = frame_h
        self.crop_w = int(frame_h * config.output_aspect_ratio)
        
        if self.crop_w > frame_w:
            self.crop_w = frame_w
            self.crop_h = int(frame_w / config.output_aspect_ratio)
        
        # Camera position (smoothed)
        self.cam_x = frame_w / 2
        self.cam_y = frame_h / 2
        
        # Target position
        self.target_x = frame_w / 2
        self.target_y = frame_h / 2
        
        # Previous target for jump detection
        self.prev_target_x = frame_w / 2
        self.prev_target_y = frame_h / 2
        
        # Kalman filter for X position
        self.kalman_x = self._create_kalman()
        self.kalman_y = self._create_kalman()
        
        # Initialize Kalman with center position
        if self.kalman_x:
            self.kalman_x.x = np.array([[frame_w / 2], [0]])
        if self.kalman_y:
            self.kalman_y.x = np.array([[frame_h / 2], [0]])
        
        # Instant snap flag
        self.snap_next = False
    
    def _create_kalman(self):
        """Create a 1D Kalman filter for position tracking."""
        try:
            from filterpy.kalman import KalmanFilter
            
            kf = KalmanFilter(dim_x=2, dim_z=1)
            
            # State transition matrix (position + velocity)
            kf.F = np.array([[1, 1],
                            [0, 1]])
            
            # Measurement matrix (we only measure position)
            kf.H = np.array([[1, 0]])
            
            # Measurement noise
            kf.R = np.array([[self.config.kalman_measurement_noise]])
            
            # Process noise
            kf.Q = np.array([[self.config.kalman_process_noise, 0],
                            [0, self.config.kalman_process_noise]])
            
            # Initial covariance
            kf.P = np.array([[1, 0],
                            [0, 1]])
            
            return kf
        except ImportError:
            print("[WARNING] filterpy not installed, using EMA fallback")
            return None
    
    def update(self, target_x: float, target_y: float, 
               is_scene_cut: bool = False) -> Tuple[CropRegion, bool]:
        """
        Update camera position and return crop region.
        Returns: (CropRegion, is_angle_change)
        """
        # Check for position jump (camera angle change)
        jump_dist = np.hypot(
            target_x - self.prev_target_x,
            target_y - self.prev_target_y
        )
        is_jump = jump_dist > self.frame_w * self.config.position_jump_threshold
        
        # Scene cut or jump = instant snap
        if is_scene_cut or is_jump or self.snap_next:
            self.cam_x = target_x
            self.cam_y = target_y
            
            # Reset Kalman filters
            if self.kalman_x:
                self.kalman_x.x = np.array([[target_x], [0]])
            if self.kalman_y:
                self.kalman_y.x = np.array([[target_y], [0]])
            
            self.snap_next = False
        else:
            # Smooth tracking
            if self.kalman_x and self.kalman_y:
                # Kalman filter update
                self.kalman_x.predict()
                self.kalman_x.update(np.array([[target_x]]))
                self.cam_x = self.kalman_x.x[0, 0]
                
                self.kalman_y.predict()
                self.kalman_y.update(np.array([[target_y]]))
                self.cam_y = self.kalman_y.x[0, 0]
            else:
                # EMA fallback
                alpha = self.config.ema_alpha_smooth
                self.cam_x = alpha * target_x + (1 - alpha) * self.cam_x
                self.cam_y = alpha * target_y + (1 - alpha) * self.cam_y
        
        # Update previous target
        self.prev_target_x = target_x
        self.prev_target_y = target_y
        self.target_x = target_x
        self.target_y = target_y
        
        return self._calculate_crop(), is_jump
    
    def _calculate_crop(self) -> CropRegion:
        """Calculate crop region based on current camera position."""
        # Position face at specified vertical position
        crop_x = int(self.cam_x - self.crop_w * self.config.face_horizontal_position)
        crop_y = int(self.cam_y - self.crop_h * self.config.face_vertical_position)
        
        crop = CropRegion(
            x=crop_x,
            y=crop_y,
            width=self.crop_w,
            height=self.crop_h
        )
        
        crop.clamp(self.frame_w, self.frame_h)
        return crop
    
    def trigger_snap(self):
        """Trigger instant snap on next update."""
        self.snap_next = True
    
    def get_center_crop(self) -> CropRegion:
        """Get a centered crop (fallback when no face detected)."""
        return CropRegion(
            x=(self.frame_w - self.crop_w) // 2,
            y=(self.frame_h - self.crop_h) // 2,
            width=self.crop_w,
            height=self.crop_h
        )


# =============================================================================
# VIDEO PROCESSOR
# =============================================================================

class VideoProcessor:
    """Main video processing pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process(self, input_path: str, output_path: str,
                preview: bool = False, debug: bool = False):
        """Process video and generate vertical output."""
        
        print(f"[INFO] Opening: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Input: {frame_w}x{frame_h} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Initialize components
        scene_detector = SceneDetector(self.config)
        face_detector = FaceDetector(self.config)
        speaker_selector = SpeakerSelector(self.config, frame_w, frame_h)
        camera = CameraController(self.config, frame_w, frame_h)
        
        out_w, out_h = camera.crop_w, camera.crop_h
        print(f"[INFO] Output: {out_w}x{out_h}")
        
        # Setup output
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        temp_video = os.path.join(
            output_dir if output_dir else ".",
            f"_temp_{os.getpid()}.mp4"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
        
        if not writer.isOpened():
            raise ValueError("Cannot create video writer")
        
        # Processing loop
        frame_count = 0
        start_time = time.time()
        
        print("[INFO] Processing...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 1. Scene detection
                is_scene_cut = scene_detector.detect_cut(frame)
                
                if is_scene_cut:
                    # Reset on scene change
                    face_detector.reset()
                    speaker_selector.reset()
                    camera.trigger_snap()
                
                # 2. Face detection
                if frame_count % self.config.detection_interval == 0 or is_scene_cut:
                    faces = face_detector.detect(frame)
                else:
                    faces = face_detector.tracked_faces
                
                # 3. Speaker selection
                speaker_id = speaker_selector.select(faces)
                
                # 4. Calculate crop
                is_angle_change = False
                if speaker_id is not None and speaker_id in faces:
                    face = faces[speaker_id]
                    crop, is_angle_change = camera.update(face.center_x, face.center_y, is_scene_cut)
                    
                    # If camera detected significant position jump, inform speaker selector
                    if is_angle_change and not is_scene_cut:
                        speaker_selector.on_camera_angle_change()
                else:
                    crop = camera.get_center_crop()
                
                # 5. Extract crop
                cropped = frame[crop.y:crop.y+crop.height, 
                               crop.x:crop.x+crop.width]
                
                # Ensure correct size
                if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                    cropped = cv2.resize(cropped, (out_w, out_h))
                
                # 6. Debug overlay
                if debug:
                    cropped = self._draw_debug(
                        cropped, faces, speaker_id, crop, 
                        is_scene_cut, frame
                    )
                
                # 7. Write frame
                writer.write(cropped)
                
                # 8. Preview (with fallback if GUI not available)
                if preview or debug:
                    try:
                        # Show cropped output
                        preview_h = 720
                        preview_w = int(preview_h * out_w / out_h)
                        preview_frame = cv2.resize(cropped, (preview_w, preview_h))
                        cv2.imshow("Output Preview", preview_frame)
                        
                        if debug:
                            # Also show full frame with overlays
                            debug_full = self._draw_debug_fullframe(
                                frame.copy(), faces, speaker_id, crop
                            )
                            debug_full = cv2.resize(debug_full, (960, 540))
                            cv2.imshow("Full Frame Debug", debug_full)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("[INFO] Stopped by user")
                            break
                        elif key == ord(' '):
                            cv2.waitKey(0)
                    except cv2.error:
                        # GUI not available (headless OpenCV)
                        if frame_count == 1:
                            print("[WARNING] Preview disabled - OpenCV has no GUI support")
                            print("[TIP] Install opencv-python instead of opencv-python-headless:")
                            print("      pip uninstall opencv-python-headless && pip install opencv-python")
                        preview = False  # Disable preview for remaining frames
                
                # Progress
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    progress = frame_count / total_frames * 100
                    eta = (total_frames - frame_count) / fps_actual
                    print(f"[INFO] {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                          f"{fps_actual:.1f} FPS | ETA: {eta:.0f}s")
        
        finally:
            cap.release()
            writer.release()
            face_detector.close()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # GUI not available
        
        elapsed = time.time() - start_time
        print(f"[INFO] Processed {frame_count} frames in {elapsed:.1f}s "
              f"({frame_count/elapsed:.1f} FPS)")
        
        # Merge audio with FFmpeg
        self._encode_with_audio(input_path, temp_video, output_path)
        
        # Cleanup
        try:
            os.remove(temp_video)
        except:
            pass
        
        print(f"[SUCCESS] Output saved to: {output_path}")
    
    def _draw_debug(self, cropped: np.ndarray, faces: Dict[int, TrackedFace],
                    speaker_id: Optional[int], crop: CropRegion,
                    is_scene_cut: bool, full_frame: np.ndarray) -> np.ndarray:
        """Draw debug info on cropped frame."""
        debug = cropped.copy()
        
        # Scene cut indicator
        if is_scene_cut:
            cv2.putText(debug, "SCENE CUT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Speaker info
        if speaker_id is not None and speaker_id in faces:
            face = faces[speaker_id]
            cv2.putText(debug, f"Speaker: {speaker_id} ({'SPEAKING' if face.is_speaking else 'silent'})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug, f"MAR: {face.mar:.3f} Score: {face.speaking_score:.4f}",
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Face count
        cv2.putText(debug, f"Faces: {len(faces)}", (10, cropped.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug
    
    def _draw_debug_fullframe(self, frame: np.ndarray, 
                               faces: Dict[int, TrackedFace],
                               speaker_id: Optional[int],
                               crop: CropRegion) -> np.ndarray:
        """Draw debug overlays on full frame."""
        debug = frame.copy()
        
        # Draw all faces
        for fid, face in faces.items():
            x1 = int(face.center_x - face.width / 2)
            y1 = int(face.center_y - face.height / 2)
            x2 = int(face.center_x + face.width / 2)
            y2 = int(face.center_y + face.height / 2)
            
            # Color based on status
            if fid == speaker_id:
                color = (0, 255, 0)  # Green for active speaker
                thickness = 3
            elif face.is_speaking:
                color = (0, 255, 255)  # Yellow for speaking but not selected
                thickness = 2
            elif face.is_moving:
                color = (0, 165, 255)  # Orange for moving but not speaking
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for silent/still
                thickness = 1
            
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)
            
            # Label - show detection quality
            label = f"ID:{fid}"
            if face.is_speaking:
                label += " SPEAKING"
            elif face.is_moving:
                label += " MOVING"
            elif face.was_recently_speaking:
                label += " (recent)"
            label += f" MAR:{face.mar:.2f}"
            if face.motion_score > 0:
                label += f" M:{face.motion_score:.1f}"
            if not face.from_mesh:
                label += " [5pt]"  # Only has 5-point landmarks (basic)
            
            cv2.putText(debug, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw crop region
        cv2.rectangle(debug, (crop.x, crop.y),
                     (crop.x + crop.width, crop.y + crop.height),
                     (0, 255, 255), 2)
        cv2.putText(debug, "CROP", (crop.x + 5, crop.y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return debug
    
    def _encode_with_audio(self, original: str, processed: str, output: str):
        """Combine processed video with original audio using FFmpeg."""
        ffmpeg = self._find_ffmpeg()
        
        if not ffmpeg:
            print("[WARNING] FFmpeg not found, copying without re-encoding")
            import shutil
            shutil.copy(processed, output)
            return
        
        print("[INFO] Encoding with audio...")
        
        cmd = [
            ffmpeg, '-y',
            '-i', processed,
            '-i', original,
            '-map', '0:v',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-movflags', '+faststart',
            output
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[WARNING] FFmpeg error: {result.stderr[:500]}")
                # Fallback without audio
                cmd_no_audio = [
                    ffmpeg, '-y',
                    '-i', processed,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    output
                ]
                subprocess.run(cmd_no_audio, check=True)
        except Exception as e:
            print(f"[ERROR] Encoding failed: {e}")
            import shutil
            shutil.copy(processed, output)
    
    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable."""
        import shutil as sh
        
        # Check PATH
        ffmpeg = sh.which('ffmpeg')
        if ffmpeg:
            return ffmpeg
        
        # Common Windows paths
        paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\ffmpeg\bin\ffmpeg.exe'),
        ]
        
        for p in paths:
            if os.path.exists(p):
                return p
        
        # Try imageio-ffmpeg
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python face_tracker.py <input.mp4> <output.mp4> [options]")
        print()
        print("Options:")
        print("  --preview    Show preview during processing")
        print("  --debug      Show debug overlays")
        print()
        print("Example:")
        print("  python face_tracker.py podcast.mp4 short.mp4 --preview")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    preview = '--preview' in sys.argv
    debug = '--debug' in sys.argv
    
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)
    
    config = Config()
    processor = VideoProcessor(config)
    
    try:
        processor.process(input_path, output_path, preview=preview, debug=debug)
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
