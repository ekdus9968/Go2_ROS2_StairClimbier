"""
Stair Detection Algorithm using LiDAR point cloud.

Approach:
1. Preprocess: Filter to region of interest
2. (Optional) Remove points near person
3. KDE + peak detection to find horizontal Z surfaces
4. Refine step positions
5. Find longest valid stair sequence (both Z and XY match)
6. Confirm regularity

Key insight for step 5:
  Peaks come from many surfaces (stairs, floor, walls, etc).
  Sort by Z, then find longest consecutive subsequence where
  BOTH Z-gap and XY-progression match stair pattern.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


class StairDetector:
    def __init__(
        self,
        forward_min: float = 0.3,
        forward_max: float = 5.0,
        side_range: float = 2.5,
        z_min: float = -0.5,
        z_max: float = 2.0,
        person_exclusion_radius: float = 0.5,
        kde_bandwidth: float = 0.02,
        peak_height_ratio: float = 0.1,
        peak_min_distance: float = 0.05,
        z_slice_thickness: float = 0.03,
        min_points_per_step: int = 20,
        min_steps: int = 3,
        step_height_min: float = 0.05,
        step_height_max: float = 0.30,
        height_std_ratio: float = 0.5,
        xy_min_distance: float = 0.05,
        xy_max_distance: float = 0.60,
    ):
        self.forward_min = forward_min
        self.forward_max = forward_max
        self.side_range = side_range
        self.z_min = z_min
        self.z_max = z_max
        self.person_exclusion_radius = person_exclusion_radius
        self.kde_bandwidth = kde_bandwidth
        self.peak_height_ratio = peak_height_ratio
        self.peak_min_distance = peak_min_distance
        self.z_slice = z_slice_thickness
        self.min_pts_per_step = min_points_per_step
        self.min_steps = min_steps
        self.step_h_min = step_height_min
        self.step_h_max = step_height_max
        self.height_std_ratio = height_std_ratio
        self.xy_min = xy_min_distance
        self.xy_max = xy_max_distance

    def detect(self, points: np.ndarray, person_position=None) -> tuple:
        info = {'reason': ''}

        # 1. Preprocess
        pts = self.preprocess(points)
        info['num_points_roi'] = len(pts)
        if len(pts) < 100:
            info['reason'] = 'not enough points in ROI'
            return False, info

        # 1.5. Remove person area
        if person_position is not None:
            pts = self.remove_person_area(pts, person_position)
            info['num_points_after_person'] = len(pts)
            if len(pts) < 100:
                info['reason'] = 'not enough points after person removal'
                return False, info

        # 2. KDE + peak detection
        peaks = self.find_z_peaks(pts)
        info['num_peaks'] = len(peaks)
        if len(peaks) < self.min_steps:
            info['reason'] = (
                f'not enough peaks ({len(peaks)} < {self.min_steps})')
            return False, info

        # 3. Refine
        all_steps = self.refine_steps(pts, peaks)
        info['num_refined_steps'] = len(all_steps)
        if len(all_steps) < self.min_steps:
            info['reason'] = (
                f'not enough refined steps ({len(all_steps)})')
            return False, info

        # 4. Find longest valid stair sequence
        stair_seq = self.find_longest_stair_sequence(all_steps)
        info['stair_sequence_length'] = len(stair_seq)
        if len(stair_seq) < self.min_steps:
            info['reason'] = (
                f'no valid stair sequence '
                f'(longest={len(stair_seq)} < {self.min_steps})')
            return False, info

        # 5. Regularity check on sequence
        valid, reason = self.check_regularity(stair_seq)
        if not valid:
            info['reason'] = reason
            info['stair_sequence'] = stair_seq
            return False, info

        # 6. Direction
        direction = self.compute_direction(stair_seq)
        heights = np.diff([s['z'] for s in stair_seq])
        info['steps'] = stair_seq
        info['num_steps'] = len(stair_seq)
        info['avg_step_height'] = float(np.mean(heights))
        info['direction_yaw'] = direction
        return True, info

    def preprocess(self, points: np.ndarray) -> np.ndarray:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x >= self.forward_min) & (x <= self.forward_max) &
            (np.abs(y) <= self.side_range) &
            (z >= self.z_min) & (z <= self.z_max)
        )
        return points[mask]

    def remove_person_area(self, points, person_pos):
        dx = points[:, 0] - person_pos[0]
        dy = points[:, 1] - person_pos[1]
        dist_xy = np.sqrt(dx ** 2 + dy ** 2)
        return points[dist_xy > self.person_exclusion_radius]

    def find_z_peaks(self, points):
        z_values = points[:, 2]
        if len(z_values) < 30:
            return np.array([])

        try:
            kde = gaussian_kde(z_values, bw_method=self.kde_bandwidth)
        except Exception:
            return np.array([])

        z_grid = np.linspace(z_values.min(), z_values.max(), 1000)
        density = kde(z_grid)
        if density.max() < 1e-6:
            return np.array([])

        dz = z_grid[1] - z_grid[0]
        distance_pts = max(1, int(self.peak_min_distance / dz))
        height = density.max() * self.peak_height_ratio
        peak_idx, _ = find_peaks(
            density, height=height, distance=distance_pts)
        return z_grid[peak_idx]

    def refine_steps(self, points, peaks):
        steps = []
        for pz in peaks:
            mask = np.abs(points[:, 2] - pz) < self.z_slice
            step_points = points[mask]
            if len(step_points) < self.min_pts_per_step:
                continue
            actual_z = float(step_points[:, 2].mean())
            center_xy = step_points[:, :2].mean(axis=0)
            size = step_points.max(axis=0) - step_points.min(axis=0)
            steps.append({
                'z': actual_z,
                'center': (float(center_xy[0]),
                            float(center_xy[1]), actual_z),
                'size': (float(size[0]), float(size[1]), float(size[2])),
                'num_points': int(len(step_points)),
            })
        return steps

    def find_longest_stair_sequence(self, all_steps: list) -> list:
        """Find longest consecutive Z-sorted sequence where each
        step-to-next transition matches stair Z gap AND XY progression.
        """
        if len(all_steps) < self.min_steps:
            return []

        # Sort by z
        sorted_steps = sorted(all_steps, key=lambda s: s['z'])
        n = len(sorted_steps)

        best_sequence = []

        for start in range(n):
            current = [sorted_steps[start]]
            for i in range(start + 1, n):
                s = sorted_steps[i]
                last = current[-1]

                # Z gap
                dz = s['z'] - last['z']
                if not (self.step_h_min <= dz <= self.step_h_max):
                    break

                # XY progression
                dxy = np.sqrt(
                    (s['center'][0] - last['center'][0]) ** 2 +
                    (s['center'][1] - last['center'][1]) ** 2
                )
                if not (self.xy_min <= dxy <= self.xy_max):
                    break

                current.append(s)

            if len(current) > len(best_sequence):
                best_sequence = current

        return best_sequence

    def check_regularity(self, sequence: list) -> tuple:
        """Check step height variance."""
        if len(sequence) < self.min_steps:
            return False, f'sequence too short ({len(sequence)})'

        z_values = np.array([s['z'] for s in sequence])
        diffs = np.diff(z_values)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        if mean_diff > 0 and (std_diff / mean_diff) > self.height_std_ratio:
            return False, (
                f'height variance too large: std={std_diff:.3f}, '
                f'mean={mean_diff:.3f}')

        return True, 'valid'

    def compute_direction(self, sequence: list) -> float:
        first = sequence[0]['center']
        last = sequence[-1]['center']
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        return float(np.arctan2(dy, dx))