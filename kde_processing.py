import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import time

# mic_positions: dict mapping mic_index -> (x, y) physical coordinates
#  A(0,  0)
#  B(d,  0)
#  C(0, -d)
#                      
#   A-----B     
#   |    /       
#   |   /         
#   |  /         
#   | /           
#   |/                       
#   C 
#                         

mic_positions = {
    0: (0.0,  0.0), # Mic A
    1: (0.1,  0.0), # Mic B
    2: (0.0, -0.1), # Mic C
}

MIC_A = 0
MIC_B = 1
MIC_C = 2

def pair_baseline_angle_rad(pos_a, pos_b):
    """Angle of the line connecting mic A and mic B, in global frame."""
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    return np.arctan2(dy, dx)

# $LEFT_MIC_$RIGHT_MIC_offset_angle = pair_baseline_angle_rad(...)
a_b_offset_angle_rad = pair_baseline_angle_rad(mic_positions[MIC_A], mic_positions[MIC_B])
b_c_offset_angle_rad = pair_baseline_angle_rad(mic_positions[MIC_B], mic_positions[MIC_C])
a_c_offset_angle_rad = pair_baseline_angle_rad(mic_positions[MIC_A], mic_positions[MIC_C])


def get_candidate_angles_rad(local_angle_rad, offset_angle_rad):
    
    return (offset_angle_rad - local_angle_rad,
            offset_angle_rad + local_angle_rad )


def estimate_azimuth_360_fast_deg(all_candidates_rad, n_bins=360):
    # all_candidates_rad: raw angles in [0, 2*pi), NOT tripled
    angles_deg = np.degrees(all_candidates_rad) % 360

    # bin into a circular histogram (1 bin per degree, adjust n_bins for resolution)
    hist, bin_edges = np.histogram(angles_deg, bins=n_bins, range=(0, 360))

    # sigma in bin units — convert your target "bandwidth" (e.g. ~25 deg) to bins
    bandwidth_deg = 25.0
    sigma_bins = bandwidth_deg / (360 / n_bins)

    smoothed = gaussian_filter1d(hist.astype(float), sigma=sigma_bins, mode='wrap')

    best_bin = np.argmax(smoothed)
    best_angle_deg = bin_edges[best_bin] + (360 / n_bins) / 2  # bin center

    return best_angle_deg

def estimate_azimuth_360_normal_deg(all_candidates_rad, resolution=360):
    # circular KDE: duplicate data shifted by ±2pi to handle wraparound
    extended = np.concatenate([all_candidates_rad, all_candidates_rad+2*np.pi, all_candidates_rad-2*np.pi])
    kde = gaussian_kde(extended, bw_method=0.1)  # tune bandwidth ~ paper's 25 deg equivalent

    theta_grid = np.linspace(0, 2*np.pi, resolution)
    density = kde(theta_grid)

    best_angle_rad = theta_grid[np.argmax(density)]
    
    return np.degrees(best_angle_rad)
