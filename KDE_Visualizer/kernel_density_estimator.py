import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import time

lst = []

def add_item_to_list(lst, item):
    """Add an item to a list, maintaining a maximum length of 100."""
    lst.append(item)
    if len(lst) > 100:
        lst.pop(0)  # Remove the oldest item if the list exceeds 100 items

add_item_to_list(lst, 1)  
add_item_to_list(lst, 2)
add_item_to_list(lst, 3)

print(lst)  # Output: [1, 2, 3]

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

# $LEFT_MIC_$RIGHT_MIC_offse_angle = pair_baseline_angle_rad(...)
a_b_offset_angle_rad = pair_baseline_angle_rad(mic_positions[MIC_A], mic_positions[MIC_B])
b_c_offset_angle_rad = pair_baseline_angle_rad(mic_positions[MIC_B], mic_positions[MIC_C])
a_c_offset_angle_rad = pair_baseline_angle_rad(mic_positions[MIC_A], mic_positions[MIC_C])



print(f"Angle of baseline AB: {np.degrees(a_b_offset_angle_rad):.2f} degrees")
print(f"Angle of baseline BC: {np.degrees(b_c_offset_angle_rad):.2f} degrees")
print(f"Angle of baseline AC: {np.degrees(a_c_offset_angle_rad):.2f} degrees")



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

    # angle_centers_deg = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.figure(figsize=(10, 5))
    # plt.plot(angle_centers_deg, smoothed, label='smoothed', linewidth=2)
    # plt.axvline(best_angle_deg, color='red', linestyle=':', label=f'Best angle: {best_angle_deg:.2f}°')
    # plt.title('Smoothed azimuth estimate')
    # plt.xlabel('Angle (degrees)')
    # plt.ylabel('Smoothed density')
    # plt.xlim(0, 360)
    # plt.xticks(np.arange(0, 361, 45))
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return best_angle_deg

def estimate_azimuth_360_normal_deg(all_candidates_rad, resolution=360):
    # circular KDE: duplicate data shifted by ±2pi to handle wraparound
    extended = np.concatenate([all_candidates_rad, all_candidates_rad+2*np.pi, all_candidates_rad-2*np.pi])
    kde = gaussian_kde(extended, bw_method=0.1)  # tune bandwidth ~ paper's 25 deg equivalent

    theta_grid = np.linspace(0, 2*np.pi, resolution)
    density = kde(theta_grid)

    best_angle_rad = theta_grid[np.argmax(density)]

    best_angle_deg = np.degrees(best_angle_rad)

    # plt.figure(figsize=(10, 5))
    # plt.plot(np.degrees(theta_grid), density, label='kde(theta_grid)', linewidth=2)
    # plt.plot(np.degrees(theta_grid), density, label='density', linestyle='--', alpha=0.8)
    # plt.axvline(best_angle_deg, color='red', linestyle=':', label=f'Best angle: {best_angle_deg:.2f}°')
    # plt.title('Circular KDE density estimate')
    # plt.xlabel('Angle (degrees)')
    # plt.ylabel('Density')
    # plt.xlim(0, 360)
    # plt.xticks(np.arange(0, 361, 45))
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    return np.degrees(best_angle_rad)

# 315 degrees
# a_b_local_angle_deg = np.radians(45)
# b_c_local_angle_deg = np.radians(90)
# a_c_local_angle_deg = np.radians(45)

# 270 degrees
# a_b_local_angle_deg = np.radians(90)
# b_c_local_angle_deg = np.radians(45)
# a_c_local_angle_deg = np.radians(0)

# 225 degrees
# a_b_local_angle_deg = np.radians(135)
# b_c_local_angle_deg = np.radians(0)
# a_c_local_angle_deg = np.radians(45)

# 180 degrees
# a_b_local_angle_deg = np.radians(180)
# b_c_local_angle_deg = np.radians(45)
# a_c_local_angle_deg = np.radians(90)

# 135 degrees
a_b_local_angle_deg = np.radians(135)
b_c_local_angle_deg = np.radians(90)
a_c_local_angle_deg = np.radians(135)

# 90 degrees
# a_b_local_angle_deg = np.radians(90)
# b_c_local_angle_deg = np.radians(135)
# a_c_local_angle_deg = np.radians(180)

# 45 degrees
# a_b_local_angle_deg = np.radians(45)
# b_c_local_angle_deg = np.radians(180)
# a_c_local_angle_deg = np.radians(135)

# 0 degrees
# a_b_local_angle_deg = np.radians(0)
# b_c_local_angle_deg = np.radians(135)
# a_c_local_angle_deg = np.radians(90)

all_candidates = []

c1, c2 = get_candidate_angles_rad(a_b_local_angle_deg, a_b_offset_angle_rad)
print(f"Candidate angles for AB: {np.degrees(c1):.2f}, {np.degrees(c2):.2f} degrees")
all_candidates.extend([c1, c2])

c1, c2 = get_candidate_angles_rad(b_c_local_angle_deg, b_c_offset_angle_rad)
print(f"Candidate angles for BC: {np.degrees(c1):.2f}, {np.degrees(c2):.2f} degrees")
all_candidates.extend([c1, c2])

c1, c2 = get_candidate_angles_rad(a_c_local_angle_deg, a_c_offset_angle_rad)
print(f"Candidate angles for AC: {np.degrees(c1):.2f}, {np.degrees(c2):.2f} degrees")
all_candidates.extend([c1, c2])

all_candidates = np.mod(all_candidates, 2*np.pi)

print(f"All candidate angles (rad): {np.degrees(all_candidates)}")


# Start the timer
start_time = time.perf_counter_ns()

# best_angle_deg = estimate_azimuth_360_normal_deg(all_candidates, resolution=360)

best_angle_deg = estimate_azimuth_360_fast_deg(all_candidates, n_bins=180)

# End the timer
end_time = time.perf_counter_ns()

# Calculate elapsed time
execution_time_us = (end_time - start_time) / 1000
print(f"Execution time: {execution_time_us:.6f} microseconds")

print(f"Best angle (deg): {best_angle_deg:.2f}")
