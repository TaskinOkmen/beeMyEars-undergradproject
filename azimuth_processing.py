import numpy as np

SPEED_OF_SOUND_METERS_PER_SECOND = 343.0 # 343.0 m/s

DEGREES_PER_RADIAN = 57.2957795 # degrees

BLOCK_SIZE = 2048 # 8192 2048 4096
HOP_SIZE   = 480 # 2040 480  1080



def calculate_tdoa_rad(mic_left_signal, mic_right_signal, fs_hz, mic_distance_m):

    time_difference_s = calculate_time_difference_s(mic_left_signal, mic_right_signal, fs_hz)

    angle_of_arrival_rad = calculate_angle_of_arrival_rad(time_difference_s, mic_distance_m)

    return angle_of_arrival_rad



def calculate_time_difference_s(mic_left_signal, mic_right_signal, fs_hz):

    # --- FFT ---
    X1 = np.fft.fft(mic_left_signal)
    X2 = np.fft.fft(mic_right_signal)

    # Cross power spectrum
    R = X1 * np.conj(X2)

    # Partial PHAT normalization (power -0.3)
    R_phat = R / ((np.abs(R) + 1e-10) ** (-0.3) + 1e-10)

    # --- IFFT -> correlation ---
    gcc = np.fft.ifft(R_phat)
    gcc = np.real(gcc)

    # --- Shift to center ---
    gcc = np.fft.fftshift(gcc)

    # --- Lag axis ---
    lags = np.arange(-len(gcc)//2, len(gcc)//2)

    # --- Peak detection ---
    peak_index = np.argmax(gcc)
    estimated_lag = lags[peak_index]

    # --- Convert to time ---
    time_difference_s = estimated_lag / fs_hz

    return time_difference_s



def calculate_angle_of_arrival_rad(time_difference_s, mic_distance_m):

    val = np.clip((time_difference_s * SPEED_OF_SOUND_METERS_PER_SECOND) / mic_distance_m, -1.0, 1.0)

    return np.arcsin(val)



def estimate_aoa_over_time_overlap_to_array(mic_left_signal, mic_right_signal, fs_hz, mic_distance_m, block_size=BLOCK_SIZE, hop_size=HOP_SIZE):

    num_samples = len(mic_left_signal)
    angles = []

    for start in range(0, num_samples - block_size, hop_size):

        end = start + block_size

        block_x1 = mic_left_signal[start:end]
        block_x2 = mic_right_signal[start:end]

        angle = calculate_tdoa_rad(
            block_x1,
            block_x2,
            fs_hz,
            mic_distance_m
        )

        angles.append(angle * DEGREES_PER_RADIAN)

    return np.array(angles)