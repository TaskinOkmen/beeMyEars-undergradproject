import numpy as np
import matplotlib.pyplot as plt

# Constant Parameters
fs_hz = 44100                      # Sampling frequency
f_hz = 1000                        # Sine frequency (1 kHz)
duration_s = 0.02                  # 20 ms signal


# (2.915451895044e-4)   -> gives 0   degrees
# (1.5e-4)              -> gives 60  degrees
# (0)                   -> gives 90  degrees
# (-1.457725947522e-4)  -> gives 120 degrees
# (-2.915451895044e-4)  -> gives 180 degrees
manual_time_difference_s = 1.457725947522e-4

delay_samples = round(manual_time_difference_s * fs_hz)

SPEED_OF_SOUND_METERS_PER_SECOND = 343.0 # 343.0 m/s
DISTANCE_BETWEEN_MICS_M = 10e-2 # 10 cm

DEGREES_PER_RADIAN = 57.2957795 # degrees

def calculate_angle_of_arrival_rad(time_difference_s, mic_distance):

    val = np.clip((time_difference_s * SPEED_OF_SOUND_METERS_PER_SECOND) / mic_distance, -1.0, 1.0)

    return np.arccos(val)


if __name__ == '__main__':

    t = np.arange(0, duration_s, 1/fs_hz)

    # Original signal
    mic_right_signal = np.random.randn(len(t))

    # Delayed signal
    mic_left_signal = np.concatenate((np.zeros(delay_samples), mic_right_signal[:-delay_samples]))
    
    # plt.subplot(2, 1, 1)
    # plt.plot(t, mic_left_signal, label = 'Mic Left', marker='o', color = 'orange')
    # plt.legend()
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(t, mic_right_signal, label = 'Mic Right', marker='o')
    # plt.legend()

    # plt.grid()
    # plt.show()
    
    # --- GCC-PHAT ---

    X1 = np.fft.fft(mic_left_signal)
    X2 = np.fft.fft(mic_right_signal)

    # Cross power spectrum
    R = X1 * np.conj(X2)

    # Partial PHAT normalization (power -0.3)
    R_phat = R / (np.abs(R) ** (-0.3) + 1e-10)

    # Inverse FFT -> cross correlation
    gcc = np.fft.ifft(R_phat)

    # Use real part
    gcc = np.real(gcc)

    # Circular Shift FFT
    gcc = np.fft.fftshift(gcc)

    lags = np.arange(-len(gcc)//2, len(gcc)//2)

    estimated_lag = lags[np.argmax(gcc)]
    time_difference_s = estimated_lag / fs_hz



    print("Estimated lag (samples):", estimated_lag)
    print("Estimated delay (seconds):", time_difference_s)
    print("Estimated delay (ms):", time_difference_s * 1000)

    angle_of_arrival_rad = calculate_angle_of_arrival_rad(time_difference_s, DISTANCE_BETWEEN_MICS_M)
    print("Angle of Arrival (rads):", angle_of_arrival_rad)
    print("Angle of Arrival (degrees):", angle_of_arrival_rad * DEGREES_PER_RADIAN)
