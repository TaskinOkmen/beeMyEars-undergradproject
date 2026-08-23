import numpy as np
import constant_params as cs

MAX_LAG_SAMPLES = int(np.ceil(cs.DISTANCE_BETWEEN_MICS_M * cs.fs_hz / cs.SPEED_OF_SOUND_METERS_PER_SECOND))


def calculate_tdoa_rad(mic_left_signal, mic_right_signal, fs_hz, mic_distance_m):

  time_difference_s = calculate_time_difference_s(mic_left_signal, mic_right_signal, fs_hz)

  angle_of_arrival_rad = calculate_angle_of_arrival_rad(time_difference_s, mic_distance_m)

  return angle_of_arrival_rad



def calculate_time_difference_s(mic_left_signal, mic_right_signal, fs_hz):
  num_samples = len(mic_left_signal)

  X1 = np.fft.rfft(mic_left_signal)
  X2 = np.fft.rfft(mic_right_signal)

  R = X1 * np.conj(X2)
  R_phat = R / ((np.abs(R) + 1e-10) ** (-0.3) + 1e-10)

  gcc = np.fft.irfft(R_phat, n=num_samples)  # NOT shifted

  # index 0        -> lag 0
  # index 1..k      -> lag +1..+k
  # index N-k..N-1  -> lag -k..-1
  neg_part = gcc[-MAX_LAG_SAMPLES:]        # lags -k .. -1
  pos_part = gcc[:MAX_LAG_SAMPLES + 1]     # lags 0 .. +k
  gcc_window = np.concatenate([neg_part, pos_part])

  peak_index = np.argmax(gcc_window)
  estimated_lag = peak_index - MAX_LAG_SAMPLES

  return estimated_lag / fs_hz



def calculate_angle_of_arrival_rad(time_difference_s, mic_distance_m):

  val = np.clip((time_difference_s * cs.SPEED_OF_SOUND_METERS_PER_SECOND) / mic_distance_m, -1.0, 1.0)

  return np.arcsin(val)



def estimate_aoa_over_time_overlap_to_array(mic_left_signal, mic_right_signal, fs_hz, mic_distance_m, block_size=cs.BLOCK_SIZE, hop_size=cs.HOP_SIZE):

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

      angles.append(angle * cs.DEGREES_PER_RADIAN)

  return np.array(angles)