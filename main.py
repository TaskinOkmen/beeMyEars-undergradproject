import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import queue
import collections
import socket
import time

from matplotlib.animation import FuncAnimation

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_azimuth(angle):
    timestamp = time.time()
    msg = f"{timestamp},{angle}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

# Constant Parameters
fs_hz = 16000                      # Sampling frequency

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

BLOCK_SIZE = 2048 # 8192 2048 4096
HOP_SIZE   = 480 # 2040 480  1080

# index start from 0, so channel 1 = index 0, channel 2 = index 1,
# channel 3 = index 2, etc.

LEFT_MIC_CHANNEL = 5
RIGHT_MIC_CHANNEL = 3

# assume channel 1 = left, 2 = right
LEFT_MIC_INDEX = LEFT_MIC_CHANNEL - 1
RIGHT_MIC_INDEX = RIGHT_MIC_CHANNEL - 1


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


#  buffers
BUFFER_SIZE = BLOCK_SIZE * 2

buffer_left = np.zeros(BUFFER_SIZE)
buffer_right = np.zeros(BUFFER_SIZE)

q = queue.Queue()
angles = collections.deque(maxlen=200)

def estimate_aoa_over_time_overlap(x1, x2, block_size, hop_size):
    global q

    num_samples = BUFFER_SIZE
    angles = []

    for start in range(0, num_samples - block_size, hop_size):

        end = start + block_size

        block_x1 = x1[start:end]
        block_x2 = x2[start:end]

        angle = calculate_tdoa_rad(
            block_x1,
            block_x2,
            fs_hz,
            DISTANCE_BETWEEN_MICS_M
        )

        final_angle = round(angle * DEGREES_PER_RADIAN)

        # angles.append( final_angle )
        send_azimuth(final_angle)
        q.put(final_angle)

    # if len(angles) == 0:
    #     return

    # # pick ONE stable estimate (median is best here)
    # final_angle = np.median(angles)

    # send_azimuth(final_angle)
    # q.put(final_angle)



def audio_callback(indata, frames, time, status):
    global buffer_left, buffer_right

    # assume channel 1 = left, 2 = right
    new_left = indata[:, LEFT_MIC_INDEX]
    new_right = indata[:, RIGHT_MIC_INDEX]

    buffer_left[BUFFER_SIZE//2 : BUFFER_SIZE//2 + frames] = new_left
    buffer_right[BUFFER_SIZE//2 : BUFFER_SIZE//2 + frames] = new_right

    estimate_aoa_over_time_overlap(buffer_left, buffer_right, BLOCK_SIZE, HOP_SIZE)

    # shift buffer (FIFO)
    np.copyto(buffer_left[:BLOCK_SIZE], buffer_left[BLOCK_SIZE:])
    np.copyto(buffer_right[:BLOCK_SIZE], buffer_right[BLOCK_SIZE:])


def update_plot(frame):
    """Update the plot with new azimuth angles."""
    while not q.empty():
        angles.append(q.get_nowait())
    line.set_data(range(len(angles)), list(angles))
    ax.relim()
    ax.autoscale_view()
    return line,


# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')
ax.set_xlabel('Frame')
ax.set_ylabel('Azimuth (deg)')
ax.set_ylim(-100, 100)
ax.grid(True)

ani = FuncAnimation(fig, update_plot, interval=25, blit=True)


# start stream
with sd.InputStream(
    samplerate=fs_hz,
    channels=8,           # your RASP-LC has 8 channels
    dtype='int16',
    blocksize=BLOCK_SIZE,
    callback=audio_callback,
):
    print("Listening... Press Ctrl+C to stop")
    plt.show()