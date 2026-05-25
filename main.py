import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import queue
import collections

from matplotlib.animation import FuncAnimation

# Constant Parameters
fs_hz = 16000                      # Sampling frequency
f_hz = 1000                        # Sine frequency (1 kHz)
duration_s = 0.1                  # 20 ms signal


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
    #angles = []

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

        #angles.append(angle)
        #print(angle * DEGREES_PER_RADIAN)
        q.put(angle * DEGREES_PER_RADIAN)



def audio_callback(indata, frames, time, status):
    global buffer_left, buffer_right

    # assume channel 1 = left, 2 = right
    new_left = indata[:, 1]
    new_right = indata[:, 2]

    buffer_left[BUFFER_SIZE//2 : BUFFER_SIZE//2 + frames] = new_left
    buffer_right[BUFFER_SIZE//2 : BUFFER_SIZE//2 + frames] = new_right

    estimate_aoa_over_time_overlap(buffer_left, buffer_right, BLOCK_SIZE, HOP_SIZE)

    # shift buffer (FIFO)
    np.copyto(buffer_left[:BLOCK_SIZE], buffer_left[BLOCK_SIZE:])
    np.copyto(buffer_right[:BLOCK_SIZE], buffer_right[BLOCK_SIZE:])



# def update_plot(frame):
#     """This is called by matplotlib for each plot update.

#     Typically, audio callbacks happen more frequently than plot updates,
#     therefore the queue tends to contain multiple blocks of audio data.

#     """
#     global plotdata
#     while True:
#         try:
#             data = q.get_nowait()
#         except queue.Empty:
#             break
#         shift = len(data)
#         plotdata = np.roll(plotdata, -shift, axis=0)
#         plotdata[-shift:, :] = data
#     for column, line in enumerate(lines):
#         line.set_ydata(plotdata[:, column])
#     return lines


# try:
#     if args.samplerate is None:
#         device_info = sd.query_devices(args.device, 'input')
#         args.samplerate = device_info['default_samplerate']

#     length = int(args.window * args.samplerate / (1000 * args.downsample))
#     plotdata = np.zeros((length, len(args.channels)))

#     fig, ax = plt.subplots()
#     lines = ax.plot(plotdata)
#     ax.axis((0, len(plotdata), -1, 1))
#     ax.set_yticks([0])
#     ax.yaxis.grid(True)
#     ax.tick_params(bottom=False, top=False, labelbottom=False,
#                    right=False, left=False, labelleft=False)
#     fig.tight_layout(pad=0)

#     stream = sd.InputStream(
#         device=args.device, channels=max(args.channels),
#         samplerate=args.samplerate, callback=audio_callback)
#     ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
#     with stream:
#         plt.show()
# except Exception as e:
#     pass


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
    callback=audio_callback
):
    print("Listening... Press Ctrl+C to stop")
    plt.show()

# if __name__ == '__main__':

#     sr, mic_right_signal = wavfile.read("recording_ch2.wav")
 
#     sr, mic_left_signal = wavfile.read("recording_ch1.wav")

#     angles = estimate_aoa_over_time_overlap_to_array(mic_left_signal, mic_right_signal, fs_hz, DISTANCE_BETWEEN_MICS_M)

#     print(angles)
    
#     plt.plot(angles, label = 'Mic Right', marker='o')
#     plt.legend()

#     plt.grid()
#     plt.show()




# TEST CODE BELOW - NOT USED IN FINAL VERSION

# angles = []

# def test_estimate_aoa_over_time_overlap(x1, x2, block_size, hop_size):

#     num_samples = BUFFER_SIZE
    

#     for start in range(0, num_samples - block_size, hop_size):

#         end = start + block_size

#         block_x1 = x1[start:end]
#         block_x2 = x2[start:end]

#         angle = calculate_tdoa_rad(
#             block_x1,
#             block_x2,
#             fs_hz,
#             DISTANCE_BETWEEN_MICS_M
#         )

#         angles.append(angle * DEGREES_PER_RADIAN)
#         #print(angle * DEGREES_PER_RADIAN)

# def test_audio_callback(left_signal, right_signal, frames):
#     global buffer_left, buffer_right

#     # assume channel 1 = left, 2 = right
#     new_left = left_signal
#     new_right = right_signal

#     buffer_left[BUFFER_SIZE//2 : BUFFER_SIZE//2 + frames] = new_left
#     buffer_right[BUFFER_SIZE//2 : BUFFER_SIZE//2 + frames] = new_right

#     test_estimate_aoa_over_time_overlap(buffer_left, buffer_right, BLOCK_SIZE, HOP_SIZE)

#     # shift buffer (FIFO)
#     np.copyto(buffer_left[:BLOCK_SIZE], buffer_left[BLOCK_SIZE:])
#     np.copyto(buffer_right[:BLOCK_SIZE], buffer_right[BLOCK_SIZE:])

# if __name__ == '__main__':

#     sr, mic_right_signal = wavfile.read("recording_ch2.wav")
 
#     sr, mic_left_signal = wavfile.read("recording_ch1.wav")


#     for i in range(0, len(mic_left_signal) - BLOCK_SIZE, HOP_SIZE):
#         test_audio_callback(mic_left_signal[i:i+BLOCK_SIZE], mic_right_signal[i:i+BLOCK_SIZE], BLOCK_SIZE)
    
#     plt.plot(angles, label = 'Mic Right', marker='o')
#     plt.legend()

#     plt.grid()
#     plt.show()