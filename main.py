import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
import constant_params as cs
import kde_processing as kdep
import queue
import collections
import socket
import threading
import azimuth_processing as asp
import time
import signal
import sys



# assume channel 1 = left, 2 = right
MIC_A_INDEX = cs.MIC_A_CHANNEL - 1
MIC_B_INDEX = cs.MIC_B_CHANNEL - 1
MIC_C_INDEX = cs.MIC_C_CHANNEL - 1

RFCOMM_CHANNEL = 4  # locked — matches Android RFCOMM_CHANNEL constant

#  buffers
BUFFER_SIZE = cs.BLOCK_SIZE * 2

buffer_mic_a = np.zeros(BUFFER_SIZE)
buffer_mic_b = np.zeros(BUFFER_SIZE)
buffer_mic_c = np.zeros(BUFFER_SIZE)

AOA_STARTS = range(0, BUFFER_SIZE - cs.BLOCK_SIZE, cs.HOP_SIZE)

bt_queue = queue.Queue(maxsize=200)

plot_queue = queue.Queue()
angles = collections.deque(maxlen=200)

server_sock = socket.socket(
    socket.AF_BLUETOOTH,
    socket.SOCK_STREAM,
    socket.BTPROTO_RFCOMM
)

# Fix Problem 2 — lets the port be reused immediately after process dies
# server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_sock.bind(("9C:B6:D0:67:63:0C", RFCOMM_CHANNEL))
server_sock.listen(1)
print(f"Waiting for phone to connect on channel {RFCOMM_CHANNEL}...")

client_sock, client_info = server_sock.accept()
print(f"✅ Phone connected from: {client_info}")



# -----------------------------
# CLEAN SHUTDOWN — Fix Problem 1
# -----------------------------

def shutdown(sig=None, frame=None):
    print("\nShutting down...")
    bt_queue.put(None)          # stop BT thread
    bt_thread.join(timeout=2)
    try:
        client_sock.close()
        print("Closed BLE client socket")
    except:
        pass
    try:
        server_sock.close()
        print("Closed BLE server socket")
    except:
        pass
    plt.close('all')
    sys.exit(0)

# catch Ctrl+C properly on Ubuntu even inside plt.show()
signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)



# -----------------------------
# BT SEND QUEUE — non-blocking from audio callback
# -----------------------------

def audio_callback(indata, frames, time, status):
    global buffer_mic_a, buffer_mic_b, buffer_mic_c

    # shift old data left by exactly `frames` samples (in-place, no allocation)
    # first half                  = second half
    buffer_mic_a[:-cs.BLOCK_SIZE] = buffer_mic_a[cs.BLOCK_SIZE:]
    buffer_mic_b[:-cs.BLOCK_SIZE] = buffer_mic_b[cs.BLOCK_SIZE:]
    # buffer_mic_c[:-cs.BLOCK_SIZE] = buffer_mic_c[cs.BLOCK_SIZE:]

    # append new chunk at the end
    buffer_mic_a[-cs.BLOCK_SIZE:] = indata[:cs.BLOCK_SIZE, MIC_A_INDEX]
    buffer_mic_b[-cs.BLOCK_SIZE:] = indata[:cs.BLOCK_SIZE, MIC_B_INDEX]
    # buffer_mic_c[-cs.BLOCK_SIZE:] = indata[:cs.BLOCK_SIZE, MIC_C_INDEX]

    estimate_aoa_over_time_overlap(buffer_mic_a, buffer_mic_b, cs.BLOCK_SIZE, cs.HOP_SIZE)



def estimate_aoa_over_time_overlap(x1_left, x2_right, block_size, hop_size):
    num_samples = BUFFER_SIZE

    for start in AOA_STARTS:

        end = start + block_size

        block_x1 = x1_left[start:end]
        block_x2 = x2_right[start:end]

        angle_rad = asp.calculate_tdoa_rad(
            block_x1,
            block_x2,
            cs.fs_hz,
            cs.DISTANCE_BETWEEN_MICS_M
        )

        final_angle = round(angle_rad * cs.DEGREES_PER_RADIAN)

        send_azimuth(final_angle)
        plot_queue.put(final_angle)



def send_azimuth(angle):
    try:
        bt_queue.put_nowait(angle)
    except queue.Full:
        pass



def bt_sender_thread():
    while True:
        try:
            angle = bt_queue.get(timeout=1.0)
            if angle is None:
                break
            msg = f"angle,{angle}\n"
            client_sock.send(msg.encode("utf-8"))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"BT send error: {e}")
            break



def update_plot(frame):
    """Update the plot with new azimuth angles."""
    while not plot_queue.empty():
        angles.append(plot_queue.get_nowait())
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

bt_thread = threading.Thread(target=bt_sender_thread, daemon=True)
bt_thread.start()

# -----------------------------
# START STREAM
# -----------------------------

try:
    with sd.InputStream(
        samplerate=cs.fs_hz,
        channels=8,
        dtype='int16',
        blocksize=cs.BLOCK_SIZE,
        callback=audio_callback,
    ):
        print("Listening... Ctrl+C or close plot window to stop")
        plt.show()
finally:
    shutdown()