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

audio_block_queue = queue.Queue(maxsize=4)

server_sock = socket.socket(
    socket.AF_BLUETOOTH,
    socket.SOCK_STREAM,
    socket.BTPROTO_RFCOMM
)

# Fix Problem 2 — lets the port be reused immediately after process dies
# server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_sock.bind((cs.BLE_MAC, RFCOMM_CHANNEL))
server_sock.listen(1)
print(f"Waiting for phone to connect on channel {RFCOMM_CHANNEL}...")

client_sock, client_info = server_sock.accept()
print(f"✅ Phone connected from: {client_info}")



# -----------------------------
# CLEAN SHUTDOWN — Fix Problem 1
# -----------------------------

def shutdown(sig=None, frame=None):
    print("\nShutting down...")
    audio_block_queue.put(None)
    processing_thread.join(timeout=2)
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



def audio_callback(indata, frames, time_info, status):

  # Minimal, bounded work: 3 small copies + a non-blocking enqueue
  block_a = indata[:cs.BLOCK_SIZE, MIC_A_INDEX].copy()
  block_b = indata[:cs.BLOCK_SIZE, MIC_B_INDEX].copy()
  block_c = indata[:cs.BLOCK_SIZE, MIC_C_INDEX].copy()

  try:
    audio_block_queue.put_nowait((block_a, block_b, block_c))
  except queue.Full:
    # Drop rather than block the realtime thread.
    pass



def estimate_aoa_over_time_overlap(x1_left, x2_right, block_size, hop_size, angle_offset_rad, mic_distance_m, candidate_angles_rad):
  num_samples = BUFFER_SIZE

  for start in AOA_STARTS:

    end = start + block_size

    block_x1 = x1_left[start:end]
    block_x2 = x2_right[start:end]

    angle_rad = asp.calculate_tdoa_rad(
        block_x1,
        block_x2,
        cs.fs_hz,
        mic_distance_m
    )

    c1, c2 = kdep.get_candidate_angles_rad(angle_rad, angle_offset_rad)

    candidate_angles_rad.extend([c1, c2])

    # final_angle = round((angle_rad + angle_offset_rad) * cs.DEGREES_PER_RADIAN) % 360

    # send_azimuth(final_angle)
    # plot_queue.put(final_angle)



def processing_thread_fn():
    buffer_mic_a = np.zeros(BUFFER_SIZE)
    buffer_mic_b = np.zeros(BUFFER_SIZE)
    buffer_mic_c = np.zeros(BUFFER_SIZE)

    while True:
      item = audio_block_queue.get()
      if item is None:
        break  # shutdown signal

      block_a, block_b, block_c = item

      # same sliding-window bookkeeping as before, just off the audio thread now
      # first half                  = second half
      buffer_mic_a[:-cs.BLOCK_SIZE] = buffer_mic_a[cs.BLOCK_SIZE:]
      buffer_mic_b[:-cs.BLOCK_SIZE] = buffer_mic_b[cs.BLOCK_SIZE:]
      buffer_mic_c[:-cs.BLOCK_SIZE] = buffer_mic_c[cs.BLOCK_SIZE:]

      buffer_mic_a[-cs.BLOCK_SIZE:] = block_a
      buffer_mic_b[-cs.BLOCK_SIZE:] = block_b
      buffer_mic_c[-cs.BLOCK_SIZE:] = block_c

      candidate_angles_rad = []

      estimate_aoa_over_time_overlap(buffer_mic_a, buffer_mic_b, cs.BLOCK_SIZE, cs.HOP_SIZE, kdep.a_b_offset_angle_rad, cs.DISTANCE_BETWEEN_MICS_A_B_M, candidate_angles_rad)

      estimate_aoa_over_time_overlap(buffer_mic_b, buffer_mic_c, cs.BLOCK_SIZE, cs.HOP_SIZE, kdep.b_c_offset_angle_rad, cs.DISTANCE_BETWEEN_MICS_B_C_M, candidate_angles_rad)

      estimate_aoa_over_time_overlap(buffer_mic_a, buffer_mic_c, cs.BLOCK_SIZE, cs.HOP_SIZE, kdep.a_c_offset_angle_rad, cs.DISTANCE_BETWEEN_MICS_A_C_M, candidate_angles_rad)


      candidate_angles_rad = np.mod(candidate_angles_rad, 2*np.pi)

      best_angle_deg = kdep.estimate_azimuth_360_normal_deg(candidate_angles_rad, resolution=180)

      final_angle = round(best_angle_deg)

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
ax.set_ylim(0, 360)
ax.grid(True)

ani = FuncAnimation(fig, update_plot, interval=25, blit=True)

bt_thread = threading.Thread(target=bt_sender_thread, daemon=True)
bt_thread.start()

processing_thread = threading.Thread(target=processing_thread_fn, daemon=True)
processing_thread.start()

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