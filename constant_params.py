# --- Configuration Parameters ---

BLOCK_SIZE = 2048 # 8192 2048 4096
HOP_SIZE   = 512 # 2040 480  1080

# index start from 0, so channel 1 = index 0, channel 2 = index 1,
# channel 3 = index 2, etc.
#                 left <--> right
# Mic positioning A--- 10 cm ---B
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

MIC_A_CHANNEL = 5
MIC_B_CHANNEL = 6
MIC_C_CHANNEL = 3

DISTANCE_BETWEEN_MICS_A_B_M = 10e-2 # 10 cm
DISTANCE_BETWEEN_MICS_B_C_M = 13.5e-2 # 13.5 cm
DISTANCE_BETWEEN_MICS_A_C_M = 10e-2 # 10 cm

# --- Configuration Parameters ---

# --- Constant Parameters ---

fs_hz = 16000                      # Sampling frequency

SPEED_OF_SOUND_METERS_PER_SECOND = 343.0 # 343.0 m/s

DEGREES_PER_RADIAN = 57.2957795 # degrees

# --- Constant Parameters ---
