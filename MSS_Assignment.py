import sys
import cv2
import numpy as np
import json

# Set the block size and search window size
block_size = 16
search_window_size = 16

dx = []
dy = []

dx.append((0, 0, 0, 1, -1))
dy.append((0, 1, -1, 0, 0))

dx.append((0, -2, 0, 2, 0, -1, 1, 1, -1))
dy.append((0, 0, -2, 0, 2, 1, -1, 1, -1))



def ds(ip, pt, cf, sf, bs, lvl):
    if lvl == 0:
      return pt

    x,y = pt
    
    # Initialize the motion vector to (0, 0)
    motion_vector = pt
    
    # Initialize the minimum SSD to a large value
    min_ssd = float('inf')

    for i in range (0, 4*lvl + 1):
      x1 = x + dx[lvl - 1][i]*bs
      y1 = y + dy[lvl - 1][i]*bs
      
                
      try :
        
        # Extract the candidate block from the search frame
        block = cf[ip[1]:ip[1]+bs, ip[0]:ip[0]+bs, :]
        candidate_block = sf[y1:y1+bs, x1:x+bs, :]

        # Compute the SSD between the block and the candidate block
        ssd = np.sum((block - candidate_block)**2)
            
        # Check if the SSD is smaller than the current minimum SSD
        if ssd < min_ssd:
            # Update the motion vector and the minimum SSD
            motion_vector = (x, y)
            min_ssd = ssd
      except Exception as e:
        continue

    if motion_vector == pt:
      return ds(ip, pt, cf, sf, bs, lvl - 1)

    return ds(ip, motion_vector, cf, sf, bs, lvl)

# Load the MP4 file using OpenCV
cap = cv2.VideoCapture('video.mp4')

# Get the video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a 4D NumPy array to store the video frames
video_array = np.empty((frames, height, width, 3), np.dtype('uint8'))

# Loop through the video frames and store them in the 4D array
frame_idx = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        video_array[frame_idx] = frame
        frame_idx += 1
    else:
        break

# Release the video capture object
cap.release()

# Initialize the compressed video array
compressed_video = []

# Loop through the frames and perform motion estimation and compensation


for i in range(0, frames - 2, 3):
    curr_frame = video_array[i]
    compressed_video.append(curr_frame)

    next_frame_1 = video_array[i+1]
    next_frame_2 = video_array[i+2]

    motion_vector = []
    # Compute the motion vectors for each block using diamond search block matching
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            motion_vector.append(ds((x,y), (x,y), curr_frame ,next_frame_1,block_size, 2))

    compressed_video.append(tuple(motion_vector))
    motion_vector.clear()

    # Compute the motion vectors for each block using diamond search block matching
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            motion_vector.append(ds((x,y), (x,y), curr_frame ,next_frame_2,block_size, 2))

    compressed_video.append(tuple(motion_vector))

    print(i*100//frames, "%")



# Open a file for writing
with open("compressed.mvd", 'w') as f:
    # Use pickle to serialize and save the object to the file
    json.dump(compressed_video, f)

