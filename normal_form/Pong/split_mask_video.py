import cv2
import numpy as np
import os
vidcap = cv2.VideoCapture('./recording/vid_mask_0.mp4')
success,image = vidcap.read()

if os.path.isdir("frames_wm"):
    os.system("rm -rf frames_wm")
os.system("mkdir frames_wm")

count = 0

masks = np.loadtxt("./recording/mask_pos_0.out")
mask_pointer = 0
conf_scores = np.loadtxt("./recording/mask_probs_0.out")
eps_len = np.loadtxt("./recording/eps_len_0.out")

while success and count < eps_len:
  if mask_pointer < len(masks) and count == masks[mask_pointer]:
    image= cv2.copyMakeBorder(image,0,20,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    mask_pointer += 1
  else:
    image= cv2.copyMakeBorder(image,0,20,0,0,cv2.BORDER_CONSTANT,value=(0,0,255))
  

  conf_score = str(conf_scores[count][1])
  # font
  font = cv2.FONT_HERSHEY_SIMPLEX
  
  # org
  org = (10, 200)
  
  # fontScale
  fontScale = 1
   
  # Blue color in BGR
  color = (255, 0, 0)
  
  # Line thickness of 2 px
  thickness = 1
   
  # Using cv2.putText() method
  image = cv2.putText(image, conf_score, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

  cv2.imwrite("./frames_wm/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  #success = False
  count += 1


