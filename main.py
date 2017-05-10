import numpy as np
import cv2

def checkMovement(prev, now):
    grey_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    grey_now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)


    features = cv2.goodFeaturesToTrack(grey_prev, maxCorners = 3, qualityLevel = 0.01, minDistance = 10); # min distance between features
    new_positions, x, y = cv2.calcOpticalFlowPyrLK(grey_prev, grey_now, features)

    changes = new_positions - features
    changes = changes.reshape(changes.shape[0],2)
    return (np.linalg.norm(changes, axis = 1) > 50).any()
    

cap = cv2.VideoCapture(0)
ret, prev_image = cap.read()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    movement = checkMovement(prev_image,frame)
    # Display the resulting frame

    if(movement):
    	cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_image = frame

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()