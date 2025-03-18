import numpy as np
import cv2

def lucas_kanade_optical_flow(video_path, output_path=None):
   
    # Opening the video file
    cap = cv2.VideoCapture(video_path)
    
    # Checking if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Reading the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    # Creating a mask image for drawing purposes
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters for ShiTomasi corner detection
    feature_params = dict(
        maxCorners=100,       # Maximum number of corners to detect
        qualityLevel=0.3,     # Minimum quality level
        minDistance=7,        # Minimum distance between corners
        blockSize=7           # Size of window for corner detection
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),     # Size of the search window
        maxLevel=2,           # Maximum pyramid level
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
    )
    
    # Detecting initial good features to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # Set up video writer if output path is provided
    if output_path:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create random colors for visualization
    color = np.random.randint(0, 255, (100, 3))
    
    frame_count = 0
    
    # Process each frame
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Selecting good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        
        # Drawing the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # Drawing the flow vector
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % 100].tolist(), 2)
            
            # Drawing a circle at the current point
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i % 100].tolist(), -1)
        
        # Combining the frame with the flow vectors
        output_frame = cv2.add(frame, mask)
        
        # Displaying the resulting frame
        cv2.imshow('Lucas-Kanade Optical Flow', output_frame)
        
        # Writing frame to output video if specified
        if output_path:
            out.write(output_frame)
        
        # Output Break on ESC key 
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
        # Updating previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
        # Every 30 frames, refresh the tracking points to ensure we keep tracking features
        if frame_count % 30 == 0:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(old_frame)
    
    # Cleaning up
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    video_path = "OPTICAL_FLOW (2).mp4"  
    output_path = "optical_flow_output.avi"  
    
    lucas_kanade_optical_flow(video_path, output_path)
