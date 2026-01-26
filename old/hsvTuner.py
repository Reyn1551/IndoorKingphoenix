import cv2
import numpy as np

def nothing(x):
    pass

def hsv_tuner():
    print("=======================================")
    print("       HSV COLOR TUNER")
    print("=======================================")
    
    # 1. Select Camera
    try:
        cam_idx = int(input("Enter Camera Index to tune (e.g., 0, 1, 2): "))
    except ValueError:
        cam_idx = 0
        
    cap = cv2.VideoCapture(cam_idx)
    cap.set(3, 320) # Set low resolution for speed
    cap.set(4, 240)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    # 2. Create Window and Trackbars
    cv2.namedWindow('HSV Tuner')
    cv2.createTrackbar('H Min', 'HSV Tuner', 0, 179, nothing)
    cv2.createTrackbar('S Min', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('V Min', 'HSV Tuner', 0, 255, nothing)
    
    cv2.createTrackbar('H Max', 'HSV Tuner', 179, 179, nothing)
    cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)

    # Set default starting positions (Generic Values)
    cv2.setTrackbarPos('H Min', 'HSV Tuner', 0)
    cv2.setTrackbarPos('S Min', 'HSV Tuner', 100)
    cv2.setTrackbarPos('V Min', 'HSV Tuner', 100)

    print("\n[INFO] Adjust sliders until the object is white and background is black.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Blur slightly to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current positions of all trackbars
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')

        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Create Mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Show Result
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Stack images for display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack([frame, mask_bgr, result])
        
        # Resize to fit screen if needed
        cv2.imshow('HSV Tuner', stacked)

        # Print values to console for easy copying
        print(f"LOWER: [{h_min}, {s_min}, {v_min}]  ||  UPPER: [{h_max}, {s_max}, {v_max}]", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Final Print
    print("\n\nFINAL VALUES TO COPY:")
    print(f"lower = [{h_min}, {s_min}, {v_min}]")
    print(f"upper = [{h_max}, {s_max}, {v_max}]")

if __name__ == "__main__":
    hsv_tuner()