import cv2

def check_cameras():
    print("=======================================")
    print("   CAMERA INDEX CHECKER")
    print("=======================================")
    print("Press 'n' to check the next camera.")
    print("Press 'q' to quit.")
    print("=======================================")

    # Check indices 0 through 9
    for index in range(10):
        print(f"\n[INFO] Attempting to open Camera Index: {index}...")
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print(f"[WARN] Camera {index} failed to open (or doesn't exist).")
            continue
            
        print(f"[SUCCESS] Camera {index} is working!")
        print("Viewing stream... (Focus on the window and press 'n' for next)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            # Add text to the frame so you know which index this is
            cv2.putText(frame, f"CAMERA INDEX: {index}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Camera Checker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'): # Next
                break
            if key == ord('q'): # Quit
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    check_cameras()