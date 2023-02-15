import cv2
import dlib

# Load the pre-trained face detection model
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmark detection model
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector(gray)
    
    # Loop through each face
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = landmark_predictor(gray, face)
        
        # Loop through each landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            
            # Draw a circle at the landmark position
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    
    # Show the frame
    cv2.imshow("Facial Landmarks", frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all windows
cv2.destroyAllWindows()