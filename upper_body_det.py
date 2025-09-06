import cv2
import os

# Load the cascade classifier for upper body detection
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Function to detect upper bodies in a frame
def detect_objects(frame, cascade_classifier, output_folder, frame_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Save the frame with detected upper body
        cv2.imwrite(os.path.join(output_folder, f"detected_upper_body_{frame_count}.png"), frame)
    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

    

# Create a window to display the results
cv2.namedWindow('Upper Body Detection')

# Create a folder to save detected upper bodies
output_folder = 'detected_upper_bodies'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect upper bodies in the frame
    frame = detect_objects(frame, cascade_classifier, output_folder, frame_count)
    frame_count += 1

    # Display the frame
    cv2.imshow('Upper Body Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
