import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initial zoom factor
zoom_factor = 1.0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Get image dimensions
    height, width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert to pixel values
            thumb_tip_x = int(thumb_tip.x * width)
            thumb_tip_y = int(thumb_tip.y * height)
            index_finger_tip_x = int(index_finger_tip.x * width)
            index_finger_tip_y = int(index_finger_tip.y * height)

            # Calculate the distance between thumb and index finger
            distance = np.sqrt((index_finger_tip_x - thumb_tip_x) ** 2 + (index_finger_tip_y - thumb_tip_y) ** 2)

            # Determine zoom factor based on the distance
            zoom_factor = max(1.0, min(3.0, distance / 100))

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Zoom the image
    center_x, center_y = width // 2, height // 2
    zoomed_width, zoomed_height = int(width / zoom_factor), int(height / zoom_factor)

    # Crop the region of interest
    left = max(center_x - zoomed_width // 2, 0)
    right = min(center_x + zoomed_width // 2, width)
    top = max(center_y - zoomed_height // 2, 0)
    bottom = min(center_y + zoomed_height // 2, height)

    cropped_image = image[top:bottom, left:right]

    # Resize back to the original image size
    zoomed_image = cv2.resize(cropped_image, (width, height))

    # Display the zoomed image
    cv2.imshow('Camera', zoomed_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
