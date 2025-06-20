import cv2
import mediapipe as mp
from playsound import playsound
import webbrowser

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Gesture detection function
def detect_gesture(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    fingers = []

    # Check if fingers are open
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Check thumb (horizontal movement)
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    total_fingers = fingers.count(1)

    # Match gestures
    if total_fingers == 0:
        return "Fist"
    elif total_fingers == 5:
        return "Open Palm"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [0, 0, 0, 0, 1]:
        return "Thumbs Up"
    else:
        return "Unknown"

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip and convert image
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    gesture_name = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture_name = detect_gesture(hand_landmarks)
            cv2.putText(image, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            # Actions based on gesture
            if gesture_name == "Thumbs Up":
                webbrowser.open("https://www.google.com")
            elif gesture_name == "Peace":
                playsound("beep.mp3")  

    cv2.imshow("Hand Gesture Recognition", image)

    key = cv2.waitKey(1) & 0xFF
    print(f"Pressed key: {chr(key) if key != 255 else 'None'}")  
    if key == ord('q') or key == 27:  # 'q' or ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
