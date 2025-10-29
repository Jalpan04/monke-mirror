import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

# --- Constants for MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for eyes
LEFT_EYE_LANDMARKS = [386, 374, 380, 373, 362, 263]
RIGHT_EYE_LANDMARKS = [159, 145, 153, 144, 133, 33]

# --- LOGIC THRESHOLDS (CALIBRATED FOR YOU!) ---
EAR_THRESHOLD_RELAXED = 0.32  # Your "open" eye threshold
EAR_THRESHOLD_CLOSED = 0.29  # Your "closed" eye threshold
# -----------------------------------------------
EAR_THRESHOLD_OPEN = 0.35  # For "shocked"
SHOCKED_MAR_THRESHOLD = 0.3
NEUTRAL_MAR_THRESHOLD = 0.1
SMILE_MHR_THRESHOLD = 0.7
GESTURE_BUFFER_SIZE = 10


# --- Helper Functions ---

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def get_ear(eye_landmarks, face_landmarks):
    try:
        v1 = get_distance(face_landmarks[eye_landmarks[0]], face_landmarks[eye_landmarks[1]])
        v2 = get_distance(face_landmarks[eye_landmarks[2]], face_landmarks[eye_landmarks[3]])
        h = get_distance(face_landmarks[eye_landmarks[4]], face_landmarks[eye_landmarks[5]])
        if h == 0: return 0.0
        return (v1 + v2) / (2.0 * h)
    except:
        return 0.0


def get_face_metrics(face_lm):
    if not face_lm:
        return {'left_ear': 0.3, 'right_ear': 0.3, 'mar': 0, 'mhr': 0, 'face_width': 0}

    left_ear = get_ear(LEFT_EYE_LANDMARKS, face_lm)
    right_ear = get_ear(RIGHT_EYE_LANDMARKS, face_lm)

    left_eye_pt = face_lm[33]
    right_eye_pt = face_lm[263]
    face_width = get_distance(left_eye_pt, right_eye_pt)

    if face_width < 1e-6:
        return {'left_ear': left_ear, 'right_ear': right_ear, 'mar': 0, 'mhr': 0, 'face_width': 0}

    top_lip_lm = face_lm[13]
    bottom_lip_lm = face_lm[14]
    vertical_lip_dist = get_distance(top_lip_lm, bottom_lip_lm)
    mar = vertical_lip_dist / face_width

    left_corner_lm = face_lm[61]
    right_corner_lm = face_lm[291]
    horizontal_lip_dist = get_distance(left_corner_lm, right_corner_lm)
    mhr = horizontal_lip_dist / face_width

    return {'left_ear': left_ear, 'right_ear': right_ear, 'mar': mar, 'mhr': mhr, 'face_width': face_width}


def get_hand_pose_states(hand_lm):
    """Checks for 'idea' and 'thumbsup' poses."""
    if not hand_lm:
        return {'is_index_up': False, 'is_thumbs_up': False}

    index_tip_y = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip_y = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip_y = hand_lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip_y = hand_lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip_y = hand_lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip_y = hand_lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip_y = hand_lm[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip_y = hand_lm[mp_hands.HandLandmark.PINKY_PIP].y
    thumb_tip_y = hand_lm[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_mcp_y = hand_lm[mp_hands.HandLandmark.THUMB_MCP].y

    is_index_up = (index_tip_y < index_pip_y and
                   middle_tip_y > middle_pip_y and
                   ring_tip_y > ring_pip_y and
                   pinky_tip_y > pinky_pip_y)

    is_thumbs_up = (thumb_tip_y < thumb_mcp_y and
                    index_tip_y > index_pip_y and
                    middle_tip_y > middle_pip_y and
                    ring_tip_y > ring_pip_y and
                    pinky_tip_y > pinky_pip_y)

    return {'is_index_up': is_index_up, 'is_thumbs_up': is_thumbs_up}


# --- Main Application ---

# --- Load Your Images ---
try:
    img_idea = cv2.imread('idea.jpg')
    img_shocked = cv2.imread('shocked.jpg')
    img_chee = cv2.imread('chee.jpg')
    img_thumbsup = cv2.imread('thumbsup.jpg')
    img_wink = cv2.imread('wink.jpg')
    img_hummm = cv2.imread('hummm.jpg')

    images = {
        'idea': img_idea, 'shocked': img_shocked, 'chee': img_chee,
        'thumbsup': img_thumbsup, 'wink': img_wink, 'hummm': img_hummm,
        'default': np.zeros((500, 500, 3), dtype=np.uint8)
    }

    # --- MODIFIED: Increased output_width ---
    output_width = 640  # Set a common width for both video and image

    # Resize all expression images to this common width, maintaining aspect ratio
    for key, img in images.items():
        if img is not None:
            # Calculate new height based on desired width
            aspect_ratio = img.shape[0] / img.shape[1]
            new_height = int(output_width * aspect_ratio)
            images[key] = cv2.resize(img, (output_width, new_height))
        else:
            print(f"Warning: Could not load image for '{key}'")
            if key != 'default':
                images[key] = np.zeros((int(output_width * 1.0), output_width, 3),
                                       dtype=np.uint8)  # Fallback to a blank image
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

# --- Initialize MediaPipe ---
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Track 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# --- Stability Buffer ---
detection_buffer = deque(['chee'] * GESTURE_BUFFER_SIZE, maxlen=GESTURE_BUFFER_SIZE)
current_stable_expression = 'chee'

# --- Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image.flags.writeable = False
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb_image)
    results_face = face_mesh.process(rgb_image)

    image.flags.writeable = True

    # --- Landmark and State Initialization ---
    face_lm = None
    detected_expression = 'chee'
    eye_status_text = ""

    # --- 1. Process Face ---
    if results_face.multi_face_landmarks:
        face_lm = results_face.multi_face_landmarks[0].landmark

    # --- 2. Get Face Metrics ---
    face_metrics = get_face_metrics(face_lm)

    # --- General Eye Status Logic ---
    if face_lm:
        if (face_metrics['left_ear'] > EAR_THRESHOLD_RELAXED and
                face_metrics['right_ear'] > EAR_THRESHOLD_RELAXED):
            eye_status_text = "Eyes: Open"
        elif (face_metrics['left_ear'] < EAR_THRESHOLD_RELAXED and
              face_metrics['right_ear'] < EAR_THRESHOLD_RELAXED):
            eye_status_text = "Eyes: Closed"
        else:
            eye_status_text = "Eyes: Winking"

    # Pre-calculate face states for gestures
    is_winking = (
            (face_metrics['left_ear'] < EAR_THRESHOLD_CLOSED and face_metrics['right_ear'] > EAR_THRESHOLD_RELAXED) or
            (face_metrics['right_ear'] < EAR_THRESHOLD_CLOSED and face_metrics['left_ear'] > EAR_THRESHOLD_RELAXED)
    )
    is_smiling = face_metrics['mhr'] > SMILE_MHR_THRESHOLD

    # --- 3. Process Hands ---
    if results_hands.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):

            hand_lm = hand_landmarks.landmark
            classification = hand_handedness.classification[0]
            hand_label = classification.label
            hand_score = classification.score

            hand_poses = get_hand_pose_states(hand_lm)

            if hand_label == 'Right':
                if hand_poses['is_thumbs_up']:
                    detected_expression = 'thumbsup'
                else:
                    detected_expression = 'hummm'

            elif hand_label == 'Left':
                if hand_poses['is_index_up']:
                    detected_expression = 'idea'

            # Draw hand landmarks and label
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)
            )
            h, w, _ = image.shape
            wrist_coords = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            cx, cy = int(wrist_coords.x * w), int(wrist_coords.y * h)
            text_position = (cx - 50, cy - 20)
            cv2.putText(
                image, f'{hand_label} ({hand_score:.2f})',
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA
            )

    # --- 4. Process Face-Only Gestures (if no hand gesture was detected) ---
    if detected_expression == 'chee' and face_lm:
        is_mouth_open_v = face_metrics['mar'] > SHOCKED_MAR_THRESHOLD
        is_eyes_wide = (face_metrics['left_ear'] > EAR_THRESHOLD_OPEN and
                        face_metrics['right_ear'] > EAR_THRESHOLD_OPEN)

        if is_mouth_open_v and is_eyes_wide:
            detected_expression = 'shocked'

        is_eyes_squeezed = (face_metrics['left_ear'] < EAR_THRESHOLD_CLOSED and
                            face_metrics['right_ear'] < EAR_THRESHOLD_CLOSED)

        if is_eyes_squeezed:
            detected_expression = 'chee'
        elif is_winking:
            detected_expression = 'wink'

    # --- 5. Stability/Debounce Logic ---
    detection_buffer.append(detected_expression)
    if all(e == detection_buffer[0] for e in detection_buffer):
        current_stable_expression = detection_buffer[0]

    # --- 6. Draw Info Text ---
    cv2.putText(
        image, eye_status_text, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
    )

    # --- 7. Display Final Output ---
    output_img = images.get(current_stable_expression, images['chee'])

    # Resize webcam feed to match output_width
    cam_h, cam_w, _ = image.shape
    cam_aspect_ratio = cam_h / cam_w
    resized_cam_h = int(output_width * cam_aspect_ratio)
    image_resized = cv2.resize(image, (output_width, resized_cam_h))

    # Ensure the output image also matches the output_width for stacking
    exp_h, exp_w, _ = output_img.shape
    if exp_w != output_width:
        aspect_ratio = exp_h / exp_w
        new_height = int(output_width * aspect_ratio)
        output_img = cv2.resize(output_img, (output_width, new_height))

    combined_output = np.vstack((image_resized, output_img))
    cv2.imshow('Expression Mirror', combined_output)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC key
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()