import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def process_hand_landmarks():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    model = load_model('models/CNN-128-rgb-aug.h5')
    labels = np.load('data/labels.npy', allow_pickle=True)
    text = ''
    sec = 0

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            sec += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                # for hand_landmarks in results.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         image,
                #         hand_landmarks,
                #         mp_hands.HAND_CONNECTIONS,
                #         mp_drawing_styles.get_default_hand_landmarks_style(),87
                hand_pixel = get_hand_pixels(image)

                draw_palm_bounding_square(image, results, text)
                if len(hand_pixel) > 0 and sec > 5:
                    text = 'Predicted: '+labels[model.predict(hand_pixel[0].reshape(1, 128, 128, 3)).argmax()]
                    sec = 0
                    
                
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

def get_frame_hand_landmark(frame, show_image=False):
    mp_hands = mp.solutions.hands
    
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.35, min_tracking_confidence=0.35) as hands:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            if show_image:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
            for hand_landmarks in results.multi_hand_landmarks:
                normalized_landmarks = hand_landmarks.landmark
                coordinates = [[lm.x, lm.y, lm.z] for lm in normalized_landmarks]
                return coordinates
    return []

def get_hand_pixels(frame):
    mp_hands = mp.solutions.hands
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            return detect_palm(image, results)
    return []


def detect_palm(image, results):
    palm_list = np.array([])
    # palm_list = np.array([np.array(cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA))])
    
    
    for hand_landmarks in results.multi_hand_landmarks:
        min_x, min_y, max_x, max_y = 1, 1, 0, 0
        for landmark in hand_landmarks.landmark:
            min_x = min(min_x, landmark.x)
            min_y = min(min_y, landmark.y)
            max_x = max(max_x, landmark.x)
            max_y = max(max_y, landmark.y)

        box_width = max_x - min_x
        box_height = max_y - min_y
        max_dimension = max(box_width, box_height)

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        padding = 10  # Define the padding size

        x_min = int(max((center_x - max_dimension / 2) * image.shape[1] - padding, 0))
        y_min = int(max((center_y - max_dimension / 2) * image.shape[0] - padding, 0))
        x_max = int(min((center_x + max_dimension / 2) * image.shape[1] + padding, image.shape[1]))
        y_max = int(min((center_y + max_dimension / 2) * image.shape[0] + padding, image.shape[0]))

        pixels_inside_box = get_pixels_inside_box(image, x_min, y_min, x_max, y_max, grayscale=False)
        if len(palm_list) == 0:
            palm_list = np.array([pixels_inside_box])
        else:
            palm_list = np.append(palm_list, [pixels_inside_box], axis=0)
    return palm_list

def draw_palm_bounding_square(image, results, text):
    for hand_landmarks in results.multi_hand_landmarks:
        min_x, min_y, max_x, max_y = 1, 1, 0, 0
        for landmark in hand_landmarks.landmark:
            min_x = min(min_x, landmark.x)
            min_y = min(min_y, landmark.y)
            max_x = max(max_x, landmark.x)
            max_y = max(max_y, landmark.y)

        box_width = max_x - min_x
        box_height = max_y - min_y
        max_dimension = max(box_width, box_height)

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        padding = 10
        x_min = int(max((center_x - max_dimension / 2) * image.shape[1] - padding, 0))
        y_min = int(max((center_y - max_dimension / 2) * image.shape[0] - padding, 0))
        x_max = int(min((center_x + max_dimension / 2) * image.shape[1] + padding, image.shape[1]))
        y_max = int(min((center_y + max_dimension / 2) * image.shape[0] + padding, image.shape[0]))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text_color = (255, 255, 255)  # White color
        background_color = (0, 255, 0)  # Green background
        line_type = cv2.LINE_AA

        # Write the text above the square
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x_min
        text_y = y_min - 5  # Adjust this value to control the distance between text and square
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), background_color, cv2.FILLED)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, line_type)

def draw_palm_bounding_box(image, results):
    image_width, image_height = image.shape[1], image.shape[0]
    
    for hand_landmarks in results.multi_hand_landmarks:
        min_x, min_y, max_x, max_y = 1, 1, 0, 0
        for landmark in hand_landmarks.landmark:
            min_x = min(min_x, landmark.x)
            min_y = min(min_y, landmark.y)
            max_x = max(max_x, landmark.x)
            max_y = max(max_y, landmark.y)
        min_x = int(min_x * image_width)
        min_y = int(min_y * image_height)
        max_x = int(max_x * image_width)
        max_y = int(max_y * image_height)

        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)


def get_pixels_inside_box(image, x_min, y_min, x_max, y_max, grayscale=False):
    
    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1
    
    pixels_inside_box = np.zeros((box_height, box_width, 3), dtype=np.uint8)

    for y in range(box_height):
        for x in range(box_width):
            if y + y_min < image.shape[0] and x + x_min < image.shape[1]:
                pixels_inside_box[y, x] = image[y + y_min, x + x_min]
    resized = np.array(cv2.resize(pixels_inside_box, (128, 128), interpolation=cv2.INTER_AREA))
    if grayscale:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized

# cap = cv2.VideoCapture(0)
# success, image = cap.read()
# print(get_hand_pixels(image).shape)

    
process_hand_landmarks()