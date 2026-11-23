import os
import math
import cv2
import mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=1)
face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

FOLDER = os.path.dirname(os.path.abspath(__file__))

# Картинки мемов: one.jpg, two.jpg, fist.jpg, ok.jpg, smile.jpg, mouth.jpg в папке Da
memes = {}
for name in ["one", "two", "fist", "ok", "smile", "mouth"]:
    img = cv2.imread(os.path.join(FOLDER, name + ".jpg"))
    if img is not None:
        memes[name] = img

LABELS = {
    "one": "ONE!",
    "two": "TWO!",
    "fist": "FIST!",
    "ok": "OK!",
    "smile": "SMILE!",
    "mouth": "MOUTH OPEN!",
}


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def hand_gesture(lm):
    index = finger_up(lm, 8, 6)
    middle = finger_up(lm, 12, 10)
    ring = finger_up(lm, 16, 14)
    pinky = finger_up(lm, 20, 18)

    # Большой и указательный соединены кольцом, остальные подняты
    if dist(lm[4], lm[8]) < 0.05 and middle and ring and pinky:
        return "ok"
    if index and not middle and not ring and not pinky:
        return "one"
    if index and middle and not ring and not pinky:
        return "two"
    if not index and not middle and not ring and not pinky:
        return "fist"
    return None


def face_gesture(fl):
    # Точки губ: 13 и 14 — внутренние верх и низ, 61 и 291 — уголки рта
    # Точки 234 и 454 — края лица, нужны для сравнения размеров
    face_width = dist(fl[234], fl[454])
    mouth_open = dist(fl[13], fl[14]) / face_width
    mouth_wide = dist(fl[61], fl[291]) / face_width

    if mouth_open > 0.12:
        return "mouth"
    if mouth_wide > 0.45:
        return "smile"
    return None


while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gesture = None

    hand_result = hands.process(rgb)
    if hand_result.multi_hand_landmarks:
        gesture = hand_gesture(hand_result.multi_hand_landmarks[0].landmark)

    if gesture is None:
        face_result = face.process(rgb)
        if face_result.multi_face_landmarks:
            gesture = face_gesture(face_result.multi_face_landmarks[0].landmark)

    if gesture:
        cv2.putText(frame, LABELS[gesture], (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        if gesture in memes:
            cv2.imshow("Мем", memes[gesture])

    cv2.imshow("Камера", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()