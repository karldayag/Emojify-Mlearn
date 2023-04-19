import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist = {0:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\angry.png",
              1:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\disgusted.png",
              2:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\fearful.png",
              3:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\happy.png",
              4:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\neutral.png",
              5:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\sad.png",
              6:r"C:\Users\Karl Joshua\Documents\Downloads\emoji-creator-project-code\emojis\emojis\surpriced.png"}
cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier('C:/Users/Karl Joshua/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # crop region of interest (ROI) from gray-scale frame
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        emoji = cv2.imread(emoji_dist[maxindex], cv2.IMREAD_UNCHANGED)
        emoji = cv2.resize(emoji, (w, h))

        alpha_s = emoji[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y:y+ h, x-130:x-130 + w, c] = (alpha_s * emoji[:, :, c] +
                                          alpha_l * frame[y:y + h, x-130:x-130 + w, c])

    cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


