import dlib
import cv2
from imutils import face_utils
import numpy as np

class Mouth_extraction(object):

    def __init__(self, frame):
        self.frame = frame
        # self.detector = dlib.get_frontal_face_detector()  # bgeb el face
        # print("inside constructor")
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")  # bgeb el 68 point

    def extract_mouth_dnn(self, width, height):
        grayImg = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # print("inside function")
        DNN = "TF"
        # if DNN == "CAFFE":
        #     modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        #     configFile = "deploy.prototxt"
        #     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        # else:
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

        blob = cv2.dnn.blobFromImage(self.frame, size=(width, height))  # kefo kda
        net.setInput(blob)
        detections = net.forward()  # bytl3 el faces ele fl frame
        # print("detection shape", detections.shape)#121 face !
        # print("detection shape[2]", detections.shape[2])
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # a5r dim 2-conff, 3:7 wid ,hig
            # print("confidence", confidence)
            if confidence > 0.0:  # ba5od awl face bs w akbr wa7d
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                x2 = int(x2 * 1.25)  # bnzbt el box 3l face aktar
                y2 = int(y2 * 1.20)

                # cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #                 cv2.imshow("image", self.frame)
                #                 cv2.waitKey(50)

                landmarks = self.predictor(grayImg, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                landmarks = face_utils.shape_to_np(landmarks)  # convert to np array
                (x, y, w, h) = cv2.boundingRect(np.array([landmarks[48:68]]))  # bngeb el boundry box bta3 el mouth
                print("x y w h", x, y, w, h)
                mouth = self.frame[y:y + h, x:x + w]  # kda b2a colered 3shan da el frame el gayle asln
                # print(mouth.shape[0])
                # print(mouth.shape[1])
                try:
                    print("mouth shape inside TRY", mouth.shape)
                    mouth = cv2.resize(mouth, (25, 40))
                    mouth.reshape(3, 25, 40)
                    return mouth

                except:  # fakes el model w hangebo bl 3afya :'D
                    # print("yyyyy", y)
                    # print("xxxxx", x)
                    y = int(height / 2)  # bta3 el sora kolha mn 3nd nos el frame
                    x = int(width / 2)
                    mouth = self.frame[y:y + 30, x:x + 30]
                    print("mouth shape inside EXCEPT", mouth.shape)
                    mouth = cv2.resize(mouth, (25, 40))
                    mouth.reshape(3, 25, 40)
                    # cv2.imshow("mouth", mouth)
                    # cv2.waitKey(50)
                    return mouth
