import cv2
from djitellopy import tello
import cvzone

thresh = 0.55
nmsThresh = 0.2
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

me = tello.Tello()
me.connect()
print("Battery: ", me.get_battery(), " %")
me.streamon()

x_cord = 760
y_cord = 0

cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("Video Feed", x_cord, y_cord)

while True:
    # success, img = cap.read()
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thresh, nmsThreshold=nmsThresh)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except:
        pass

    cv2.imshow("Video Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        print("Exit Command Has Been Executed.")
        break

cv2.destroyAllWindows()
me.land()
me.reboot()
