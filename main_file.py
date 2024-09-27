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
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

me = tello.Tello()
me.connect()
print("Battery: ", me.get_battery(), " %")
me.streamoff()
me.streamon()

me.takeoff()
me.move_up(120)

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

    me.send_rc_control(0, 0, 0, 0)

    cv2.imshow("Video Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        me.move_forward(50)
        print("Forward Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        me.move_back(50)
        print("Back Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        me.move_left(25)
        print("Left Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        me.move_right(25)
        print("Right Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        me.move_up(25)
        print("Up Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        me.move_down(25)
        print("Down Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('j'):
        me.rotate_clockwise(90)
        print("Rotate Clockwise Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('k'):
        me.rotate_counter_clockwise(90)
        print("Rotate Counter-Clockwise Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('x'):
        print("Exit Command Has Been Executed.")
        break

cv2.destroyAllWindows()
me.land()
me.reboot()
