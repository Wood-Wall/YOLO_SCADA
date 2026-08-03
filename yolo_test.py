import cv2                  # opencv
from ultralytics import YOLO    #YOLO
"""
测试Yolo +open cv, 显示物体识别
"""

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    while True:
       ret, frame = cap.read()
       if ret == False:
           break
       results = model(frame)
       result = results[0]
       annotated = result.plot()
       cv2.imshow("YOLO",annotated)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    cap.release()
    cv2.destoryAllWindows()

if __name__ == "__main__":
    main()