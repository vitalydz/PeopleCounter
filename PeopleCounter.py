from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*


cap = cv2.VideoCapture("../Videos/1.mp4")


model = YOLO("../YoloWeights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "plant", "bed",
              "table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('mask.png')

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [150, 170, 490, 170]
limitsDown = [840, 650, 1120, 650]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread("Graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0,0))

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,0,255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=5)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 20 < cy < limitsUp[1] + 20:
            if totalCountUp.count(Id) == 0:
                totalCountUp.append(Id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,0,255), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[1] + 20:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0,0,255), 5)

        # cvzone.putTextRect(img, f' Count = {len(totalCount)}', (50, 50))
        cv2.putText(img, str(len(totalCountUp)), (194, 90), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 8)
        cv2.putText(img, str(len(totalCountDown)), (710, 90), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 8)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)