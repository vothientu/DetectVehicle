import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('B.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0
temp =0

trackermoto = Tracker()
trackercar = Tracker()
trackertruck = Tracker()
trackerbus = Tracker()


cy1 = 222
cy2 = 368
offset = 6

# Thêm một biến flag để kiểm soát trạng thái dừng lại
stop_flag = False

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    listcar = []
    listtruck = []
    listmotorcycle = []
    listbus = []
             
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            listcar.append([x1, y1, x2, y2])
        if 'truck' in c:
            listtruck.append([x1, y1, x2, y2])
        if 'motorcycle' in c:
            listmotorcycle.append([x1, y1, x2, y2])
        if 'bus' in c:
            listbus.append([x1, y1, x2, y2])
    
    bbox_car= trackercar.update(listcar)
    for bbox in bbox_car:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "car", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    bbox_truck= trackertruck.update(listtruck)
    for bbox in bbox_truck:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "truck", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)  

    bbox_motorcycle= trackermoto.update(listmotorcycle)
    for bbox in bbox_motorcycle:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "motorcycle", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) 
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)      
    bbox_bus= trackerbus.update(listbus)
    for bbox in bbox_bus:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.putText(frame, "bus", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) 
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (0, 230), (1000, 230), ( 0,255, 255), 2)
    cv2.line(frame, (0, 350), (1000, 350), (0, 255, 255), 2)
    cv2.imshow("RGB", frame)

    if stop_flag:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        stop_flag = True

cap.release()
cv2.destroyAllWindows()
