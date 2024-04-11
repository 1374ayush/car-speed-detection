import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

#  

model=YOLO('yolov8s.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker=Tracker()
count=0

cap=cv2.VideoCapture('highway_mini.mp4')


while True:    
    ret,frame = cap.read()
    ''' ret: This variable indicates whether the frame was successfully read or not. It is a boolean value, where True indicates that a frame was successfully read, and False indicates the end of the video stream or an error occurred while reading the frame.

    frame: This variable contains the actual frame data that is read from the video. It is typically represented as a NumPy array, where each element in the array corresponds to a pixel value in the image.
    '''
    if not ret:
        break
    count += 1
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
    #print(results)
    a=results[0].boxes.data
    a = a.detach().cpu().numpy() 
    '''  code converts the bounding box data from a tensor to a NumPy array. It first detaches the tensor from the computation graph (detach()), then moves it from GPU memory to CPU memory (cpu()), and finally converts it to a NumPy array (numpy()). '''
    px=pd.DataFrame(a).astype("float")
    '''
    This line creates a pandas DataFrame (px) from the bounding box data (a). Each row in the DataFrame represents a bounding box, and each column represents a coordinate or attribute of the bounding box. The astype("float") ensures that all values in the DataFrame are treated as floating-point numbers.

    x1, y1, x2, y2, confidenceRatio, classid
    ex : 0   268.701050  174.515259  305.326935  199.400818  0.843476  2.0 

    pandas libraray is used for data manipulation and analysis,in our case we pass ND-array or dictonary and it convert it in a single 2d array so called dataframes.
    '''
    #print("px: ", px)

    list=[]
    # px.iterrows this will consist of the number of rows.
    for index,row in px.iterrows():
#        print(row) 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        # here we are only focused over car , we can extend it for multiple vehicles
        if 'car' in c:
            list.append([x1,y1,x2,y2])
            #print(c)

    bbox_id=tracker.update(list)
    #print(bbox_id)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        red_line_y=198
        blue_line_y=268   
        offset = 7

        #place the dot at center and a rectangle block at the particular object of the frame over coordinates.
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                 
    
    #colors to be used
    text_color = (255,255,255)  
    red_color = (0, 0, 255)  
    blue_color = (255, 0, 0) 
    green_color = (0, 255, 0) 

    cv2.line(frame,(172,198),(774,198),red_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('red line'),(172,198),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    cv2.line(frame,(8,268),(927,268),blue_color,3)  # seconde line
    cv2.putText(frame,('blue line'),(8,268),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)    


    cv2.imshow("frames", frame)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()