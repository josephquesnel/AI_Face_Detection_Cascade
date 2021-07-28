import cv2, time

jo_cascade = cv2.CascadeClassifier('AI_Face_Detection_Cascade/Cascade/cascade.xml')

webcam = cv2.VideoCapture(0)

while True:
    
    successful_frame_read, frame = webcam.read()
    greyimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facecoord = jo_cascade.detectMultiScale(greyimg)
    
    for (x,y,w,h) in facecoord:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255))
    
    cv2.imshow('face detector', frame)
    
    key = cv2.waitKey(1)
    if key in (ord('q'), ord('Q')):
        break
    
    # Saves images from loop source (webcam in this case), used to get neg and pos images
    if key in (ord('s'), ord('S')): 
       cv2.imwrite(f"AI_Face_Detection_Cascade/NewPos/{time.time()}.jpg", frame)
    
webcam.release()