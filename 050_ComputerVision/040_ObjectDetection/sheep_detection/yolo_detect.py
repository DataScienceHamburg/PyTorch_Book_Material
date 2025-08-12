#%% packages
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

#%% load model
model = YOLO('yolov8n.pt')  # load the smallest YOLOv8 model

#%% predict
results = model('kiki.jpg')  # predict on an image

#%% visualize results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    
    # Print detected objects
    print("\nDetected objects:")
    for box in r.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"- {class_name}: {confidence:.2f}") 
# %%
