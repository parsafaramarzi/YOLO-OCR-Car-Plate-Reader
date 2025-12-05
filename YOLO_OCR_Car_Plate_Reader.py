import imageio
import cv2
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import re

reader = easyocr.Reader(['en'] , gpu= True, model_storage_directory= 'import')
model = YOLO('yolo11n.pt', task = 'detect')
cap = cv2.VideoCapture('cartraffic01.mp4')
writer = imageio.get_writer("output/YOLO_OCR_Car_Plate.mp4", fps=30, codec='libx264', quality=8)

aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
new_height = 900
new_width = int(new_height * aspect_ratio)

plate_numbers = []
number_of_occurances = {}

def is_valid_plate(concat_number):
    normalized_plate = concat_number.replace(' ', '').upper()
    pattern_1 = re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{1}\d{2,5}$')
    pattern_2 = re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{2}\d{2,4}$')
    pattern_3 = re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{3}\d{2,3}$')
    if pattern_1.match(normalized_plate):
        return True
    if pattern_2.match(normalized_plate):
        return True
    if pattern_3.match(normalized_plate):
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    results = model.predict(source = frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 2 or class_id == 5 or class_id == 7 or class_id == 3:
                coordinates = box.xyxy[0]                        
                x1, y1, x2, y2 = map(int, coordinates.tolist())
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2)
                plate_x1 = int(x1 + (x2 - x1)*15/100)
                plate_y1 = int(y1 + (y2 - y1)*40/100)
                plate_x2 = int(x2 - (x2 - x1)*15/100)
                plate_y2 = int(y2 - (y2 - y1)*10/100)
                cv2.rectangle(frame,(plate_x1,plate_y1),(plate_x2,plate_y2),(255,0,0), 2)
                plate_region = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                plate_image = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_RGB2BGR))
                plate_array = np.array(plate_image)
                plate_number = reader.readtext(plate_array)
                concat_number = ' '.join([number[1] for number in plate_number])
                if is_valid_plate(concat_number):
                    print(concat_number)
                    cv2.putText(
                        img = frame,
                        text = concat_number,
                        org = (plate_x1, plate_y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 0, 255),
                        thickness=2
                    )
                    if concat_number != '':
                        plate_numbers.append(concat_number)
                        if concat_number in number_of_occurances:
                            number_of_occurances[concat_number] += 1
                        else:
                            number_of_occurances[concat_number] = 1

    frame = cv2.resize(frame, (new_width, new_height))
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow('Detections', frame)
    if cv2.waitKey(1) == 13 :
        cap.release()
        writer.close()
        cv2.destroyAllWindows()
        break

print("Plate Numbers Detected So Far:", plate_numbers)
print("Number of Occurrences:", number_of_occurances)

cap.release()
writer.close()
cv2.destroyAllWindows()