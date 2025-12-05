import imageio
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import csv

ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
model = YOLO('license-plate-finetune-v1x.pt')
cap = cv2.VideoCapture('cartraffichd01.mp4')
writer = imageio.get_writer("output/YOLO_OCR_Car_Plate.mp4", fps=30, codec='libx264', quality=8)
print(model.names)
aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
new_height = 900
new_width = int(new_height * aspect_ratio)
demo_height = 720
demo_width = int(demo_height * aspect_ratio)

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

def write_plate_records_csv(number_of_occurances):
    output_filename = 'output/plate_records.csv'
    header = ['Plate_Number', 'Count_of_Occurrences']
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for plate, count in number_of_occurances.items():
            writer.writerow([plate, count])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    results = model.predict(source = frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 0:
                coordinates = box.xyxy[0]                        
                x1, y1, x2, y2 = map(int, coordinates.tolist())
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2)
                plate_region = frame[y1:y2, x1:x2]
                result = ocr_model.predict(plate_region)
                if result and isinstance(result, list) and "rec_texts" in result[0]:
                    rec_texts = result[0]["rec_texts"]
                    plate_text = " ".join(rec_texts).strip()
                if is_valid_plate(plate_text):
                    print(plate_text)
                    cv2.putText(
                        img = frame,
                        text = plate_text,
                        org = (x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=(0, 0, 255),
                        thickness=2
                    )
                    if plate_text != '':
                        if plate_text in number_of_occurances:
                            number_of_occurances[plate_text] += 1
                        else:
                            number_of_occurances[plate_text] = 1

    demo_frame = cv2.resize(frame, (demo_width, demo_height))
    writer.append_data(cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB))
    frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow('Detections', frame)
    if cv2.waitKey(1) == 13 :
        cap.release()
        writer.close()
        cv2.destroyAllWindows()
        break

print("Number of Occurrences:", number_of_occurances)
write_plate_records_csv(number_of_occurances)

cap.release()
writer.close()
cv2.destroyAllWindows()