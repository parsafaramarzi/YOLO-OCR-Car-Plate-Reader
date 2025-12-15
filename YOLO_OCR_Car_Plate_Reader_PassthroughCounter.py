import csv
import re
from collections import defaultdict, Counter
import cv2
import imageio
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from paddleocr import PaddleOCR

VIDEO_PATH = 'cartraffic.mp4'
MODEL_PATH = 'license-plate-finetune-v1x.pt'
OUTPUT_VIDEO_PATH = 'output/yolov_plate_passthrough_counter_voted.mp4'
OUTPUT_CSV_PATH = 'output/plate_records_pass_voted.csv'
TARGET_WIDTH = 1280
FPS = 30

def is_valid_plate(normalized_plate):
    patterns = [
        re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{1}\d{2,5}$'),
        re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{2}\d{2,4}$'),
        re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{3}\d{2,3}$')
    ]
    return any(pattern.match(normalized_plate) for pattern in patterns)

ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
model = YOLO(MODEL_PATH)

track_history = defaultdict(lambda: [])
plate_passcount_dict = {}         
plate_position_dict = {}          
track_all_plates_dict = defaultdict(list)

videocapture = cv2.VideoCapture(VIDEO_PATH)
success, frame = videocapture.read()

if success:
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = int(TARGET_WIDTH / aspect_ratio)
    TARGET_SIZE = (TARGET_WIDTH, new_height)

videocapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
writer = imageio.get_writer(OUTPUT_VIDEO_PATH, fps=FPS, codec='libx264', quality=8)

def get_processed_plate_info(frame, box, ocr_model):
    def normalize_plate(plate_text):
        return plate_text.replace(' ', '').upper()

    x1, y1, x2, y2 = map(int, box) 
    plate_region = frame[y1:y2, x1:x2]
    result = ocr_model.predict(plate_region)
    
    plate_text_raw = ""
    if result and isinstance(result, list) and len(result) > 0:
        first_result = result[0]
        if isinstance(first_result, dict) and "rec_texts" in first_result:
            recognized_texts = first_result["rec_texts"]
            if recognized_texts and isinstance(recognized_texts, list):
                plate_text_raw = " ".join(recognized_texts).strip()
    
    normalized_plate = normalize_plate(plate_text_raw)
    
    is_valid = is_valid_plate(normalized_plate)
    
    return plate_text_raw, normalized_plate, is_valid

def reconstruct_best_plate(plate_list):
    if not plate_list:
        return "UNKNOWN", False
    
    length_counts = Counter(len(plate) for plate in plate_list)
    most_common_length = length_counts.most_common(1)[0][0]
    
    common_length_plates = [p for p in plate_list if len(p) == most_common_length]
    
    if not common_length_plates:
        return "UNKNOWN", False

    reconstructed_plate = []
    
    for i in range(most_common_length):
        position_chars = [p[i] for p in common_length_plates if len(p) > i]
        
        if position_chars:
            most_common_char = Counter(position_chars).most_common(1)[0][0]
            reconstructed_plate.append(most_common_char)
        else:
            reconstructed_plate.append('?')
            
    best_plate_string = "".join(reconstructed_plate)
    
    is_reconstructed_valid = is_valid_plate(best_plate_string)
    
    return best_plate_string, is_reconstructed_valid

def aggregate_counts_by_plate(plate_passcount_dict, track_all_plates_dict):
    final_count_dict = defaultdict(int)
    
    for track_id, count in plate_passcount_dict.items():
        best_plate, is_valid = reconstruct_best_plate(track_all_plates_dict[track_id])
        
        if is_valid:
            final_key = best_plate
        else:
            final_key = f'UNKNOWN_TRACK_{track_id}'

        final_count_dict[final_key] += count
        
    return dict(final_count_dict)

def write_id_records_csv(plate_count_dict, filename):
    header = ['Plate_Text', 'Count_of_Passes']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for plate, count in plate_count_dict.items():
            writer.writerow([plate, count])

while videocapture.isOpened():
    success, frame = videocapture.read()
    
    if not success:
        break
        
    H, W = frame.shape[:2]
    target_box_y1 = int(H / 2)
    target_box_y2 = int(H / 2 + H / 10)
    target_box_x1 = int(W / 2 - W / 5 * 2)
    target_box_x2 = int(W / 2 + W / 5 * 2)
    target_zone = (target_box_x1, target_box_y1, target_box_x2, target_box_y2)
    
    results = model.track(frame, persist=True, verbose=False)
    
    annotator = Annotator(frame, line_width=4)
    annotator.box_label(target_zone, label='Target Zone', color=(255, 0, 0))
    
    if results[0].boxes.id is None:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow("Input", frame)
        if cv2.waitKey(1) == 13:
            break
        continue
        
    boxes = results[0].boxes.xyxy.tolist()
    track_ids = results[0].boxes.id.int().tolist()
    
    counted_current_ids = 0

    for i, box in enumerate(boxes):
        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box) 
        track_id = track_ids[i]
        
        midpoint_x = x1 + (x2 - x1) / 2
        midpoint_y = y1 + (y2 - y1) / 2
        
        track_history[track_id].append((midpoint_x, midpoint_y))
        
        plate_text_raw, plate_text_normalized, is_valid_ocr = get_processed_plate_info(frame, box, ocr_model)
        
        reconstructed_plate_text, is_valid_reconstructed = reconstruct_best_plate(track_all_plates_dict[track_id])

        prev_state = plate_position_dict.get(track_id, "out")
        current_state = prev_state
        
        is_in_target_zone = (midpoint_x > target_box_x1 and midpoint_x < target_box_x2 and
                             midpoint_y > target_box_y1 and midpoint_y < target_box_y2)

        if is_valid_ocr:
            track_all_plates_dict[track_id].append(plate_text_normalized)
        
        if is_valid_reconstructed:
            status = "Valid"
            if is_in_target_zone:
                color = (0, 255, 0)       
            else:
                color = (0, 100, 255)     
        else:
            status = "Invalid"
            if is_in_target_zone:
                color = (255, 165, 0)     
            else:
                color = (255, 0, 0)

        if is_in_target_zone:
            counted_current_ids += 1
            current_state = "in"
            
            if prev_state == "out":
                plate_passcount_dict[track_id] = plate_passcount_dict.get(track_id, 0) + 1
            
            label_plate_text = reconstructed_plate_text if reconstructed_plate_text else "N/A"   
        else: 
            current_state = "out"
            
            label_plate_text = reconstructed_plate_text
        
        plate_position_dict[track_id] = current_state

        label_text = f'ID: {track_id} Plate: {label_plate_text} ({status})'
        annotator.box_label(box, label=label_text, color=color)

    count_text = f'Vehicles in Target Zone: {counted_current_ids}'
    maxwidth = len(count_text) * 10
    
    cv2.rectangle(frame, (0, 0), (390 + maxwidth, 75), (50, 50, 50), -1)
    cv2.putText(frame, count_text, org=(30, 50), color=(100, 255, 100), fontScale=1.4, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)

    frame = cv2.resize(frame, TARGET_SIZE)
    
    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow("Input", frame)
    
    if cv2.waitKey(1) == 13:
        break

final_plate_counts = aggregate_counts_by_plate(plate_passcount_dict, track_all_plates_dict)

print(f"Final Aggregated (Voted) Plate Pass Counts: {final_plate_counts}")
write_id_records_csv(final_plate_counts, OUTPUT_CSV_PATH)

videocapture.release()
writer.close()
cv2.destroyAllWindows()