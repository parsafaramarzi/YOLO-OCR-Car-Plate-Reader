import csv
import re
from collections import defaultdict, Counter
import cv2
import imageio
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from paddleocr import PaddleOCR
from scipy.spatial.distance import cdist

VIDEO_PATH = 'cartraffic.mp4'
SPECIALIZED_MODEL_PATH = 'license-plate-finetune-v1n.pt'
GENERAL_MODEL_PATH = 'yolo11n.pt'
OUTPUT_VIDEO_PATH = 'output/yolov_vehicle_plate_passthrough_voted.mp4'
OUTPUT_CSV_PATH = 'output/plate_records_pass_voted.csv'
TARGET_WIDTH = 1280
FPS = 30

VEHICLE_CLASS_IDS = [2, 5, 7, 6]

ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
model = YOLO(SPECIALIZED_MODEL_PATH)
general_model = YOLO(GENERAL_MODEL_PATH)

track_history = defaultdict(lambda: [])
plate_passcount_dict = {}
plate_position_dict = {}
track_all_plates_dict = defaultdict(list)
car_plate_map = {} 

videocapture = cv2.VideoCapture(VIDEO_PATH)
success, frame = videocapture.read()

if success:
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = int(TARGET_WIDTH / aspect_ratio)
    TARGET_SIZE = (TARGET_WIDTH, new_height)

videocapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
writer = imageio.get_writer(OUTPUT_VIDEO_PATH, fps=FPS, codec='libx264', quality=8)

def is_valid_plate(normalized_plate):
    patterns = [
        re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{1}\d{2,5}$'),
        re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{2}\d{2,4}$'),
        re.compile(r'^(0[1-9]|[1-7]\d|8[01])[A-Z]{3}\d{2,3}$')
    ]
    return any(pattern.match(normalized_plate) for pattern in patterns)

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

def get_status_and_bgr_color(is_valid_reconstructed, is_in_target_zone):
    if is_valid_reconstructed:
        status = "Valid"
        if is_in_target_zone:
            color = (0, 255, 0)
        else:
            color = (255, 100, 0)
    else:
        status = "Invalid"
        if is_in_target_zone:
            color = (0, 100, 255)
        else:
            color = (0, 0, 255)
    return status, color

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

    plate_results = model.track(frame, persist=True, verbose=False)
    general_results = general_model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASS_IDS)

    annotator = Annotator(frame, line_width=4)
    annotator.box_label(target_zone, label='Target Zone', color=(0, 0, 0))

    vehicle_detections = []
    if general_results[0].boxes.id is not None:
        vehicle_boxes = general_results[0].boxes.xyxy.tolist()
        vehicle_track_ids = general_results[0].boxes.id.int().tolist()
        for vh_box, vh_id in zip(vehicle_boxes, vehicle_track_ids):
            vehicle_detections.append({
                'id': vh_id,
                'box': vh_box,
                'center': ((vh_box[0] + vh_box[2]) / 2, (vh_box[1] + vh_box[3]) / 2)
            })
    
    plate_detections = []
    if plate_results[0].boxes.id is not None:
        plate_boxes = plate_results[0].boxes.xyxy.tolist()
        plate_track_ids = plate_results[0].boxes.id.int().tolist()
        for pbox, pid in zip(plate_boxes, plate_track_ids):
            plate_detections.append({
                'id': pid,
                'box': pbox,
                'center': ((pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2)
            })

    plate_to_vehicle_map = {}
    
    if vehicle_detections and plate_detections:
        vehicle_centers = [d['center'] for d in vehicle_detections]
        plate_centers = [d['center'] for d in plate_detections]

        distances = cdist(plate_centers, vehicle_centers)
        
        min_distance_indices = distances.argmin(axis=1)

        for plate_idx, vehicle_idx in enumerate(min_distance_indices):
            plate_track_id = plate_detections[plate_idx]['id']
            vehicle_track_id = vehicle_detections[vehicle_idx]['id']
            plate_to_vehicle_map[plate_track_id] = vehicle_track_id

    for plate_det in plate_detections:
        pbox = plate_det['box']
        plate_track_id = plate_det['id']
        
        associated_vehicle_id = plate_to_vehicle_map.get(plate_track_id)
        
        if associated_vehicle_id is None:
             continue
        
        _, plate_text_normalized, is_valid_ocr = get_processed_plate_info(frame, pbox, ocr_model)
        
        if is_valid_ocr:
            track_all_plates_dict[associated_vehicle_id].append(plate_text_normalized)

    counted_current_ids = 0
    vehicle_label_info = defaultdict(lambda: {"plate": "N/A", "status": "N/A", "color": (150, 150, 150)})
    
    for vehicle_det in vehicle_detections:
        vehicle_track_id = vehicle_det['id']
        vehicle_box = vehicle_det['box']
        midpoint_x = vehicle_det['center'][0]
        midpoint_y = vehicle_det['center'][1]

        reconstructed_plate_text = "N/A"
        is_valid_reconstructed = False
        
        reconstructed_plate_text, is_valid_reconstructed = reconstruct_best_plate(
            track_all_plates_dict[vehicle_track_id]
        )

        prev_state = plate_position_dict.get(vehicle_track_id, "out")
        is_in_target_zone = (midpoint_x > target_box_x1 and midpoint_x < target_box_x2 and
                             midpoint_y > target_box_y1 and midpoint_y < target_box_y2)

        status, color = get_status_and_bgr_color(is_valid_reconstructed, is_in_target_zone)
        
        if is_in_target_zone:
            counted_current_ids += 1
            current_state = "in"
            
            if prev_state == "out":
                plate_passcount_dict[vehicle_track_id] = plate_passcount_dict.get(vehicle_track_id, 0) + 1
        else:
            current_state = "out"
        
        plate_position_dict[vehicle_track_id] = current_state
        
        vehicle_label_info[vehicle_track_id] = {
            "plate": reconstructed_plate_text, 
            "status": status, 
            "color": color
        }
        
        info = vehicle_label_info[vehicle_track_id]
        label_text = f'ID: {vehicle_track_id} Plate: {info["plate"]} ({info["status"]})'
        annotator.box_label(vehicle_box, label=label_text, color=info["color"])

    count_text = f'Vehicles in Target Zone: {counted_current_ids}'
    maxwidth = len(count_text) * 10

    cv2.rectangle(frame, (0, 0), (390 + maxwidth, 75), (50, 50, 50), -1)
    cv2.putText(frame, count_text, org=(30, 50), color=(100, 255, 100), fontScale=1.4, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2)

    frame = cv2.resize(frame, TARGET_SIZE)

    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow("Input", frame)

    if cv2.waitKey(1) == 13:
        break

final_count_dict = defaultdict(int)

for vehicle_track_id, count in plate_passcount_dict.items():
    best_plate, is_valid = reconstruct_best_plate(track_all_plates_dict[vehicle_track_id])

    if is_valid:
        final_key = best_plate
    else:
        final_key = f'UNKNOWN_VEHICLE_TRACK_{vehicle_track_id}' 

    final_count_dict[final_key] += count

final_plate_counts = dict(final_count_dict) 

print(f"Final Aggregated (Voted) Plate Pass Counts: {final_plate_counts}")
write_id_records_csv(final_plate_counts, OUTPUT_CSV_PATH)

videocapture.release()
writer.close()
cv2.destroyAllWindows()