import cv2
import numpy as np
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import threading
import json
import os
import sys
import sqlite3
import webbrowser
from datetime import datetime, timedelta
import time
import easyocr

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class ParkingStatus:
    VACANT = "Vacant"
    OCCUPIED_VALID = "Occupied (Valid)"
    OCCUPIED_UNREADABLE = "Occupied (Plate Unreadable)"
    VIOLATION_WRONG_PLATE = "Violation: Wrong Plate"
    VIOLATION_NO_PLATE = "Violation: No Plate"
    RESERVED = "Reserved"
    OCCUPIED_RESERVED = "Occupied (Reserved)"

# Global variables for model
net = None
CONFIDENCE_THRESHOLD = 0.2
reader = None

def init_ocr():
    """Initialize the OCR reader."""
    global reader
    if reader is None:
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['en'])
    return reader

def detect_license_plate(frame, box):
    """Detect license plate using EasyOCR."""
    global reader
    try:
        # Initialize OCR if needed
        if reader is None:
            reader = init_ocr()
        
        # Extract the region of interest (ROI)
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply some preprocessing
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.medianBlur(gray, 3)
        
        # Detect text
        results = reader.readtext(gray)
        
        # Process results
        if results:
            # Get the text with highest confidence
            text = max(results, key=lambda x: x[2])[1]
            
            # Clean up the text (remove spaces and special characters)
            text = ''.join(c for c in text if c.isalnum())
            
            # Validate the plate format (you can adjust this based on your needs)
            if len(text) >= 4:  # Most plates have at least 4 characters
                return text.upper()
        
        return None
    except Exception as e:
        print(f"Error in license plate detection: {e}")
        return None

def detect_cars(frame):
    """Simple detection to check if a hotwheels car is present in the frame."""
    global net
    
    if net is None:
        try:
            # Load pre-trained Caffe model
            model_dir = get_resource_path("models")
            os.makedirs(model_dir, exist_ok=True)
            
            prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
            model = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
            
            if not os.path.exists(prototxt) or not os.path.exists(model):
                print("Downloading model files...")
                import urllib.request
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
                    prototxt
                )
                urllib.request.urlretrieve(
                    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
                    model
                )
            
            net = cv2.dnn.readNetFromCaffe(prototxt, model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return []

    # Process frame
    height, width = frame.shape[:2]
    
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    
    # Detect objects
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections
    detected_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            # Get coordinates
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            x1, y1, x2, y2 = box.astype(int)
            
            # Add to detected boxes if size is reasonable
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                detected_boxes.append((x1, y1, x2, y2, confidence))
    
    return detected_boxes

def is_inside_box(x, y, box):
    """Check if a point (x,y) is inside the given box."""
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_boxes(event, x, y, flags, param):
    global drawing_mode, current_box, start_point, boxes

    if drawing_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start_point:
            current_box = [start_point[0], start_point[1], x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            if start_point:
                x1, y1 = start_point
                x2, y2 = x, y
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                start_point = None
                current_box = []

def load_boxes():
    """Load parking spot boxes from file."""
    global boxes
    if os.path.exists(BOXES_FILE):
        with open(BOXES_FILE, "r") as f:
            boxes = json.load(f)
            print(f"Loaded {len(boxes)} boxes from {BOXES_FILE}")

def save_boxes():
    """Save parking spot boxes to file."""
    with open(BOXES_FILE, "w") as f:
        json.dump(boxes, f)
        print(f"Saved {len(boxes)} boxes to {BOXES_FILE}")

def setup_database():
    """Initialize the SQLite database with booking support."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    
    # Register adapter for datetime objects
    sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
    sqlite3.register_converter('datetime', lambda dt: datetime.fromisoformat(dt.decode()))
    
    # Parking records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spot_id TEXT,
            status TEXT,
            timestamp datetime,
            license_plate TEXT,
            rfid_tag TEXT,
            entry_time datetime,
            exit_time datetime,
            tariff FLOAT
        )
    """)
    
    # RFID mapping table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rfid_mapping (
            rfid_tag TEXT PRIMARY KEY,
            license_plate TEXT,
            owner_name TEXT,
            registration_date datetime
        )
    """)
    
    # Booking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spot_id TEXT,
            license_plate TEXT,
            start_time datetime,
            end_time datetime,
            status TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def is_spot_reserved(spot_id):
    """Check if a parking spot is currently reserved."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    current_time = datetime.now()
    
    cursor.execute("""
        SELECT license_plate FROM bookings 
        WHERE spot_id = ? 
        AND start_time <= ? 
        AND end_time >= ? 
        AND status = 'active'
    """, (spot_id, current_time, current_time))
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

def log_violation(spot_id, detected_plate, expected_plate):
    """Log a parking violation."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    timestamp = datetime.now()
    
    cursor.execute("""
        INSERT INTO parking_records (spot_id, status, timestamp, license_plate)
        VALUES (?, ?, ?, ?)
    """, (spot_id, "VIOLATION", timestamp, detected_plate))
    
    violation = {
        "spot_id": spot_id,
        "detected_plate": detected_plate,
        "expected_plate": expected_plate,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if violation not in parking_status["violations"]:
        parking_status["violations"].append(violation)
    
    conn.commit()
    conn.close()

def update_parking_status():
    """Update parking status for each defined spot."""
    global parking_status, drawing_mode
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Flip the frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)  # 1 means horizontal flip
            cars = detect_cars(frame)

            vacant_count = 0
            occupied_count = 0
            spots = []
            violations = []

            for idx, box in enumerate(boxes):
                spot_id = f"p{idx+1}"
                occupied = False
                license_plate = None
                max_confidence = 0
                
                # First check for car and license plate
                for (x1, y1, x2, y2, confidence) in cars:
                    car_center_x = (x1 + x2) // 2
                    car_center_y = (y1 + y2) // 2
                    if is_inside_box(car_center_x, car_center_y, box):
                        occupied = True
                        max_confidence = confidence * 100
                        license_plate = detect_license_plate(frame, box)
                        break
                
                # Then check database for bookings
                conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
                cursor = conn.cursor()
                current_time = datetime.now()
                
                # Check current bookings for this spot
                cursor.execute("""
                    SELECT license_plate, end_time 
                    FROM bookings 
                    WHERE spot_id = ? 
                    AND start_time <= ? 
                    AND end_time >= ? 
                    AND status = 'active'
                """, (spot_id, current_time, current_time))
                
                booking = cursor.fetchone()

                # If there's a car with a license plate, check if it's booked elsewhere
                if occupied and license_plate:
                    cursor.execute("""
                        SELECT spot_id, license_plate
                        FROM bookings 
                        WHERE license_plate = ? 
                        AND spot_id != ?
                        AND start_time <= ? 
                        AND end_time >= ? 
                        AND status = 'active'
                    """, (license_plate, spot_id, current_time, current_time))
                    wrong_spot_booking = cursor.fetchone()
                else:
                    wrong_spot_booking = None
                
                conn.close()

                if booking:
                    booked_plate, end_time = booking
                    if occupied:
                        if license_plate == booked_plate:
                            status = ParkingStatus.OCCUPIED_RESERVED
                            occupied_count += 1
                        else:
                            status = ParkingStatus.VIOLATION_WRONG_PLATE
                            violation = {
                                "spot_id": spot_id,
                                "detected_plate": license_plate,
                                "expected_plate": booked_plate,
                                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            violations.append(violation)
                            log_violation(spot_id, license_plate, booked_plate)
                            occupied_count += 1
                    else:
                        status = ParkingStatus.RESERVED
                        occupied_count += 1
                elif wrong_spot_booking and occupied:
                    # Vehicle is parked in wrong spot
                    booked_spot, plate = wrong_spot_booking
                    status = ParkingStatus.VIOLATION_WRONG_PLATE
                    violation = {
                        "spot_id": spot_id,
                        "detected_plate": license_plate,
                        "expected_spot": booked_spot,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "message": f"Vehicle booked for {booked_spot} but parked in {spot_id}"
                    }
                    violations.append(violation)
                    log_violation(spot_id, license_plate, f"Should be in {booked_spot}")
                    occupied_count += 1
                else:
                    if occupied:
                        if license_plate:
                            status = ParkingStatus.OCCUPIED_VALID
                        else:
                            status = ParkingStatus.OCCUPIED_UNREADABLE
                        occupied_count += 1
                    else:
                        status = ParkingStatus.VACANT
                        vacant_count += 1

                spot_info = {
                    "id": spot_id,
                    "status": status,
                    "license_plate": license_plate if occupied else None
                }
                
                if booking:
                    spot_info["reserved_for"] = booked_plate
                    spot_info["booking_time"] = end_time
                
                spots.append(spot_info)

                # Visualization
                color = (0, 255, 0)  # Green for vacant
                if status == ParkingStatus.OCCUPIED_VALID or status == ParkingStatus.OCCUPIED_RESERVED:
                    color = (0, 255, 255)  # Yellow for valid occupation
                elif status == ParkingStatus.VIOLATION_WRONG_PLATE:
                    color = (0, 0, 255)  # Red for violation
                elif status == ParkingStatus.RESERVED:
                    color = (255, 191, 0)  # Orange for reserved
                elif status == ParkingStatus.OCCUPIED_UNREADABLE:
                    color = (128, 0, 128)  # Purple for unreadable
                
                # Draw rectangle for parking spot
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Display confidence only if a car is detected in the box
                if max_confidence > 0:
                    confidence_text = f"{max_confidence:.2f}%"
                    cv2.putText(frame, confidence_text, (box[0], box[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display license plate and status
                if license_plate:
                    cv2.putText(frame, f"Plate: {license_plate}", (box[0], box[1]-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display status
                status_y = box[3] + 20
                cv2.putText(frame, f"Status: {status}", (box[0], status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display reservation if exists
                if booking:
                    cv2.putText(frame, f"Reserved: {booked_plate}", (box[0], status_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update parking status
            parking_status["spots"] = spots
            parking_status["vacant_count"] = vacant_count
            parking_status["occupied_count"] = occupied_count
            parking_status["violations"] = violations

            try:
                # Show the frame
                cv2.imshow("Smart Parking System", frame)
            except cv2.error as e:
                print(f"Warning: Could not display frame: {e}")
                pass

            # Handle key events
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    sys.exit(0)
                elif key == ord('d'):
                    if drawing_mode:
                        save_boxes()
                    drawing_mode = not drawing_mode
                elif key == ord('c'):
                    boxes.clear()
                    save_boxes()
                    print("All boxes cleared!")
            except cv2.error:
                time.sleep(0.1)
            
            return parking_status

        except Exception as e:
            print(f"Error in frame processing: {e}")
            time.sleep(1)
            continue

def handle_rfid_scan():
    """Handle RFID scans from Arduino."""
    global arduino, parking_status
    
    while True:
        if arduino:
            try:
                if arduino.in_waiting:
                    data = arduino.readline().decode('utf-8').strip()
                    if data.startswith("RFID:"):
                        rfid_tag = data[5:]  # Remove "RFID:" prefix
                        process_rfid_entry_exit(rfid_tag)
            except Exception as e:
                print(f"Arduino communication error: {e}")
        time.sleep(0.1)

def process_rfid_entry_exit(rfid_tag):
    """Process vehicle entry/exit based on RFID scan."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    
    # Get license plate from RFID mapping
    cursor.execute("SELECT license_plate FROM rfid_mapping WHERE rfid_tag = ?", (rfid_tag,))
    result = cursor.fetchone()
    
    if not result:
        print(f"Unknown RFID tag: {rfid_tag}")
        conn.close()
        return
        
    license_plate = result[0]
    current_time = datetime.now()
    
    # Check if vehicle is entering or exiting
    cursor.execute("""
        SELECT id, entry_time FROM parking_records 
        WHERE rfid_tag = ? AND exit_time IS NULL
        ORDER BY entry_time DESC LIMIT 1
    """, (rfid_tag,))
    
    active_session = cursor.fetchone()
    
    if active_session:
        # Vehicle is exiting
        record_id, entry_time = active_session
        entry_time = datetime.fromisoformat(entry_time)
        duration = (current_time - entry_time).total_seconds() / 3600.0  # hours
        tariff = calculate_tariff(duration)
        
        cursor.execute("""
            UPDATE parking_records 
            SET exit_time = ?, tariff = ?
            WHERE id = ?
        """, (current_time, tariff, record_id))
        
        print(f"Vehicle exit - License: {license_plate}, Duration: {duration:.2f}h, Tariff: ${tariff:.2f}")
    else:
        # Vehicle is entering
        cursor.execute("""
            INSERT INTO parking_records (rfid_tag, license_plate, entry_time, status)
            VALUES (?, ?, ?, 'ACTIVE')
        """, (rfid_tag, license_plate, current_time))
        
        print(f"Vehicle entry - License: {license_plate}")
    
    conn.commit()
    conn.close()

def calculate_tariff(duration_hours):
    """Calculate parking tariff based on duration."""
    base_rate = 2.00  # $2 per hour
    return base_rate * duration_hours

# API endpoints
@app.route('/parking_status', methods=['GET'])
def get_parking_status():
    return jsonify(parking_status)

@app.route('/book_spot', methods=['POST'])
def book_spot():
    data = request.json
    spot_id = data['spot_id']
    license_plate = data['license_plate']
    start_time = datetime.fromisoformat(data['start_time'])
    end_time = datetime.fromisoformat(data['end_time'])
    
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    
    # Check if spot is available for the requested time
    cursor.execute("""
        SELECT COUNT(*) FROM bookings 
        WHERE spot_id = ? 
        AND status = 'active'
        AND (
            (start_time <= ? AND end_time >= ?) OR
            (start_time <= ? AND end_time >= ?) OR
            (start_time >= ? AND end_time <= ?)
        )
    """, (spot_id, start_time, start_time, end_time, end_time, start_time, end_time))
    
    if cursor.fetchone()[0] > 0:
        conn.close()
        return jsonify({"error": "Spot not available for the selected time period"}), 400
    
    # Create booking
    cursor.execute("""
        INSERT INTO bookings (spot_id, license_plate, start_time, end_time, status)
        VALUES (?, ?, ?, ?, 'active')
    """, (spot_id, license_plate, start_time, end_time))
    
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Booking successful"}), 200

@app.route('/parking_history', methods=['GET'])
def get_parking_history():
    """API endpoint to get parking history."""
    license_plate = request.args.get('license_plate')
    
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    
    if license_plate:
        cursor.execute("""
            SELECT entry_time, exit_time, tariff 
            FROM parking_records 
            WHERE license_plate = ?
            ORDER BY entry_time DESC
        """, (license_plate,))
    else:
        cursor.execute("""
            SELECT license_plate, entry_time, exit_time, tariff 
            FROM parking_records 
            ORDER BY entry_time DESC
        """)
    
    records = cursor.fetchall()
    conn.close()
    
    return jsonify({"records": records})

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# Arduino setup
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
except:
    print("Warning: Arduino not connected. RFID functionality will be disabled.")
    arduino = None

# Helper function to handle PyInstaller file paths
def get_resource_path(relative_path):
    """Get the absolute path to a resource, compatible with PyInstaller."""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# File paths
BOXES_FILE = get_resource_path("boxes.json")
DB_FILE = get_resource_path("parking.db")

# Initialize camera with better error handling and debugging
def init_camera():
    """Initialize the camera with multiple retries and debugging."""
    if not hasattr(cv2, 'VideoCapture'):
        raise ImportError("OpenCV VideoCapture not available. Please reinstall opencv-python")
        
    for camera_index in range(3):  # Try first 3 indices
        print(f"Attempting to open camera {camera_index}...")
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Try DirectShow backend
            if cap.isOpened():
                # Try to read a test frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully opened camera {camera_index}")
                    print(f"Frame size: {frame.shape}")
                    return cap
                else:
                    print(f"Camera {camera_index} opened but couldn't read frame")
                    cap.release()
            else:
                print(f"Failed to open camera {camera_index}")
        except Exception as e:
            print(f"Error trying camera {camera_index}: {str(e)}")
    
    raise Exception("No working camera found")

# Create test image for development mode
def create_test_image():
    """Create a test image for when no camera is available."""
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some test content
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), 2)  # Green rectangle
    cv2.rectangle(test_image, (350, 100), (550, 300), (0, 0, 255), 2)  # Red rectangle
    # Add text if putText is available
    if hasattr(cv2, 'putText'):
        cv2.putText(test_image, "Test Mode - No Camera", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return test_image

try:
    print("Initializing camera...")
    cap = init_camera()
except Exception as e:
    print(f"Error initializing camera: {e}")
    print("Running in test mode with static image...")
    test_image = create_test_image()
    
    class TestCapture:
        def read(self):
            return True, test_image.copy()
        def isOpened(self):
            return True
        def release(self):
            pass
    
    cap = TestCapture()

# Shared parking status data
parking_status = {
    "spots": [],
    "vacant_count": 0,
    "occupied_count": 0,
    "violations": []
}

# User-drawn boxes
boxes = []
drawing_mode = False
current_box = []
start_point = None

# Initialize window if GUI is available
try:
    cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Smart Parking System", draw_boxes)
except cv2.error:
    print("Warning: GUI functionality not available. Running in headless mode.")
    # In headless mode, we'll need to pre-define parking spots
    # You can add default parking spot coordinates here if needed
    boxes = [
        [100, 100, 300, 300],  # Example parking spot 1
        [350, 100, 550, 300]   # Example parking spot 2
    ]

# Load existing boxes
load_boxes()

# Main Execution
if __name__ == "__main__":
    setup_database()
    
    server_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False))
    server_thread.daemon = True
    server_thread.start()
    
    threading.Timer(1, lambda: webbrowser.open("http://localhost:5000/")).start()
    
    try:
        # Load existing boxes
        load_boxes()
        
        # Main loop
        while True:
            try:
                # Update parking status
                status = update_parking_status()
                
                # Wait for key press (if GUI available)
                try:
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Exit on 'q' key
                    if key == ord('q'):
                        break
                    
                    # Toggle drawing mode on 'd' key
                    elif key == ord('d'):
                        drawing_mode = not drawing_mode
                        if not drawing_mode:
                            save_boxes()
                    
                    # Clear boxes on 'c' key
                    elif key == ord('c'):
                        boxes.clear()
                        save_boxes()
                        print("All boxes cleared!")
                except cv2.error:
                    # If GUI not available, just sleep briefly
                    time.sleep(0.1)
            
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)  # Prevent rapid error loops
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        try:
            update_parking_status()
        except:
            pass
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass