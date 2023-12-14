import cv2
import easyocr
import re
import os
from datetime import datetime

# Constants for OCR processing
CONFIDENCE_THRESHOLD = 0.1
MINIMUM_X_THRESHOLD = 20
PADDING = 10

def load_and_preprocess_image(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def extract_text_details(gray_image):
    # Use EasyOCR to extract text details from the image
    reader = easyocr.Reader(['en'], gpu=False)
    details = reader.readtext(gray_image, detail=1)
    return details

def add_padding(x, y, w, h, max_width, max_height):
    # Add padding around the detected text area
    x = max(x - PADDING, 0)
    y = max(y - PADDING, 0)
    w = min(w + 2 * PADDING, max_width - x)
    h = min(h + 2 * PADDING, max_height - y)
    return x, y, w, h

def draw_and_extract_grouped_text(image, details):
    # Draw bounding boxes around detected text and extract text details
    extracted_texts = []
    for (bbox, text, prob) in details:
        if prob >= CONFIDENCE_THRESHOLD:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(bottom_right[0], top_right[0]))
            y_max = int(max(bottom_right[1], bottom_left[1]))

            if x_min < MINIMUM_X_THRESHOLD or abs(x_max - x_min) > (image.shape[1] - 150):
                continue

            x, y, w, h = add_padding(x_min, y_min, x_max - x_min, y_max - y_min, image.shape[1], image.shape[0])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            extracted_texts.append({'text': text, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max})
    return extracted_texts

def clean_and_extract_timestamp(text):
    # Extract and format timestamp from text
    timestamp_regex = r'(\d{1,2}:\d{2}(?:\s?[AP]M)?)|(\d{1,2}\.\d{2})'
    timestamp = re.search(timestamp_regex, text)
    if timestamp:
        timestamp_str = timestamp.group()
        if '.' in timestamp_str:
            hour, minute = map(int, timestamp_str.split('.'))
            am_pm = 'AM' if hour < 12 else 'PM'
            hour = hour % 12
            hour = hour if hour else 12
            timestamp_str = f"{hour}:{minute:02d} {am_pm}"
        return '', timestamp_str
    return text, None

def categorize_text(extracted_texts):
    # Categorize extracted texts into formatted messages
    formatted_messages = []
    current_message = {'text': '', 'timestamp': '', 'type': None}

    for text_info in sorted(extracted_texts, key=lambda x: (x['y_min'], x['x_min'])):
        cleaned_text, timestamp = clean_and_extract_timestamp(text_info['text'])
        text_type = "Receiver" if text_info['x_min'] < 150 else "Sender"

        if timestamp:
            if current_message['text']:
                formatted_message = f"{current_message['type']}: {current_message['text']} {current_message['timestamp']}".strip()
                formatted_messages.append(formatted_message)
                current_message = {'text': '', 'timestamp': '', 'type': None}
            current_message['timestamp'] = timestamp
        elif cleaned_text:
            current_message['text'] += (' ' + cleaned_text).strip()
            current_message['type'] = text_type

    if current_message['text']:
        formatted_message = f"{current_message['type']}: {current_message['text']} {current_message['timestamp']}".strip()
        formatted_messages.append(formatted_message)

    return formatted_messages

def main(image_path, output_folder):
    # Main function to process an image and save output
    image, gray_image = load_and_preprocess_image(image_path)
    details = extract_text_details(gray_image)
    extracted_texts = draw_and_extract_grouped_text(image, details)
    categorized_texts = categorize_text(extracted_texts)

    image_name = os.path.basename(image_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_image_path = os.path.join(output_folder, f'{image_name}_{timestamp}.png')
    cv2.imwrite(output_image_path, image)

    return categorized_texts

input_folder = 'input'
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path}...")
        formatted_messages = main(image_path, output_folder)

        base_filename = os.path.splitext(filename)[0]
        output_filename = os.path.join(output_folder, f'{base_filename}_categorized_texts.txt')
        with open(output_filename, 'w') as f:
            for message in formatted_messages:
                f.write(f"{message}\n")
            f.write("\n")
