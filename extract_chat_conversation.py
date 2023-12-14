import cv2  # Import OpenCV for image processing
import easyocr  # Import EasyOCR for optical character recognition
import re  # Import regex for pattern matching
import os  # Import os for file and directory operations
from datetime import datetime  # Import datetime for handling timestamps

# Constants for OCR processing
CONFIDENCE_THRESHOLD = 0.1  # Confidence threshold for OCR detection
MINIMUM_X_THRESHOLD = 20  # Minimum x-coordinate threshold for text categorization
PADDING = 10  # Padding to be added around text bounding boxes

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image from the given path.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the original image and the grayscale image.
    """
    # Load an image from the given path
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def extract_text_details(gray_image):
    """
    Extracts text details from a grayscale image.

    Parameters:
    gray_image (numpy.ndarray): The grayscale image to extract text details from.

    Returns:
    list: A list of text details extracted from the image.
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)
    # Extract text details from the grayscale image
    details = reader.readtext(gray_image, detail=1)
    return details

def add_padding(x, y, w, h, max_width, max_height):
    """
    Add padding to the bounding box dimensions.

    Args:
        x (int): The x-coordinate of the bounding box.
        y (int): The y-coordinate of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        max_width (int): The maximum width of the bounding box.
        max_height (int): The maximum height of the bounding box.

    Returns:
        tuple: A tuple containing the modified x, y, w, and h values.
    """
    # Add padding to the bounding box dimensions
    x = max(x - PADDING, 0)
    y = max(y - PADDING, 0)
    w = min(w + 2 * PADDING, max_width - x)
    h = min(h + 2 * PADDING, max_height - y)
    return x, y, w, h

def draw_and_extract_grouped_text(image, details):
    """
    Draws bounding boxes around text in an image and extracts the grouped text.

    Args:
        image: The image to process.
        details: A list of tuples containing the bounding box coordinates, text, and probability.

    Returns:
        A list of dictionaries, each containing the extracted text and its corresponding bounding box coordinates.
    """
    # Draw bounding boxes and extract text
    extracted_texts = []
    for (bbox, text, prob) in details:
        if prob >= CONFIDENCE_THRESHOLD:
            # Unpack bounding box coordinates
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(bottom_right[0], top_right[0]))
            y_max = int(max(bottom_right[1], bottom_left[1]))

            # Check if text is within the defined thresholds
            if x_min < MINIMUM_X_THRESHOLD or abs(x_max - x_min) > (image.shape[1] - 150):
                continue

            # Add padding and draw bounding boxes
            x, y, w, h = add_padding(x_min, y_min, x_max - x_min, y_max - y_min, image.shape[1], image.shape[0])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            extracted_texts.append({'text': text, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max})
    return extracted_texts

def clean_and_extract_timestamp(text):
    """
    Cleans and extracts a timestamp from the given text.

    Parameters:
        text (str): The text to search for a timestamp.

    Returns:
        Tuple[str, Optional[str]]: A tuple containing the cleaned text and the extracted timestamp. If a timestamp is found, it is converted to the format 'hh:mm AM/PM'. If no timestamp is found, the original text is returned and the timestamp is None.
    """
    # Define regex for timestamp extraction
    timestamp_regex = r'(\d{1,2}:\d{2}(?:\s?[AP]M)?)|(\d{1,2}\.\d{2})'
    # Search for timestamp in the text
    timestamp = re.search(timestamp_regex, text)
    if timestamp:
        timestamp_str = timestamp.group()
        # Convert 'hh.mm' format to 'hh:mm AM/PM'
        if '.' in timestamp_str:
            hour, minute = map(int, timestamp_str.split('.'))
            am_pm = 'AM' if hour < 12 else 'PM'
            hour = hour % 12
            hour = hour if hour else 12
            timestamp_str = f"{hour}:{minute:02d} {am_pm}"
        return '', timestamp_str
    return text, None

def categorize_text(extracted_texts):
    """
    Categorize texts into formatted messages.

    Args:
        extracted_texts (list): A list of dictionaries containing information about extracted texts.
            Each dictionary should have the following keys:
                - text (str): The extracted text.
                - timestamp (str): The timestamp associated with the text.
                - x_min (int): The minimum x-coordinate of the text bounding box.
                - y_min (int): The minimum y-coordinate of the text bounding box.

    Returns:
        list: A list of formatted messages.
            Each formatted message is a string that includes the type of the message (Sender or Receiver),
            the text content, and the associated timestamp.
    """
    # Categorize texts into formatted messages
    formatted_messages = []
    current_message = {'text': '', 'timestamp': '', 'type': None}

    for text_info in sorted(extracted_texts, key=lambda x: (x['y_min'], x['x_min'])):
        cleaned_text, timestamp = clean_and_extract_timestamp(text_info['text'])
        # Determine if the text is from the sender or receiver
        text_type = "Receiver" if text_info['x_min'] < 150 else "Sender"

        if timestamp:
            # Append current message with timestamp
            if current_message['text']:
                formatted_message = f"{current_message['type']}: {current_message['text']} {current_message['timestamp']}".strip()
                formatted_messages.append(formatted_message)
                current_message = {'text': '', 'timestamp': '', 'type': None}
            current_message['timestamp'] = timestamp
        elif cleaned_text:
            # Append text to the current message
            current_message['text'] += (' ' + cleaned_text).strip()
            current_message['type'] = text_type

    # Append the last message if it exists
    if current_message['text']:
        formatted_message = f"{current_message['type']}: {current_message['text']} {current_message['timestamp']}".strip()
        formatted_messages.append(formatted_message)

    return formatted_messages

def main(image_path, output_folder):
    """
    Process an image and save the output.

    Args:
        image_path (str): The path to the image file.
        output_folder (str): The path to the folder where the output image will be saved.

    Returns:
        list: A list of categorized texts extracted from the image.
    """
    # Process an image and save the output
    image, gray_image = load_and_preprocess_image(image_path)
    details = extract_text_details(gray_image)
    extracted_texts = draw_and_extract_grouped_text(image, details)
    categorized_texts = categorize_text(extracted_texts)

    # Save the processed image with bounding boxes
    image_name = os.path.basename(image_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_image_path = os.path.join(output_folder, f'{image_name}_{timestamp}.png')
    cv2.imwrite(output_image_path, image)

    return categorized_texts

# Process all images in the input folder
input_folder = 'input'
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process each image and save categorized texts
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path}...")
        formatted_messages = main(image_path, output_folder)

        # Save formatted messages to a text file
        base_filename = os.path.splitext(filename)[0]
        output_filename = os.path.join(output_folder, f'{base_filename}_categorized_texts.txt')
        with open(output_filename, 'w') as f:
            for message in formatted_messages:
                f.write(f"{message}\n")
            f.write("\n")
