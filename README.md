# Chat Text Extraction and Categorization Enhancement

## Project Overview

This project focuses on enhancing the capabilities of a Python script to extract and categorize text from WhatsApp chat screenshots. The primary goal is to accurately categorize text into sender and receiver, detect emojis, and recognize timestamps associated with each message.

## Folder Structure

- `emoji_images/`: Contains chat screenshots with emojis for analysis.
- `extract_chat_conversation.py`: Main Python script for text extraction and categorization.
- `input/`: Directory for chat screenshots to be analyzed.
- `my_screenshots/`: Personal chat screenshots used for testing.
- `output/`: Output directory for processed images.
- `README.md`: Documentation of the project.
- `requirements.txt`: Lists the Python dependencies (easyocr, opencv-python).

## Key Updates

1. **Improved Categorization**: The script can now categorize text into sender and receiver based on the alignment and position of the text box in the image.
2. **Emoji Detection**: Current OCR libraries struggle to detect emojis. Alternative libraries or APIs are being explored for this feature.
3. **Timestamp Recognition**: Regular expressions are used to extract common time formats from the text. The script has been enhanced to recognize and format timestamps accurately.

## Technical Details

- **Original Script**: The initial version used Pytesseract, which was fast but inadequate for timestamp detection.
- **EasyOCR**: Switched to EasyOCR for better timestamp recognition, despite being more resource-intensive and requiring a GPU for optimal performance.
- **Sender/Receiver Identification**: Based on the position of the text box in the image, the script categorizes text into sender or receiver.
- **Timestamp Formatting**: The script includes logic to convert various timestamp formats (like 'hh.mm') to a standard 'hh:mm AM/PM' format and associate them correctly with the corresponding text.

### Sender and Receiver Identification

- The script identifies the sender and receiver based on the text box's position and alignment within the image. Text boxes on the left and aligned to the left are categorized as the sender, while those on the right and aligned to the right are categorized as the receiver.

### Timestamps

- Timestamps are crucial for the context and are extracted using a regular expression that matches common time formats. The script has been updated to handle different formats like "hh:mm" and "hh.mm" and convert them into a standardized "hh:mm AM/PM" format.

## Final Code Overview

The final script, `extract_chat_conversation.py`, processes images from the `input/` folder and outputs the categorized text into the `output/` folder. The script uses EasyOCR for text detection and custom logic for categorizing and formatting the extracted text.

### Key Functions:

- `load_and_preprocess_image`: Loads an image and converts it to grayscale.
- `extract_text_details`: Uses EasyOCR to extract text details from the image.
- `draw_and_extract_grouped_text`: Draws bounding boxes around detected text and extracts details.
- `clean_and_extract_timestamp`: Extracts and formats timestamps from the text.
- `categorize_text`: Categorizes extracted texts into formatted messages based on their position and content.
- `main`: Processes each image and saves the output.

## Running the Script

Ensure you have the required dependencies installed by running:

```bash
pip install -r requirements.txt
```

To process images, place them in the `input/` folder and run:

```bash
python extract_chat_conversation.py
```

Processed outputs will be saved in the `output/` folder.

## Future Enhancements

- **Emoji Detection**: Investigate advanced OCR solutions or APIs for effective emoji detection.
- **Optimization**: Explore methods to optimize the script for speed and efficiency, especially for systems without GPU support.

## Conclusion

This project represents a significant step in enhancing chat text extraction and categorization. While challenges like emoji detection remain, the improvements in timestamp recognition and text categorization are notable advancements.

*Note: This project is part of an ongoing effort to improve text extraction from images, with a focus on chat screenshots. Future updates may include enhanced emoji detection and further optimizations.*