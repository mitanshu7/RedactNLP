# Import the required libraries
import gradio as gr
import cv2 # OpenCV, to read and manipulate images
import easyocr # EasyOCR, for OCR
from pdf2image import convert_from_path # pdf2image, to convert PDF to images
import img2pdf # img2pdf, to convert images to PDF
import torch # PyTorch, for deep learning   
from transformers import pipeline # Hugging Face Transformers, for NER
import os # OS, for file operations
import multiprocessing as mp # Multiprocessing, to speed up the process
from glob import glob # Glob, to get file paths

##########################################################################################################
# Initiate the models

# Easyocr model
print("Initiating easyocr")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Use gpu if available
print("Using gpu if available")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Ner model
print("Initiating nlp pipeline")
nlp = pipeline("token-classification", model="dslim/distilbert-NER", device=device)

# Image format
img_format = 'ppm'

# DPI
dpi = 150

##########################################################################################################
## Functions

def convert_to_images(pdf_file_path):

    # Create a directory to store pdf images
    pdf_images_dir = f'{pdf_file_path}_images'
    os.makedirs(pdf_images_dir, exist_ok=True)

    # Convert the PDF to images
    print("Converting PDF to images...")
    convert_from_path(pdf_file_path, dpi=dpi, thread_count=mp.cpu_count(), output_folder=pdf_images_dir, fmt=img_format)

    # Fix the file names
    for file in os.listdir(pdf_images_dir):
        os.rename(os.path.join(pdf_images_dir, file), os.path.join(pdf_images_dir, file.split('-')[-1]))

    # Return the directory with the images
    return pdf_images_dir

def redact_image(pdf_image_path, redaction_score_threshold):

    # Loop through the images
    print("Redacting sensitive information...")

    print(f"Processing {pdf_image_path}...")
    # Read the image
    cv_image = cv2.imread(pdf_image_path)

    # Read the text from the image
    result = reader.readtext(cv_image, height_ths=0, width_ths=0, x_ths=0, y_ths=0)

    # Get the text from the result
    text = ' '.join([text for (bbox, text, prob) in result])

    # Perform NER on the text
    ner_results = nlp(text)

    # Draw bounding boxes
    for ((bbox, text, prob),ner_result) in zip(result, ner_results):

        # Get the coordinates of the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Calculate the centers of the top and bottom of the bounding box
        # center_top = (int((top_left[0] + top_right[0]) / 2), int((top_left[1] + top_right[1]) / 2))
        # center_bottom = (int((bottom_left[0] + bottom_right[0]) / 2), int((bottom_left[1] + bottom_right[1]) / 2))


        # If the NER result is not empty, and the score is high
        if len(ner_result) > 0 and ner_result['score'] > redaction_score_threshold:

            # Get the entity and score
            # entity = ner_result[0]['entity']
            # score = str(ner_result[0]['score'])

            # Apply a irreversible redaction
            cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 0), -1)
        # else:
            # entity = 'O'
            # score = '0'
            
        # # Draw the bounding box
        # cv2.rectangle(cv_image, top_left, bottom_right, (0, 255, 0), 1)
        # # Draw the entity and score
        # cv2.putText(cv_image, entity, center_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(cv_image, score, center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the redacted image
    print(f"Saving redacted {pdf_image_path}...")
    redacted_image_path = pdf_image_path.replace(f'.{img_format}', f'_redacted.{img_format}')
    # Save the redacted image in png format
    cv2.imwrite(redacted_image_path, cv_image)

    return redacted_image_path

def stich_images_to_pdf(redacted_image_files, input_pdf_name):

    # Sort the redacted images
    redacted_image_files.sort()

    # Convert the redacted images to a single PDF
    print("Converting redacted images to PDF...")
    redacted_pdf_path = f'/tmp/{input_pdf_name}_redacted.pdf'
    with open(redacted_pdf_path, 'wb') as f:
        f.write(img2pdf.convert(redacted_image_files))

    print(f"PDF saved as {redacted_pdf_path}")

    return redacted_pdf_path

def cleanup(redacted_image_files, pdf_images, pdf_images_dir):

    # Remove the directory with the images
    print("Cleaning up...")

    # Remove the redacted images
    for file in redacted_image_files:
        os.remove(file)

    # Remove the pdf images
    for file in pdf_images:
        os.remove(file)

    # Remove the pdf images directory
    os.rmdir(pdf_images_dir)

    return None

def predict(input_pdf_path, sensitivity):

    print("Setting threshold")
    # Convert sensitivity to threshold
    redaction_score_threshold = (100-sensitivity)/100

    # Get file name
    print("Getting filename")
    input_pdf_name = input_pdf_path.split('.')[-2]

    # Convert the PDF to images
    print("Converting pdf to images")
    pdf_images_dir = convert_to_images(input_pdf_path)

    # Get the file paths of the images
    print("Gathering converted images")
    pdf_images = glob(f'{pdf_images_dir}/*.{img_format}', recursive=True)
    pdf_images.sort()

    # Redact images
    print("Redacting images")
    redacted_image_files = []

    for pdf_image in pdf_images:

        redacted_image_files.append(redact_image(pdf_image, redaction_score_threshold))


    # Convert the redacted images to a single PDF
    print("Stitching images to pdf")
    redacted_pdf_path = stich_images_to_pdf(redacted_image_files, input_pdf_name)

    # Cleanup
    print("Cleaning residue")
    cleanup(redacted_image_files, pdf_images, pdf_images_dir)

    return redacted_pdf_path

##########################################################################################################

contact_text = """
# Contact Information

👤  [Mitanshu Sukhwani](https://www.linkedin.com/in/mitanshusukhwani/)

✉️  mitanshu.sukhwani@gmail.com

🐙  [mitanshu7](https://github.com/mitanshu7)
"""

##########################################################################################################
# Gradio interface

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # Title and description
    gr.Markdown("# RedactNLP: Redact your PDF!")
    gr.Markdown("## How redaction happens:")
    gr.Markdown("""
                1. The PDF pages are converted to images.
                2. EasyOCR is run on the converted images to extract text.
                3. "FacebookAI/xlm-roberta-large-finetuned-conll03-english" model does the token classification.
                4. Non-recoverable mask is applied to identified elements.
                """)
    
    # Input Section
    pdf_file_input = gr.File(file_count='single', file_types=['pdf'], label='Upload PDF', show_label=True, interactive=True)
    
    # Slider for results count
    slider_input = gr.Slider(
        minimum=0, maximum=100, value=80, step=1, 
        label="Sensitivity to remove elements. Higher is more sensitive, hence will redact aggresively."
    )

    # Submission Button
    submit_btn = gr.Button("Redact")
    
    # Output section
    output = gr.File(file_count='single', file_types=['pdf'], label='Download redacted PDF', show_label=True, interactive=False)

    # Attribution
    gr.Markdown(contact_text)

    # Link button click to the prediction function
    submit_btn.click(predict, [pdf_file_input, slider_input], output)


################################################################################

if __name__ == "__main__":
    demo.launch()

