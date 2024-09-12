# RedactNLP: Redact Your PDF!

RedactNLP is a tool that allows you to automatically redact sensitive information from PDF documents using natural language processing and computer vision techniques.  

Visit https://huggingface.co/spaces/bluuebunny/RedactNLP for the hosted demo on Huggingface Spaces.

## How Redaction Works

1. **PDF to Images**: The PDF pages are converted into images.
2. **Text Extraction**: Using EasyOCR, text is extracted from the images.
3. **Entity Identification**: The "dslim/distilbert-NER" model classifies tokens in the extracted text to identify sensitive elements such as names, locations, or organizations.
4. **Redaction**: A non-recoverable mask is applied to all identified sensitive elements, ensuring that they cannot be recovered from the redacted document.

## Features

- Automatic redaction of sensitive information from PDFs.
- Uses OCR to handle scanned or image-based PDFs.
- Leverages state-of-the-art NLP models for entity recognition.
- Ensures irreversible redaction of confidential information.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mitanshu7/RedactNLP.git
cd RedactNLP
pip install -r requirements.txt
```

## Usage

To redact a PDF:

```bash
python app.py
```
Then navigate to http://localhost:7860

## Contributing

Feel free to open issues or submit pull requests if you would like to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
