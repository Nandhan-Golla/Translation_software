Project Title: Offline translation software that converts resource materials from English to Indian regional languages with high linguistic accuracy and cultural relevance, addressing the diverse linguistic landscape of India.
                                                                                      AI/ML & Data Analytics Team -5
 Team Members:
 1) Golla Naga Sri Nandhan - 24MIC7285
 2) Mithun Pattabhi - 23BCE8347
 3) Reddim Nitheesh Kumar Reddy - 23MIC7129

Project Description:
Our Project focuses on breaking language barriers by developing an offline AI-powered translation system that translates resource materials from English to various Indian regional languages. Unlike cloud-based solutions, our software
ensures high accuracy and cultural relevance without requiring an internet connection, making it ideal for remote areas with low connectivity.
Features:
AI-Powered Translation – Uses advanced Neural Machine Translation (NMT) models for context-aware translations.
Cultural & Linguistic Accuracy – Goes beyond word-to-word replacement, understanding grammar, idioms, and cultural nuances.
Optimized for Indian Languages – Supports multiple regional languages.
Efficient Processing – Translates sentences at an average speed of less than 10 seconds per sentence.
How It Works:
Preprocessing: The input text is cleaned, tokenized, and prepared for translation.
Translation Engine: Our AI model translates the text while preserving its meaning and context.
Output Generation: The final translation is structured into natural, readable sentences.

Architectural Diagram:
![architecture_diagram](https://github.com/user-attachments/assets/4952c358-e318-4d73-b7f7-be90db39f164)


References:
[1]	Open Neural Network Exchange (ONNX). (2022). Optimizing Neural Models for Edge and Offline Applications.
[2]	AI4Bharat. (2021). Building Open-Source AI Models for Indian Languages. AI4Bharat Research Publications.
[3]	Wu, Y., Schuster, M., Chen, Z., Le, Q. V., et al. (2016). Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. arXiv preprint arXiv:1609.08144.

## Project Structure

```
offline-translator
├── src
│   ├── model.py          # Contains model definition and loading logic
│   ├── translator.py     # Implements translation functionality
│   └── utils
│       └── __init__.py   # Utility functions for preprocessing and postprocessing
├── requirements.txt      # Lists project dependencies
├── .gitignore            # Specifies files to ignore in version control
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/Nandhan-Golla/Translation_software.git

   cd Translation_software/offline-translator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the offline translator, you can run the `translator.py` script. Ensure that you have the necessary model files available in the specified directory.

Example command:
```
python src/translator.py --input "Hello, how are you?" --source_lang "en" --target_lang "es"
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
