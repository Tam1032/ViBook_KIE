# Key Information Extraction from Vietnamese book cover images
This is the source code for the paper "Key Information Extraction from Vietnamese book cover images" which is presented at the RIVF2023 International Conference. 
In this project, we aim to classify the text extracted from the OCR model into 4 related fields: "title", "author", "publisher", "other" by mainly utilzing NLP techniques.
We first upload the code to build Machine Learning models following two approaches: text classification approach and text-box classifcation approach. 
The text-box classifcation approach utilize the bounding box coordinates to enhance the result beside the text content.
We will upload the source code for other approaches later.
## Installation
You can install the necessary libraries by using the following command: ``pip install -r requirements.txt``
## Model performances
The best Machine Learning model for all these two approaches is the SVM model. On the F1-score metric, it achieved the result 0.93 following the text classifcation approach and 0.96 for the text-box classication approach.
You can check the results of the SVM and other Machine Learning models by running the two Jupyter notebooks.
