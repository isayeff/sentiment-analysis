# SentimentSense

**SentimentSense** is a sentiment analysis project that classifies text as positive or negative using Naïve Bayes and a rule-based dictionary approach. It works with two datasets: **Rotten Tomatoes movie reviews** and **Nokia phone reviews**. The project compares machine learning (Naïve Bayes) and rule-based methods for sentiment classification.

### Key Features:

* **Naïve Bayes** for sentiment classification based on training data.
* **Dictionary-based classification** using predefined positive and negative sentiment words.
* **Improved rule-based system** with features like negation handling, intensifiers, and diminisher rules.
* **Evaluation metrics** such as accuracy, precision, recall, and F1 score.

### Project Setup:

This project does not use any external Python libraries, so there's no need for a `requirements.txt` file. It can be run directly with Python, as it relies only on built-in Python libraries like `re` for regular expressions.

### How to Run:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Run the Python script:**
   The project can be run directly using Python:

   ```bash
   python Sentiment.py
   ```

### Expected Output:

The program will output classification results and evaluation metrics, such as accuracy, precision, recall, and F1 score, for the test datasets.

### Datasets Used:

* **Rotten Tomatoes movie reviews** for training and testing sentiment analysis.
* **Nokia phone reviews** for testing the rule-based and Naïve Bayes approaches.

This README.md file explains the project clearly and mentions that no external libraries are used, so there's no need for a Python environment setup or `requirements.txt` file. Make sure to replace `YOUR_USERNAME` with your actual GitHub username in the clone URL.
