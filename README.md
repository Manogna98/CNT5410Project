<!-- PROJECT TITLE -->
<h1 align="center">Email Spam Detection Using Machine Learning</h1>

<!-- PROJECT DESCRIPTION -->
## <br>**➲ Project description**
Email spam detection system is used to detect spam email using a Machine Learning Model and Python, where we have a dataset containing large no. of emails from which we extract important words using support vector machine to detect spam email.
We then store predicted spam email ids in a file used by Haraka, to block these emails from future spamming.

<!-- PREREQUISTIES -->
## <br>**➲ Prerequisites**
This is list of required packages and modules for the project to be installed :
* <a href="https://www.python.org/downloads/" target="_blank">Python 3.x</a>
* Pandas 
* Numpy
* Scikit-learn
* NLTK
* Haraka

Install all required packages :
 ```sh
  brew install python3
  brew postinstall python3
  pip3 install –upgrade pip
  pip3 install -r requirements.txt  
  brew install node
  npm install -g Haraka
  ```
 
<!-- PYTHON CODING SECTIONS -->
## <br>**➲ Python Coding Sections**
the project code is divided into sections as follows:
<br>

- Section 1 | Data Preprocessing :<br>
In this section we aim to do some operations on the dataset before training the model on it,
<br>processes like :
  - Load dataset
  - Check for duplicates and remove them 
  - Check for missing data for each column 
  - Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
  - Convert the text into a matrix of token counts
  - Split the data into training and testing sets<br><br>

- Section 2 | Model Creation :<br>
The dataset is ready for training, so we create a Support Vector Machine "SVM" model using scikit-learn and thin fit it to the data.<br>

- Section 3 | Model Evaluation :<br>
Finally we evaluate the model by getting accuracy, classification report and confusion matrix.

- Section 4 | Store Domains For Haraka :<br>
Now, we store the spam e-mail IDs in Haraka config files for Haraka to block them at the receiver end.

<!-- INSTALLATION -->
## ➲ Installation
1. Clone the repo
   ```sh
   git clone 
   ```
2. Run the code from cmd
   ```sh
   python3 email_spam_detection_svm.py
   ```
3. Haraka server from cmd
   ```sh
   haraka -c Haraka
   ```
4. Haraka client from cmd
   ```sh
   swaks -f bill@yahoo.com -t matt@hotmail.com -s localhost
   ```





