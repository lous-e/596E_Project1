## Running the Review Dangerousness Classifier Using FlaskML

You can deploy your binary classification model using FlaskML to provide a web-based interface for users to classify app reviews as either Safe or Dangerous. 
Flask is a lightweight web framework in Python that allows you to create RESTful APIs, which can be used to interact with the model.

#### Create the Flask App: 

In the project directory, create a new file called `Server_app.py`. This file will define the web service and handle requests to classify app reviews.

#### How It Works:

Loading the Model: The pretrained and fine-tuned BERT model is loaded using BertForSequenceClassification and its weights are restored from the saved model file (bert_classifier.pth).
Tokenization: Each incoming review is tokenized using the BERT tokenizer to convert the review text into a format that the model can understand.
Classification: The model predicts whether the review indicates that the app is dangerous (1) or safe (0).
API Endpoint: The /classify route is defined as a POST method, where users can submit their review text as JSON and receive the classification result.

### Run using flask - individual Review Classifiction

`python Server_app.py `

This will start the web application to classify the review 

Open the powershell

Use the command to run the model

 `Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -Body '{"text":"This app is  best for children"}' -ContentType "application/json"`

 We can change the -Body to any text review that need to be classified.

 ![image](https://github.com/user-attachments/assets/2ae04eb2-81b6-4c8b-9d75-846ff459f664)


 ### Output:

 
![image](https://github.com/user-attachments/assets/15de838f-6a44-4858-99f5-1bdf42d578f6)


Note: To run using linux os:

`curl -X POST -F "file=@reviews.csv" http://localhost:5000/predict`
