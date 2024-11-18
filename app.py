import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

# Load the model
model = SimpleNN(1536, 3)
model.load_state_dict(torch.load('model.pth'))
model.eval()


# First, install the safetensors library if you haven't already
# pip install safetensors
import os, time
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["OPENAI_API_BASE"] = "https://aocc-gpt-eus2.openai.azure.com/"
os.environ["OPENAI_DEPLOYMENT_NAME"] = "text-embedding-ada-002"

from openai import AzureOpenAI

client = AzureOpenAI(
    # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
    api_version= os.environ["OPENAI_API_VERSION"],
    # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME"],
)



def preprocess_text(text):
    # time.sleep(0.01)
    print(text)
    return client.embeddings.create(input = [text], model=os.environ["OPENAI_DEPLOYMENT_NAME"]).data[0].embedding

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    data = request.json
    input_text = data['text']

    # Record the start time
    start_time = time.time()

    # Preprocess the text
    input_tensor = torch.tensor([preprocess_text(input_text)])

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class and confidence
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Prepare the response
    response = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'elapsed_time': elapsed_time
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)

# curl -X POST -H "Content-Type: application/json" -d '{"text":"Your input text here"}' http://localhost:5000/predict
