# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1027" height="695" alt="image" src="https://github.com/user-attachments/assets/1440702b-e4af-497b-ae63-09ea9058293f" />


## DESIGN STEPS


### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.


### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.


### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.


### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.





## PROGRAM

### Name: SUDHAARSAN S

### Register Number: 212224040334

```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

```


```python
dataset = pd.read_csv("/content/drive/MyDrive/customers.csv")
dataset.head(5)
```





  <div id="df-b59d10f6-2268-44fa-812e-0a5af77f80ac" class="colab-df-container">
    <div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gender</th>
      <th>Ever_Married</th>
      <th>Age</th>
      <th>Graduated</th>
      <th>Profession</th>
      <th>Work_Experience</th>
      <th>Spending_Score</th>
      <th>Family_Size</th>
      <th>Var_1</th>
      <th>Segmentation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>462809</td>
      <td>Male</td>
      <td>No</td>
      <td>22</td>
      <td>No</td>
      <td>Healthcare</td>
      <td>1.0</td>
      <td>Low</td>
      <td>4.0</td>
      <td>Cat_4</td>
      <td>D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>462643</td>
      <td>Female</td>
      <td>Yes</td>
      <td>38</td>
      <td>Yes</td>
      <td>Engineer</td>
      <td>NaN</td>
      <td>Average</td>
      <td>3.0</td>
      <td>Cat_4</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>466315</td>
      <td>Female</td>
      <td>Yes</td>
      <td>67</td>
      <td>Yes</td>
      <td>Engineer</td>
      <td>1.0</td>
      <td>Low</td>
      <td>1.0</td>
      <td>Cat_6</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>461735</td>
      <td>Male</td>
      <td>Yes</td>
      <td>67</td>
      <td>Yes</td>
      <td>Lawyer</td>
      <td>0.0</td>
      <td>High</td>
      <td>2.0</td>
      <td>Cat_6</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>462669</td>
      <td>Female</td>
      <td>Yes</td>
      <td>40</td>
      <td>Yes</td>
      <td>Entertainment</td>
      <td>NaN</td>
      <td>High</td>
      <td>6.0</td>
      <td>Cat_6</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b59d10f6-2268-44fa-812e-0a5af77f80ac')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  

  </div>





```python
dataset.drop(columns=['ID'])
```





  <div id="df-4a02e25d-1aca-4b57-b28e-9855aa6bc536" class="colab-df-container">
    <div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Ever_Married</th>
      <th>Age</th>
      <th>Graduated</th>
      <th>Profession</th>
      <th>Work_Experience</th>
      <th>Spending_Score</th>
      <th>Family_Size</th>
      <th>Var_1</th>
      <th>Segmentation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>No</td>
      <td>22</td>
      <td>No</td>
      <td>Healthcare</td>
      <td>1.0</td>
      <td>Low</td>
      <td>4.0</td>
      <td>Cat_4</td>
      <td>D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>Yes</td>
      <td>38</td>
      <td>Yes</td>
      <td>Engineer</td>
      <td>NaN</td>
      <td>Average</td>
      <td>3.0</td>
      <td>Cat_4</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>Yes</td>
      <td>67</td>
      <td>Yes</td>
      <td>Engineer</td>
      <td>1.0</td>
      <td>Low</td>
      <td>1.0</td>
      <td>Cat_6</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Yes</td>
      <td>67</td>
      <td>Yes</td>
      <td>Lawyer</td>
      <td>0.0</td>
      <td>High</td>
      <td>2.0</td>
      <td>Cat_6</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>Yes</td>
      <td>40</td>
      <td>Yes</td>
      <td>Entertainment</td>
      <td>NaN</td>
      <td>High</td>
      <td>6.0</td>
      <td>Cat_6</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10690</th>
      <td>Male</td>
      <td>No</td>
      <td>29</td>
      <td>No</td>
      <td>Healthcare</td>
      <td>9.0</td>
      <td>Low</td>
      <td>4.0</td>
      <td>Cat_6</td>
      <td>B</td>
    </tr>
    <tr>
      <th>10691</th>
      <td>Female</td>
      <td>No</td>
      <td>35</td>
      <td>Yes</td>
      <td>Doctor</td>
      <td>1.0</td>
      <td>Low</td>
      <td>1.0</td>
      <td>Cat_6</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10692</th>
      <td>Female</td>
      <td>No</td>
      <td>53</td>
      <td>Yes</td>
      <td>Entertainment</td>
      <td>NaN</td>
      <td>Low</td>
      <td>2.0</td>
      <td>Cat_6</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10693</th>
      <td>Male</td>
      <td>Yes</td>
      <td>47</td>
      <td>Yes</td>
      <td>Executive</td>
      <td>1.0</td>
      <td>High</td>
      <td>5.0</td>
      <td>Cat_4</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10694</th>
      <td>Female</td>
      <td>No</td>
      <td>43</td>
      <td>Yes</td>
      <td>Healthcare</td>
      <td>9.0</td>
      <td>Low</td>
      <td>3.0</td>
      <td>Cat_7</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
<p>10695 rows × 10 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4a02e25d-1aca-4b57-b28e-9855aa6bc536')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  

  </div>





```python
dataset.fillna({"Work_Experience": 0, "Family_Size": dataset["Family_Size"].median()}, inplace=True)
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    dataset[col] = LabelEncoder().fit_transform(dataset[col])
```


```python
label_encoder = LabelEncoder()
dataset["Segmentation"] = label_encoder.fit_transform(dataset
 ["Segmentation"])
X = dataset.drop(columns=["Segmentation"])
y = dataset["Segmentation"].values
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

    /tmp/ipykernel_7718/920478267.py:7: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      y_train = torch.tensor(y_train, dtype=torch.long)
    /tmp/ipykernel_7718/920478267.py:8: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      y_test = torch.tensor(y_test, dtype=torch.long)
    /tmp/ipykernel_7718/920478267.py:10: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      X_train = torch.tensor(X_train, dtype=torch.float32)
    /tmp/ipykernel_7718/920478267.py:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      X_test = torch.tensor(X_test, dtype=torch.float32)
    /tmp/ipykernel_7718/920478267.py:12: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      y_train = torch.tensor(y_train, dtype=torch.long)
    /tmp/ipykernel_7718/920478267.py:13: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      y_test = torch.tensor(y_test, dtype=torch.long)
    


```python
input_size = X_train.shape[1]
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=100)
```

    Epoch [100/100], Loss: 1.1647
    


```python
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())
```


```python
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name: SUDHARSAN S")
print("Register No:212224040334")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
```
```
    Name: SUDHARSAN S
    Register No:212224040334
    Test Accuracy: 0.46%
    Confusion Matrix:
     [[167 128 136 153]
     [123 137 145  85]
     [ 63  76 272  61]
     [101  33  57 402]]
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.37      0.29      0.32       584
               1       0.37      0.28      0.32       490
               2       0.45      0.58      0.50       472
               3       0.57      0.68      0.62       593
    
        accuracy                           0.46      2139
       macro avg       0.44      0.45      0.44      2139
    weighted avg       0.44      0.46      0.44      2139
    
    
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```


    
![png](ex2_11_0.png)
    



```python
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)

    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: SUDHARSAN S")
print("Register No:212224040334")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
```
```
    Name: THAVANESH B
    Register No:212224040352
    Predicted class for sample input: 3
    Actual class for sample input: 3
    

```
### Dataset Information
<img width="1122" height="629" alt="image" src="https://github.com/user-attachments/assets/790ad456-1a89-477a-abc6-5dea52f9b6e2" />


### OUTPUT

## Confusion Matrix

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/24d6e4b7-161b-4eaa-bef3-683bb10f8907" />


## Classification Report
<img width="909" height="479" alt="image" src="https://github.com/user-attachments/assets/872533bf-c4cd-4d4c-8277-ece13ed26d9d" />


### New Sample Data Prediction
<img width="1335" height="398" alt="image" src="https://github.com/user-attachments/assets/1f532252-9e96-42f4-9d39-d9bdba7691ec" />


## RESULT
Neural network classification model for the given dataset is successfully developed.
