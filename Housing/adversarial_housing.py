import sys
import os

import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import parametrize_with_checks

# from fairlearn.adversarial import AdversarialFairnessRegressor
from fairlearn.adversarial._adversarial_mitigation import _AdversarialFairness
from fairlearn.adversarial._constants import _TYPE_COMPLIANCE_ERROR
from fairlearn.adversarial._preprocessor import FloatTransformer

from Data_info import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fairlearn.adversarial import AdversarialFairnessRegressor as AFR

import dalex as dx

username = "rbk440"
alpha = 100.0

CURRENT_PATH = os.getcwd()
print(CURRENT_PATH)

data = pd.read_csv("../../../home/rbk440/Ml_ready_all_percent.csv")
data_class = Data_info(data)

# Prepare the data
X_train_scaled = torch.tensor(data_class.X_train_scaled[:20000], dtype=torch.float32)
y_percentage_train_scaled = torch.tensor(data_class.y_percentage_train_scaled[:20000], dtype=torch.float32)

# Encode sensitive_features
label_encoder = LabelEncoder()
sensitive_features_encoded = label_encoder.fit_transform(data_class.protected_white_vs_other[:20000])
sensitive_features_encoded = sensitive_features_encoded.astype(np.float32)

# Check if all inputs are in the correct range
print("X_train_scaled min and max values:", X_train_scaled.min().item(), X_train_scaled.max().item())
print("y_percentage_train_scaled min and max values:", y_percentage_train_scaled.min().item(), y_percentage_train_scaled.max().item())
print("sensitive_features_encoded min and max values:", sensitive_features_encoded.min(), sensitive_features_encoded.max())

# Assert all values are in range [0, 1]
assert X_train_scaled.min() >= 0 and X_train_scaled.max() <= 1, "X_train_scaled values out of range"
assert y_percentage_train_scaled.min() >= 0 and y_percentage_train_scaled.max() <= 1, "y_percentage_train_scaled values out of range"
assert sensitive_features_encoded.min() >= 0 and sensitive_features_encoded.max() <= 1, "sensitive_features_encoded values out of range"

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 266)
        self.fc2 = nn.Linear(266, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Ensure sigmoid activation
        return x

# Define a simpler Adversary model
class SimpleAdversary(nn.Module):
    def __init__(self, input_size):
        super(SimpleAdversary, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Ensure sigmoid activation for binary output
        return x


# Initialize models
input_size = X_train_scaled.shape[1]
predictor_model = MLP(input_size)
adversary_model = SimpleAdversary(1)  # Adversary input size depends on predictor's output

# Optimizers
predictor_optimizer = optim.Adam(predictor_model.parameters(), lr=0.001)
adversary_optimizer = optim.Adam(adversary_model.parameters(), lr=0.001)

# Check model outputs
batch_slice = slice(0, 32)
X_batch = X_train_scaled[batch_slice]
y_batch = y_percentage_train_scaled[batch_slice]
sensitive_batch = sensitive_features_encoded[batch_slice]

predictor_output = predictor_model(X_batch)
print("Predictor output min and max values:", predictor_output.min().item(), predictor_output.max().item())

adversary_output = adversary_model(predictor_output)
print("Adversary output min and max values:", adversary_output.min().item(), adversary_output.max().item())

# Train the model using Fairlearn's AdversarialFairnessRegressor
model_tr = AFR(
    backend="torch",
    predictor_model=predictor_model,
    adversary_model=adversary_model,
    predictor_optimizer=predictor_optimizer,
    adversary_optimizer=adversary_optimizer,
    constraints="demographic_parity",
    learning_rate=0.001,
    alpha=alpha,
    epochs=100,
    batch_size=32,
    shuffle=True,
    progress_updates=60,
    skip_validation=True
)


model_tr.fit(X=X_train_scaled.numpy(), y=y_percentage_train_scaled.numpy(), sensitive_features=sensitive_features_encoded)
print("Acabé entrenamiento")

print(model_tr)
# print(model_afr)
print(predictor_model)
print(adversary_model)

# Save the trained models
PATH_PRED = f'../../../home/{username}/housing_pred_{alpha}.pth'
torch.save(predictor_model.state_dict(), PATH_PRED)

PATH_ADV = f'../../../home/{username}/housing_adv_{alpha}.pth'
torch.save(predictor_model.state_dict(), PATH_ADV)

# Save model performance and fairness results
explainer = dx.Explainer(
    predictor_model,
    data=data_class.X_val_scaled[:20000],
    y=data_class.y_percentage_val_scaled[:20000],
    predict_function=lambda model, data: model(torch.tensor(data.to_numpy(), dtype=torch.float32)).view(-1).detach().numpy(),
    model_type="regression",
    label="Og housing"
)

mod_res = explainer.model_performance().result
mf_model = explainer.model_fairness(protected=data_class.protected_white_vs_other[:20000], privileged="White")
fairness = mf_model.result
print(fairness)
mod_res.to_csv(f"../../../home/{username}/mod_{str(alpha)}_res.csv")
fairness.to_csv(f"../../../home/{username}/fairness_{str(alpha)}.csv")