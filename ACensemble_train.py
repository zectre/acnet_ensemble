import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
from ray import tune
from ray.tune import CLIReporter


# Set the dataset directory
dataset_dir = 'dataset/'

# Function to load training data
def load_trainingdata():
    TOA_xtrain_path = os.path.join(dataset_dir, 'TOA_xtrain.npy')
    angles_xtrain_path = os.path.join(dataset_dir, 'angles_xTrain.npy')
    AOT_xtrain_path = os.path.join(dataset_dir, 'AOT_xTrain.npy')
    iCOR_YTrain_path = os.path.join(dataset_dir, 'iCOR_YTrain.npy')

    TOA_xtrain = np.load(TOA_xtrain_path)
    angles_xtrain = np.load(angles_xtrain_path)
    AOT_xtrain = np.load(AOT_xtrain_path)
    ytrain_iCOR = np.load(iCOR_YTrain_path)

    # Similarly, load validation data

    return TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR

# Function to load testing data
def load_validata():
    TOA_XVali_path = os.path.join(dataset_dir, 'TOA_XVali.npy')
    angles_XVali_path = os.path.join(dataset_dir, 'angles_XVali.npy')
    AOT_XVali_path = os.path.join(dataset_dir, 'AOT_XVali.npy')
    iCOR_YVali_path = os.path.join(dataset_dir, 'iCOR_YVali.npy')

    TOA_xvali = np.load(TOA_XVali_path)
    angles_xvali = np.load(angles_XVali_path)
    AOT_xvali = np.load(AOT_XVali_path)
    yvali_iCOR = np.load(iCOR_YVali_path)

    # Similarly, load other testing data

    return TOA_xvali, angles_xvali, AOT_xvali, yvali_iCOR

# Load training data
TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR = load_trainingdata()

# Load testing data
TOA_xvali, angles_xvali, AOT_xvali, yvali_iCOR = load_validata()

# Convert NumPy arrays to PyTorch tensors
TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, yvali_iCOR = map(torch.tensor, 
                                                                                                    [TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, yvali_iCOR])
# Combine inputs and targets into a PyTorch dataset
train_dataset = TensorDataset(TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR)
val_dataset = TensorDataset(TOA_xvali, angles_xvali, AOT_xvali, yvali_iCOR)

# DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the PyTorch model class
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        # Create 3CN layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.flatten = nn.Flatten()

        # Combine x (TOA_extracted from CNN layers) and Angles data and AOT data
        self.fc1 = nn.Linear(input_size, 324)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(324, 400)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(400, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 200)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(200, 100)
        self.dropout5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(100, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, TOA_input, angles_input, AOT_input):
        x = self.conv1(TOA_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = torch.reshape(x, (x.size(0), -1))
        combined = torch.cat((x, angles_input, AOT_input), dim=1)
        y = self.fc1(combined)
        y = self.dropout1(y)
        y = self.fc2(y)
        y = self.dropout2(y)
        y = self.fc3(y)
        y = self.dropout3(y)
        y = self.fc4(y)
        y = self.dropout4(y)
        y = self.fc5(y)
        y = self.dropout5(y)
        y = self.fc6(y)
        y = self.sigmoid(y)
        return y
    
    
# PyTorch Lightning Module for Ray Tune
class PyTorchModel(tune.Trainable):
    def _setup(self, config):
        self.model = MyModel(input_size=324).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.batch_size = config["batch_size"]

    def _train(self):
        self.model.train()
        for TOA_batch, angles_batch, AOT_batch, y_batch in train_loader:
            TOA_batch, angles_batch, AOT_batch, y_batch = (
                TOA_batch.to(device),
                angles_batch.to(device),
                AOT_batch.to(device),
                y_batch.to(device),
            )
            self.optimizer.zero_grad()
            outputs = self.model(TOA_batch, angles_batch, AOT_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

        # Validation
        self.model.eval()
        with torch.no_grad():
            for TOA_batch, angles_batch, AOT_batch, y_batch in val_loader:
                TOA_batch, angles_batch, AOT_batch, y_batch = (
                    TOA_batch.to(device),
                    angles_batch.to(device),
                    AOT_batch.to(device),
                    y_batch.to(device),
                )
                val_outputs = self.model(TOA_batch, angles_batch, AOT_batch)
                val_loss = self.criterion(val_outputs, y_batch)

        return {"loss": loss.item(), "val_loss": val_loss.item()}
    
# Ray Tune Configuration Space
config_space = {
    "lr": tune.loguniform(1e-6, 1e-2),
    "batch_size": tune.choice([64, 128, 256]),
}


# Set up TensorBoard logging
tensorboard_dir = os.path.abspath("tensorboard_logs")

# Ray Tune Hyperparameter Search
analysis = tune.run(
    PyTorchModel,
    config=config_space,
    stop={"training_iteration": 10},  # Adjust the number of iterations
    num_samples=5,  # Adjust the number of samples
    resources_per_trial={"gpu": 1},  # Use 1 GPU per trial
    local_dir=tensorboard_dir
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="val_loss", mode="min")
best_lr = best_config["lr"]
best_batch_size = best_config["batch_size"]

# Train the final model with the best hyperparameters
final_model = PyTorchModel(config={"lr": best_lr, "batch_size": best_batch_size})
for _ in range(10):  # Adjust the number of iterations
    final_model.train()
    
# Save the trained model
torch.save(final_model.model.state_dict(), 'final_model.pth')