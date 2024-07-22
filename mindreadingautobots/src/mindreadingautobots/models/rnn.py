"""rnn.py - RNN model for forecasting and next-bit prediction plus hyperparameter tuning."""
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from ray import train




class BinaryRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(BinaryRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden):
        """
        Compute the forward pass for a given hidden layer. 

        NOTE: the output is flattened before being returned.

        Args:
            x: (batch_size, seq_length, input_size)
            hidden: (n_layers, batch_size, hidden_dim)

        Returns:
            output: (batch_size*seq_length, output_dim)
            hidden: (n_layers, batch_size, hidden_dim)
        """
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)

        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.reshape(-1, self.hidden_dim)  
        
        # get final output 
        output = self.fc(r_out)
        
        return output, hidden
    
    def predict(self, x, hidden):
        """Predict the next bit given the current input and hidden state."""
        self.eval()
        batch_size, seq_length, _ = x.shape
        output, _ = self.forward(x, hidden)
        activation = torch.nn.functional.sigmoid(output).reshape(-1, 1)
        predicted = (activation > 0.5).float()
        return predicted.reshape(batch_size, seq_length, 1)
    
    
def train_binary_rnn(config, data, checkpoint_dir=None, verbose=False, return_model=False):
    """Train the RNN from within a raytune context. 
    
    Everything in this function needs to be reachable from the scope
    of a raytune process called from wherever you're calling it from.

    TODO: model checkpointing

    Args:
        config: raytune-compatible dictionary of hyperparameters

    Returns:
        model: the trained model
        hidden: the hidden state for the trained model

    """
    # batch_size, epoch and iteration
    BATCH_SIZE = 10
    epochs = config["epochs"]
    n_eval = config["n_eval"]

    # Data setup: WE have a fixed train/val split of 80/20
    n_train = int(len(data) * 0.8)
    data_loader = DataLoader(data[:n_train], batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(data[n_train:], batch_size=BATCH_SIZE, shuffle=False)
    seq_len = data.shape[1]

    print(config)
    # Model setup
    hidden = None # initial hidden state
    model = BinaryRNN(1, config["hidden_size"], 1, config["num_layers"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(epochs):  # You can adjust the number of epochs
        model.train()
        for X in data_loader:
            assert X.shape[0] == BATCH_SIZE # this is for if you didn't divide your dataset evenly, dummy
            inputs = X[:, :-1,:] # (batch_size, seq_length, 1)
            target = X[:, 1:,:]
            optimizer.zero_grad()
            output, hidden = model(inputs, hidden) # output is of shape (batch_size*seq_length, 1)
            hidden = hidden.data # this is to prevent backpropagation through the entire dataset
            # note, we are using BCEWithLogitsLoss, which contains a sigmoid activation already
            train_loss = criterion(output, target.reshape(-1, 1))
            train_loss.backward()
            optimizer.step()


        # Validation loss gets reported to raytune
        val_loss = 0.0
        val_steps = 0
        correct = 0
        model.eval()
        for _ in range(n_eval):
            X_val = next(iter(val_data_loader))
            inputs_val = X_val[:, :-1,:]
            target_val = X_val[:, 1:,:]
            target_flat = target_val.reshape(-1, 1)
            optimizer.zero_grad()
            output_val, _ = model(inputs_val, hidden) # output is of shape (batch_size*seq_length, 1)
            # Compute accuracy
            # Convert the sigmoid output to a binary prediction
            activation = torch.nn.functional.sigmoid(output_val).reshape(-1, 1)
            predicted = (activation > 0.5).float()
            Z = (predicted == target_flat).long()
            correct += Z.sum().item()

            # Compute loss
            loss = criterion(output_val,  target_flat)
            val_loss += loss.item()
            val_steps += 1
            if verbose and False:
                print("output", output_val.flatten())
                print(output_val.shape)
                print("act", activation.flatten())
                print("predicted", predicted.flatten())
                print("target", target_flat.flatten())
        # if val_steps > 4: # for mini-batching on the validation set
        #     break

        # Report all metrics in a single `report` call!
        metrics = {
                    "loss": (val_loss / val_steps),
                    "mean_accuracy": correct / (BATCH_SIZE * seq_len * val_steps),
                    "train_loss": train_loss.item()
                    }
        train.report(metrics)
        if verbose:
            print(metrics)

    if return_model:
        return (model, hidden)
