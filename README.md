# Zaremba_LSTM_GRU_Addition
A reimplementation of the "small" LSTM model as described in "Recurrent Neural Network Regularization", by Zaremba et al. (2015) (https://arxiv.org/abs/1409.2329). I add to this paper by applying the GRU model to the same data (with and without regularization).

<img width="774" alt="Screenshot 2024-09-25 at 11 48 34 PM" src="https://github.com/user-attachments/assets/aa0aaea9-6a81-408b-9ef5-50bae5e172f7">


## Project Overview
This project reimplements the "small" LSTM model as described in "Recurrent Neural Network Regularization" by Zaremba et al. (2015). In addition, we apply the GRU model to the same dataset (with and without regularization).

## Content Overview
This project includes a Jupyter Notebook file which creates a training and testing regime for evaluating different models on the Penn Tree Bank dataset. Specifically, we test the “small” LSTM model architecture with and without dropout regularization, as well as the GRU RNN with and without regularization.

## Dependencies
All dependencies are imported within the markdown file. Simply run the file to import the correct packages. Please use Python 3.10+.

## Data Upload
We used `wget` to retrieve the dataset easily whenever we run the notebook. After importing the dataset, we build a vocabulary – mapping each word in the data to a unique integer index. We convert each data portion (training, validation, test) into tensors of word indices before they are fed into the models.

## Key Functions and Classes

### `LSTMModel`
The `LSTMModel` class defines an LSTM Recurrent Neural Network. This includes functionality for initializing parameters, dropout (optional), embeddings, a decoder linear layer, a method to initialize the model's weights, and a forward function to allow the data's forward pass. Finally, `LSTMModel` includes a function that initializes the hidden and cell states for the LSTM.

### `GRUModel`
The `GRUModel` class defines a GRU Recurrent Neural Network. This includes functionality for initializing parameters, dropout (optional), embeddings, a decoder linear layer, a method to initialize the model's weights, and a forward function to allow the data's forward pass. Like the LSTMModel, this class initializes the hidden state, with only minor differences from `LSTMModel`.

### `evaluate`
The `evaluate` function evaluates the model's performance using `model.eval()` to set evaluation mode. It initializes the loss to zero and calculates the loss (using the provided `criterion` loss function) of the model on data. The function adds the total loss for each chunk the model sees and returns the average loss across all chunks of data.

### `train`
The `train` function trains an RNN model using a specified optimizer and loss function. Gradient clipping is employed to avoid exploding gradients. This function returns the average loss across all data chunks for the current iteration of training.

### `plot_perplexities`
This function plots the train and test word-level perplexities as a function of training epochs. It is called once within the `train_and_evaluate` function to plot these values at the end of model training.

### `train_and_evaluate`
This function combines training and evaluation for an RNN model (agnostic to GRU vs. LSTM) across multiple epochs. It allows specification of the learning rate, epoch to begin learning rate decay, final epoch, and more. Note that learning rate is decreased by a factor of 2 (halved) at each epoch once learning rate decay is activated. The function also saves model weights to a specified path, calls `plot_perplexities`, and returns the final perplexities on the train, validation, and test sets.

## How to Train and Test a Model
To run our cells for training and evaluating model performance, it is required that all importing and data processing cells are run, and that the `train` and `evaluate` functions are defined. Parameters must be defined in a cell to instantiate the models. Finally, `train_and_evaluate` should be defined for iterative training with adjustable epochs and output evaluation metrics.

### Example: LSTM with No Dropout

```python
DROPOUT = 0
LR = 3
DECAY_EPOCH = 7
MAX_EPOCH = 13

model = LSTMModel(VOCAB_SIZE, RNN_SIZE, LAYERS, DROPOUT).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_perplexity_lstm, val_perplexity_lstm, test_perplexity_lstm = train_and_evaluate(
   model, train_data, valid_data, test_data, optimizer, criterion,
   SEQ_LENGTH, MAX_EPOCH, DECAY_EPOCH, "lstm_model.pth", "LSTM (With Dropout = " + str(DROPOUT) + ") Perplexities vs Epoch"
)

print(f"Final Train Perplexity: {train_perplexity_lstm:.4f}")
print(f"Final Test Perplexity: {test_perplexity_lstm:.4f}")
print(f"Final Validation Perplexity: {val_perplexity_lstm:.4f}")
```

In this example, the model is instantiated with our parameters. We select an optimizer and criterion for a loss function. Finally, we run `train_and_evaluate` and collect the final perplexities for performance metrics. Setting `MAX_EPOCH` equal to `DECAY_EPOCH` nullifies learning rate decay.

Instructions for how to use saved model weights for inference are under the "OPTIONAL: Infer With Trained Model (example)" heading.

## Results

### LSTM (No Regularization)
**Parameters:**
- Dropout = 0.00  
- Learning Rate = 3  
- Learning Rate Decay Epoch = 7  
- Number of Epochs = 13  

**Final Results:**
- Train Perplexity: 65.6332  
- Test Perplexity: 117.4962  
- Validation Perplexity: 120.1541  

---

### LSTM with Dropout Regularization
**Parameters:**
- Dropout = 0.33  
- Learning Rate = 6  
- Learning Rate Decay Epoch = 20  
- Number of Epochs = 30  

**Final Results:**
- Train Perplexity: 71.3923  
- Test Perplexity: 93.1329  
- Validation Perplexity: 96.6946  

---

### GRU (No Regularization)
**Parameters:**
- Dropout = 0.00  
- Learning Rate = 1  
- Learning Rate Decay Epoch = 7  
- Number of Epochs = 13  

**Final Results:**
- Train Perplexity: 73.4147  
- Test Perplexity: 118.0094  
- Validation Perplexity: 122.1770  

---

### GRU with Dropout Regularization
**Parameters:**
- Dropout = 0.30  
- Learning Rate = 2  
- Learning Rate Decay Epoch = 7  
- Number of Epochs = 13  

**Final Results:**
- Train Perplexity: 60.9409  
- Test Perplexity: 98.5551  
- Validation Perplexity: 102.1842  


