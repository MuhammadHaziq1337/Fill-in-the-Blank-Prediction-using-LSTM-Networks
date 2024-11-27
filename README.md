## Report on LSTM Model Development and Performance

### 1. Introduction

The objective of this project was to develop, train, and evaluate Long Short-Term Memory (LSTM) models for a fill-in-the-blank language modeling task. The task was based on the RACE dataset, which involves masked word prediction to simulate natural language understanding. This report outlines the data preprocessing pipeline, LSTM model architecture, training process, hyperparameter tuning experiments, and performance evaluation.

### 2. Data Preprocessing Details

#### 2.1 Dataset Description

The dataset used for this project was the RACE dataset, which comprises passages and related information. The preprocessing aimed to create a "fill-in-the-blank" dataset by masking specific words in sentences, transforming the data into a format compatible with LSTM input.

#### 2.2 Preprocessing Steps

1. **Sentence Splitting and Filtering**:
   - Articles were split into individual sentences.
   - Sentences with fewer than four words were excluded to ensure meaningful sequences for the task.
   
2. **Blank Creation**:
   - Words from the latter half of each sentence were removed, using a calculated index based on sentence length.
   - The removed word (termed the Missing Word) became the target label for prediction.
   - The remaining sentence was split into:
     - **Part A**: The portion of the sentence preceding the blank.
     - **Part B**: The reversed sequence following the blank.

3. **Tokenization**:
   - Part A and Part B were tokenized using the BERT tokenizer.
   - Sequences were padded or truncated to a fixed maximum length of 50 tokens.
   - Target labels (Missing Word) were encoded into integer representations using the same tokenizer.

### 3. LSTM Model Architecture

#### 3.1 Model Overview

Two distinct LSTM models were designed:

1. **Forward LSTM Model**: Processes sequences in the natural left-to-right order, learning patterns from Part A of the input.
2. **Backward LSTM Model**: Processes sequences in reverse order, learning patterns from Part B of the input.

Both models shared the same underlying architecture.

#### 3.2 Architecture Details

1. **Embedding Layer**:
   - A shared embedding layer with a vocabulary size of 30,522 and embedding dimension of 128.
   - This layer transforms tokenized input sequences into dense vector representations.

2. **LSTM Layer**:
   - A single-layer LSTM with 256 hidden units to process embedded sequences.
   - The hidden states capture sequential dependencies in the data.

3. **Fully Connected Layer**:
   - The LSTM output from the final time step is passed through a dense layer with 30,522 output neurons.
   - This corresponds to the vocabulary size and predicts the probability of each word in the vocabulary.

4. **Activation and Output**:
   - The final output uses a softmax activation function to generate probabilities for each word in the vocabulary.

### 4. Training Details

#### 4.1 Dataset Preparation

The preprocessed data was split into training (80%) and validation (20%) subsets using the `train_test_split` function from scikit-learn. The data was loaded into DataLoaders for batch processing, with a batch size of 32.

#### 4.2 Loss Function and Optimizer

- **Loss Function**: Cross-entropy loss was used to measure the difference between predicted probabilities and true labels.
- **Optimizer**: Adam optimizer with a learning rate of 0.001 was employed for gradient-based optimization.

#### 4.3 Training Process

Both the forward and backward LSTM models were trained separately for 5 epochs. Each epoch involved:

1. **Forward Pass**: Feeding tokenized sequences (Part A or Part B) into the respective model.
2. **Loss Calculation**: Comparing predictions to ground truth labels using cross-entropy loss.
3. **Backward Pass and Optimization**: Computing gradients and updating model weights.

### 5. Hyperparameter Tuning and Performance Comparison

#### 5.1 Overview

The models were trained and evaluated under three hyperparameter configurations. The goal was to assess how different settings of batch size, learning rate, embedding dimension, hidden units, and number of layers affected performance.

#### 5.2 Hyperparameter Configurations

- **Set 1**: {batch_size: 32, learning_rate: 0.001, embedding_dim: 128, hidden_dim: 256, num_layers: 1}
- **Set 2**: {batch_size: 64, learning_rate: 0.001, embedding_dim: 128, hidden_dim: 256, num_layers: 2}
- **Set 3**: {batch_size: 64, learning_rate: 0.0005, embedding_dim: 128, hidden_dim: 256, num_layers: 3}

#### 5.3 Results Summary

| Hyperparameter Set | Model       | Average Loss | Average Accuracy |
|--------------------|-------------|--------------|------------------|
| Set 1             | Forward LSTM | 5.5453       | 0.1569           |
|                    | Backward LSTM | 5.4619       | 0.1715           |
| Set 2             | Forward LSTM | 5.3842       | 0.1814           |
|                    | Backward LSTM | 5.3167       | 0.1910           |
| Set 3             | Forward LSTM | 5.2921       | 0.1936           |
|                    | Backward LSTM | 5.2478       | 0.1964           |

### 6. Conclusion

The LSTM models for fill-in-the-blank prediction showed promising results, with the backward LSTM achieving slightly better accuracy across all configurations. The hyperparameter tuning experiments revealed that adjusting the batch size and number of LSTM layers contributed to performance improvements. Further work may involve experimenting with more complex architectures, such as bidirectional LSTMs or Transformer-based models, to improve accuracy and handle longer sentences with more complex contexts.
