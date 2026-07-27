# Dynamic-Logistic-Regression-Ensemble-Model-

A machine learning implementation of a **Deep Dynamic Ensemble Model** - a differentiable, hierarchical classifier that combines multiple logistic regression models through soft routing decisions.

## Overview

This project implements a **Deep Dynamic Ensemble Logistic Regression** model from scratch using NumPy. Rather than using hard routing decisions, this implementation uses sigmoid functions (logistic functions) to create soft routing probabilities, allowing dynamic weighting of multiple logistic regression predictions at each level of the hierarchy. The result is a deep ensemble that learns how to optimally combine predictions across a hierarchical tree structure.

### Key Features

- **Dynamic Ensemble Routing**: Each internal node learns a logistic regression-based routing function to dynamically weight predictions from left and right child ensembles
- **Hierarchical Logistic Regression**: Leaf nodes implement logistic regression classifiers; internal nodes use logistic functions to combine child predictions
- **Numerical Gradient Descent**: Uses finite difference approximation for computing gradients (no backpropagation required)
- **Hierarchical Tree Structure**: Complete binary tree with configurable depth, enabling deep ensemble models
- **Training & Evaluation**: Includes loss computation, training loop, and accuracy metrics

## Model Architecture

### Node Structure
Each node in the tree contains:
- **Node ID**: Unique identifier using binary tree numbering
- **Parameters (θ)**: Weight vector of size equal to feature dimension (logistic regression weights)
- **Children**: References to left and right child ensembles (None for leaf nodes)

### Forward Pass: Hierarchical Logistic Regression

**For Leaf Nodes (Base Logistic Regression):**
```
prediction = sigmoid(X @ θ)     # Standard logistic regression
```

**For Internal Nodes (Dynamic Ensemble Weighting):**
```
routing_prob = sigmoid(X @ θ)                              # Logistic routing function
left_ensemble_pred = forward_pass(left_child, X)           # Left child ensemble prediction
right_ensemble_pred = forward_pass(right_child, X)         # Right child ensemble prediction
final_prediction = routing_prob * left_ensemble_pred + (1 - routing_prob) * right_ensemble_pred
```

This creates a **deep ensemble** where each internal node learns to dynamically weight two sub-ensembles using logistic functions.

### Loss Function
Binary cross-entropy loss:
```
Loss = -mean(y * log(predictions) + (1-y) * log(1-predictions))
```

## Training

The model is trained using **numerical gradient descent**:

1. For each epoch, iterate through all nodes
2. For each weight parameter, compute finite difference gradient:
   ```
   grad[i] = (loss(θ[i]+ε) - loss(θ[i]-ε)) / (2ε)
   ```
3. Update weights using gradient descent:
   ```
   θ[i] -= learning_rate * grad[i]
   ```

### Hyperparameters
- **Tree Depth**: Controls number of layers (depth=2 → 7 nodes)
- **Epochs**: Number of training iterations (default: 500)
- **Learning Rate**: Gradient descent step size (default: 0.01)
- **Numerical Gradient Epsilon**: Step size for finite differences (1e-4)

## Results

- **Train Accuracy**: 71.50%
- **Test Accuracy**: 70.63%

## Usage Example

```python
# Build a tree of depth 2 (7 nodes total)
root = build_tree_recursive(depth=2)

# Flatten tree into list of all nodes
all_nodes = []
get_all_nodes(root, all_nodes)

# Train the model
loss_history = train_tree(
    root, 
    all_nodes, 
    X_train_scaled, 
    y_train,
    epochs=500,
    learning_rate=0.01
)

# Make predictions
train_predictions = forward_pass(root, X_train_scaled)
test_predictions = forward_pass(root, X_test_scaled)

# Evaluate
train_acc = accuracy(train_predictions, y_train)
test_acc = accuracy(test_predictions, y_test)
```

## Dependencies

- NumPy
- Matplotlib (for visualization)
- Scikit-learn (for data preprocessing and metrics)

## Implementation Details

### Activation Functions
- **Sigmoid**: Used for both routing probabilities and leaf predictions
  - Includes overflow protection (clipping z to [-500, 500])

### Numerical Stability
- Input features are scaled to prevent numerical issues
- Sigmoid function includes overflow protection
- Epsilon value of 1e-4 chosen for stable gradient computation

### Tree Construction
The `build_tree_recursive()` function creates a complete binary tree using binary tree numbering:
- Node 1: Root
- Node 2i: Left child of node i
- Node 2i+1: Right child of node i

## Why Deep Dynamic Ensemble Logistic Regression?

Deep Dynamic Ensemble Models offer advantages over traditional classifiers:

1. **Differentiability**: Fully differentiable hierarchical structure allows end-to-end learning
2. **Dynamic Ensembling**: Learns optimal weighting of sub-ensembles at each level using logistic functions
3. **Hierarchical Composition**: Combines simple logistic regressors into a deep, powerful ensemble
4. **Smooth Decision Boundaries**: Logistic soft routing and weighted ensemble averaging creates smooth, non-linear decision boundaries
5. **Interpretability**: Each node represents a learned logistic regression with clear probabilistic interpretation
6. **Ensemble Diversity**: Multiple logistic regression models at each level capture different decision boundaries

## Limitations

- Numerical gradient computation is slower than backpropagation
- Finite difference gradients may be less accurate than analytical gradients
- Not suitable for very deep trees due to computational complexity

## Future Improvements

- Implement automatic differentiation (backpropagation) for faster training
- Add regularization to prevent overfitting
- Experiment with different activation functions (ReLU, tanh)
- Implement batch processing for better scalability
- Add cross-validation for hyperparameter tuning

## Author Notes

This implementation demonstrates the core concepts of differentiable tree models from first principles, using only NumPy for gradient computation. It serves as an educational tool for understanding how neural network-style learning can be applied to tree structures.
