# Iterative MLSMOTE for Multi-Label Balancing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains a Python implementation of the MLSMOTE (Multi-Label Synthetic Minority Over-sampling Technique) algorithm, along with a new iterative balancing function, `balance_to_majority`.

This new function allows you to oversample all minority classes in a multi-label dataset to match the sample count of a specified majority class, providing more granular control over the balancing process.

## Key Features

* A vectorized, batch-based implementation of **MLSMOTE**.
* A new **`balance_to_majority`** function that iteratively runs MLSMOTE on each minority class until its sample count matches a target majority class.
* Handles cases with very few minority samples by duplicating existing ones when SMOTE cannot be applied.
* Includes detailed logging to monitor the balancing process.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/pukkahb/Iterative-MLSMOTE.git](https://github.com/pukkahb/Iterative-MLSMOTE.git)
    cd Iterative-MLSMOTE
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Example

Here is a complete example of how to use the `balance_to_majority` function to balance a sample multi-label dataset.

```python
import pandas as pd
from sklearn.datasets import make_classification
from mlsmote import balance_to_majority

# 1. Create a sample imbalanced multi-label dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=5,
    n_labels=2, # Average number of labels per instance
    random_state=42
)

X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
Y_train = pd.DataFrame(y, columns=[f'label_{i}' for i in range(5)])

# Make the dataset more imbalanced for a better demo
# Let's make label_0 the clear majority class
Y_train = Y_train.drop(Y_train[Y_train['label_1'] == 1].sample(frac=0.8).index)
Y_train = Y_train.drop(Y_train[Y_train['label_2'] == 1].sample(frac=0.9).index)
Y_train = Y_train.drop(Y_train[Y_train['label_3'] == 1].sample(frac=0.95).index)
Y_train = Y_train.drop(Y_train[Y_train['label_4'] == 1].sample(frac=0.85).index)
# Also drop the corresponding rows from X_train
X_train = X_train.loc[Y_train.index].reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)


print("--- Original Label Distribution ---")
print(Y_train.sum().sort_values(ascending=False))
print(f"\nOriginal dataset shape: X={X_train.shape}, Y={Y_train.shape}")


# 2. Balance the dataset to match the count of 'label_0'
X_balanced, Y_balanced = balance_to_majority(
    X_train,
    Y_train,
    majority_label='label_0',
    random_state=42
)

# 3. View the new, balanced distribution
print("\n--- Balanced Label Distribution ---")
print(Y_balanced.sum().sort_values(ascending=False))
print(f"\nBalanced dataset shape: X={X_balanced.shape}, Y={Y_balanced.shape}")
```

## Acknowledgements and References
This implementation was developed with the help of Gemini and is based on the concepts and code from the following excellent resources. Proper credit goes to their original authors.

  * **Nitesh Sukhwani's MLSMOTE GitHub Repository:** [https://github.com/niteshsukhwani/MLSMOTE/blob/master/mlsmote.py](https://github.com/niteshsukhwani/MLSMOTE/blob/master/mlsmote.py)
  * **Tolga Dincer's Kaggle Notebook on Upsampling Multi-Label Data:** [https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote](https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote)
