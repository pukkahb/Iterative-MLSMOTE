import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Finds underrepresented target labels based on occurrence frequency.
    (This function is kept for its original use but is not used by the new balancing function).
    """
    # ... (previous code for get_tail_label remains unchanged) ...
    if df.empty: return []
    irlbl = df.sum(axis=0)
    if irlbl.empty: return []
    if irlbl.nunique() > 1:
        lower_bound, upper_bound = irlbl.quantile(ql[0]), irlbl.quantile(ql[1])
        irlbl_filtered = irlbl[(irlbl >= lower_bound) & (irlbl <= upper_bound)]
    else:
        irlbl_filtered = irlbl
    if irlbl_filtered.empty: return []
    max_freq = irlbl_filtered.max()
    if max_freq == 0: return []
    imbalance_ratios = max_freq / irlbl_filtered
    valid_ratios = imbalance_ratios[irlbl_filtered > 0]
    if valid_ratios.empty: return []
    threshold_irlbl = valid_ratios.median()
    tail_label = valid_ratios[valid_ratios > threshold_irlbl].index.tolist()
    logging.info(f"Identified tail labels: {tail_label} using ql={ql} and median IR threshold={threshold_irlbl:.2f}")
    return tail_label

def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    Identifies samples that contain at least one of the "tail labels".
    (This function is also kept for its original use).
    """
    # ... (previous code for get_minority_samples remains unchanged) ...
    tail_labels = get_tail_label(y, ql=ql)
    if not tail_labels:
        return pd.DataFrame(columns=X.columns), pd.DataFrame(columns=y.columns)
    existing_tail_labels = [label for label in tail_labels if label in y.columns]
    if not existing_tail_labels:
        return pd.DataFrame(columns=X.columns), pd.DataFrame(columns=y.columns)
    has_tail_label = y[existing_tail_labels].any(axis=1)
    minority_indices = y[has_tail_label].index
    if minority_indices.empty:
        return pd.DataFrame(columns=X.columns), pd.DataFrame(columns=y.columns)
    X_sub = X.loc[minority_indices].reset_index(drop=True)
    y_sub = y.loc[minority_indices].reset_index(drop=True)
    return X_sub, y_sub

def MLSMOTE_batch_vectorized(
    X_minority: pd.DataFrame,
    y_minority: pd.DataFrame,
    n_samples_to_generate: int,
    n_neighbors_for_smote: int = 5,
    label_strategy: str = 'union',
    return_combined: bool = False,
    random_state=None
):
    """
    Faster, vectorized implementation of MLSMOTE with configurable label generation.
    (This function is kept as it is the core vectorized implementation).
    """
    # ... (previous code for MLSMOTE_batch_vectorized remains unchanged) ...
    if random_state is not None: np.random.seed(random_state)
    if X_minority.empty or y_minority.empty or n_samples_to_generate <= 0:
        return (X_minority.copy(), y_minority.copy()) if return_combined else (pd.DataFrame(columns=X_minority.columns), pd.DataFrame(columns=y_minority.columns))
    if X_minority.shape[0] < 2:
        synthetic_X = pd.concat([X_minority] * n_samples_to_generate, ignore_index=True)
        synthetic_y = pd.concat([y_minority] * n_samples_to_generate, ignore_index=True)
        return (pd.concat([X_minority, synthetic_X], ignore_index=True), pd.concat([y_minority, synthetic_y], ignore_index=True)) if return_combined else (synthetic_X, synthetic_y)
    
    k_neighbors = min(n_neighbors_for_smote + 1, X_minority.shape[0])
    nn = NearestNeighbors(n_neighbors=k_neighbors).fit(X_minority.values)
    _, indices = nn.kneighbors(X_minority.values)

    ref_indices = np.random.choice(X_minority.shape[0], size=n_samples_to_generate)
    neighbor_choices = [np.random.choice(indices[i][1:]) for i in ref_indices]
    X_ref = X_minority.iloc[ref_indices].values
    X_neighbor = X_minority.iloc[neighbor_choices].values
    ratios = np.random.rand(n_samples_to_generate, 1)
    synthetic_features = X_ref + ratios * (X_ref - X_neighbor)
    
    y_neighbors_all = y_minority.values[indices[ref_indices]]
    if label_strategy == 'union': synthetic_labels = (y_neighbors_all.sum(axis=1) > 0).astype(int)
    elif label_strategy == 'majority': synthetic_labels = (y_neighbors_all.sum(axis=1) > (k_neighbors // 2)).astype(int)
    else: raise ValueError("Unknown label_strategy")

    new_X_df = pd.DataFrame(synthetic_features, columns=X_minority.columns)
    new_y_df = pd.DataFrame(synthetic_labels, columns=y_minority.columns)
    
    if return_combined:
        return pd.concat([X_minority.reset_index(drop=True), new_X_df], ignore_index=True), pd.concat([y_minority.reset_index(drop=True), new_y_df], ignore_index=True)
    else:
        return new_X_df, new_y_df

# --- NEW FUNCTION ---
def balance_to_majority(X_train: pd.DataFrame, Y_train: pd.DataFrame, majority_label: str, random_state=None):
    """
    Oversamples minority classes iteratively to match the count of a specified majority class.

    Args:
        X_train (pd.DataFrame): The original training features.
        Y_train (pd.DataFrame): The original training labels.
        majority_label (str): The name of the label to use as the target count for oversampling.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame, pd.DataFrame: The balanced feature and label DataFrames.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting balancing process. Target label for count: '{majority_label}'.")

    label_counts = Y_train.sum()
    if majority_label not in label_counts.index:
        logger.error(f"Majority label '{majority_label}' not found in Y_train columns. Aborting.")
        return X_train, Y_train

    target_count = label_counts[majority_label]
    logger.info(f"Target sample count for each label is: {target_count}")

    X_balanced = X_train.copy()
    Y_balanced = Y_train.copy()

    # Sort labels by count to oversample smallest classes first, which is more stable
    sorted_labels = label_counts.sort_values().index

    for label in sorted_labels:
        # Don't oversample the majority class itself
        if label == majority_label:
            continue

        # Recalculate count in each iteration, as it changes
        current_count = Y_balanced[label].sum()

        if current_count >= target_count:
            logger.info(f"Label '{label}' already has {current_count} samples (>= target {target_count}). Skipping.")
            continue

        n_to_generate = target_count - current_count
        logger.info(f"Balancing '{label}': Need to generate {n_to_generate} samples (current: {current_count}).")

        # Get all samples containing the current minority label from the *currently balanced* dataset
        minority_indices = Y_balanced[Y_balanced[label] == 1].index
        
        if len(minority_indices) < 2:
            logger.warning(f"Cannot apply SMOTE for label '{label}': only {len(minority_indices)} sample(s) found. Duplicating samples instead.")
            if len(minority_indices) == 1:
                # Find the single sample to duplicate
                X_to_duplicate = X_balanced.loc[minority_indices]
                Y_to_duplicate = Y_balanced.loc[minority_indices]
                # Add the duplicated samples
                X_balanced = pd.concat([X_balanced] + [X_to_duplicate] * n_to_generate, ignore_index=True)
                Y_balanced = pd.concat([Y_balanced] + [Y_to_duplicate] * n_to_generate, ignore_index=True)
            continue # Move to next label

        X_minority_subset = X_balanced.loc[minority_indices]
        Y_minority_subset = Y_balanced.loc[minority_indices]

        # Generate ONLY the new synthetic samples
        X_synthetic, y_synthetic = MLSMOTE_batch_vectorized(
            X_minority_subset,
            Y_minority_subset,
            n_samples_to_generate=n_to_generate,
            return_combined=False,
            random_state=random_state
        )

        # Add the new samples to the balanced dataset
        X_balanced = pd.concat([X_balanced, X_synthetic], ignore_index=True)
        Y_balanced = pd.concat([Y_balanced, y_synthetic], ignore_index=True)
        logger.info(f"Finished balancing for '{label}'. New total samples: {len(Y_balanced)}. New count for '{label}': {Y_balanced[label].sum()}")

    logger.info("Finished iterative balancing for all minority labels.")
    logger.info(f"Final balanced dataset shape: X={X_balanced.shape}, Y={Y_balanced.shape}")
    logger.info(f"Final label distribution:\n{Y_balanced.sum().to_string()}")

    return X_balanced, Y_balanced
