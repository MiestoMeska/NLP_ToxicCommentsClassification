import torch

def compare_datasets(train_df, val_df, test_df, label_columns):
    """
    Compare the distribution of toxic and non-toxic comments in train, validation, and test datasets.
    Also compare the distribution of each toxic class.
    
    Parameters:
    - train_df, val_df, test_df (pd.DataFrame): DataFrames for the train, validation, and test splits.
    - label_columns (list): List of toxic class labels (e.g., ['toxic', 'severe_toxic', ...])
    
    Returns:
    - None: Prints the distributions for each dataset.
    """

    print("\n--- Binary Toxic vs. Non-Toxic Distribution ---")
    for df_name, df in zip(['Training', 'Validation', 'Test'], [train_df, val_df, test_df]):
        print(f"\n{df_name} Set:")
        print(df['is_toxic'].value_counts(normalize=True) * 100)
    
    print("\n--- Distribution of Each Toxic Class ---")
    for df_name, df in zip(['Training', 'Validation', 'Test'], [train_df, val_df, test_df]):
        print(f"\n{df_name} Set:")
        class_distribution = df[label_columns].mean() * 100
        print(class_distribution)

def compute_class_weights(train_df, label_columns):
    """
    Computes class weights for multi-label classification.

    Args:
    - train_df (pd.DataFrame): The training dataframe containing the labels.
    - label_columns (list of str): List of label columns to compute class weights for.

    Returns:
    - class_weights_tensor (torch.Tensor): A tensor of normalized class weights.
    """
    class_frequencies = train_df[label_columns].mean()
    class_weights = 1.0 / class_frequencies
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)
    
    print("Class Weights:", class_weights_tensor)
    return class_weights_tensor

def compute_toxic_nontoxic_weights(train_df, label_column):
    """
    Computes class weights for binary classification (toxic vs non-toxic).

    Args:
    - train_df (pd.DataFrame): The training dataframe containing the label.
    - label_column (str): The column name containing the toxic vs non-toxic labels.

    Returns:
    - class_weights_tensor (torch.Tensor): A tensor of class weights for toxic (1) and non-toxic (0).
    """
    total_samples = len(train_df)
    toxic_count = train_df[label_column].sum()
    non_toxic_count = total_samples - toxic_count
    toxic_frequency = toxic_count / total_samples
    non_toxic_frequency = non_toxic_count / total_samples
    class_weights = [1.0 / non_toxic_frequency, 1.0 / toxic_frequency]
    class_weights = [w / sum(class_weights) for w in class_weights]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("Class Weights:", class_weights_tensor)
    return class_weights_tensor