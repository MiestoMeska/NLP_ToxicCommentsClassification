import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from tqdm import tqdm

def evaluate_binary_and_multilabel(binary_preds, binary_targets, multilabel_preds, multilabel_targets):
    """
    Function to handle both binary and multilabel evaluation.
    """
    print("\nEvaluating Binary Classification (Toxic vs Non-Toxic)")
    precision, recall, f1, _ = precision_recall_fscore_support(binary_targets, binary_preds, average='binary')
    conf_matrix = confusion_matrix(binary_targets, binary_preds)
    print(f"Binary Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    print("\nEvaluating Multilabel Classification (Toxicity Classes)")
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for i, label_name in enumerate(label_names):
        precision, recall, f1, _ = precision_recall_fscore_support(multilabel_targets[:, i], multilabel_preds[:, i], average='binary', zero_division=0)
        conf_matrix = confusion_matrix(multilabel_targets[:, i], multilabel_preds[:, i])
        print(f"\nConfusion Matrix for {label_name}:")
        print(conf_matrix)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

def predict_probabilities(model, dataloader, device='cuda'):
    all_binary_probs = []
    all_multilabel_probs = []
    all_binary_targets = []
    all_multilabel_targets = []

    model.eval()
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference for probabilities"):
        input_ids = batch['ids'].to(device)
        attention_mask = batch['mask'].to(device)
        binary_target = batch['binary_targets'].float().to(device)
        multilabel_target = batch['multi_targets'].float().to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, (list, tuple)):
                binary_output = outputs[0]
                multilabel_output = outputs[1]
                if isinstance(multilabel_output, list):
                    multilabel_output = torch.cat(multilabel_output, dim=1)
            else:
                raise ValueError("Expected model to return a list/tuple of outputs, but got a single tensor.")

            binary_probs = torch.sigmoid(binary_output).cpu().numpy()
            multilabel_probs = torch.sigmoid(multilabel_output).cpu().numpy()

        all_binary_probs.append(binary_probs)
        all_multilabel_probs.append(multilabel_probs)
        all_binary_targets.append(binary_target.cpu().numpy())
        all_multilabel_targets.append(multilabel_target.cpu().numpy())

    all_binary_probs = np.concatenate(all_binary_probs).flatten()
    all_multilabel_probs = np.concatenate(all_multilabel_probs)
    all_binary_targets = np.concatenate(all_binary_targets).flatten()
    all_multilabel_targets = np.concatenate(all_multilabel_targets)

    return all_binary_probs, all_multilabel_probs, all_binary_targets, all_multilabel_targets

def evaluate_model_with_thresholds(binary_probs, multilabel_probs, binary_targets, multilabel_targets, binary_threshold, multilabel_thresholds):
    binary_preds = (binary_probs >= binary_threshold).astype(int)
    multilabel_preds = np.zeros_like(multilabel_probs)
    for i in range(multilabel_preds.shape[1]):
        multilabel_preds[:, i] = (multilabel_probs[:, i] >= multilabel_thresholds[i]).astype(int)

    evaluate_binary_and_multilabel(binary_preds, binary_targets, multilabel_preds, multilabel_targets)

def threshold_optimization(binary_probs, multilabel_probs, binary_targets, multilabel_targets):
    optimal_binary_threshold = 0.5
    optimal_multilabel_thresholds = []

    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1 = 0
    for threshold in thresholds:
        binary_preds = (binary_probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(binary_targets, binary_preds, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            optimal_binary_threshold = threshold

    for i in range(multilabel_probs.shape[1]):
        best_f1 = 0
        optimal_threshold = 0.5
        for threshold in thresholds:
            multilabel_preds = (multilabel_probs[:, i] >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(multilabel_targets[:, i], multilabel_preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
        optimal_multilabel_thresholds.append(optimal_threshold)

    return optimal_binary_threshold, optimal_multilabel_thresholds

def plot_roc_curve(binary_probs, binary_targets, multilabel_probs, multilabel_targets):
    label_names = ['Binary', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    plt.figure(figsize=(8, 6))
    for i, (probs, targets) in enumerate([(binary_probs, binary_targets)] + list(zip(multilabel_probs.T, multilabel_targets.T))):
        fpr, tpr, _ = roc_curve(targets, probs)
        auc_score = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f'{label_names[i]} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Binary and Multilabel Classification')
    plt.legend(loc='lower right')
    plt.show()

def plot_threshold_vs_metrics(binary_probs, binary_targets, multilabel_probs, multilabel_targets):
    thresholds = np.arange(0.0, 1.01, 0.01)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    binary_precisions, binary_recalls, binary_f1_scores = [], [], []
    for threshold in thresholds:
        binary_preds = (binary_probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(binary_targets, binary_preds, average='binary', zero_division=0)
        binary_precisions.append(precision)
        binary_recalls.append(recall)
        binary_f1_scores.append(f1)

    fig, axs_binary = plt.subplots(1, 1, figsize=(5, 5))
    axs_binary.plot(thresholds, binary_precisions, label='Binary Precision')
    axs_binary.plot(thresholds, binary_recalls, label='Binary Recall')
    axs_binary.plot(thresholds, binary_f1_scores, label='Binary F1-Score')
    axs_binary.set_title('Binary Threshold vs Metrics')
    axs_binary.legend()

    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i in range(multilabel_probs.shape[1]):
        precisions, recalls, f1_scores = [], [], []
        for threshold in thresholds:
            preds = (multilabel_probs[:, i] >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(multilabel_targets[:, i], preds, average='binary', zero_division=0)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        row, col = divmod(i, 3)
        axs[row, col].plot(thresholds, precisions, label=f'{label_names[i]} Precision')
        axs[row, col].plot(thresholds, recalls, label=f'{label_names[i]} Recall')
        axs[row, col].plot(thresholds, f1_scores, label=f'{label_names[i]} F1-Score')
        axs[row, col].set_title(f'{label_names[i]} Threshold vs Metrics')
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()


def run_evaluation_and_threshold_optimization(model, dataloader, device='cuda'):
    binary_probs, multilabel_probs, binary_targets, multilabel_targets = predict_probabilities(model, dataloader, device=device)

    print("Initial Evaluation with Default Thresholds (0.5 for all):")
    evaluate_model_with_thresholds(binary_probs, multilabel_probs, binary_targets, multilabel_targets, binary_threshold=0.5, multilabel_thresholds=[0.5] * 6)

    print("\nOptimizing Thresholds...")
    optimal_binary_threshold, optimal_multilabel_thresholds = threshold_optimization(binary_probs, multilabel_probs, binary_targets, multilabel_targets)

    print(f"\nOptimal Binary Threshold: {optimal_binary_threshold}")
    for i, threshold in enumerate(optimal_multilabel_thresholds):
        print(f"Optimal Threshold for label {i+1}: {threshold}")

    print("\nPlotting Threshold vs Metrics...")
    plot_threshold_vs_metrics(binary_probs, binary_targets, multilabel_probs, multilabel_targets)

    print("\nPlotting ROC Curves...")
    plot_roc_curve(binary_probs, binary_targets, multilabel_probs, multilabel_targets)

    print("\nSetting optimized thresholds in the model...")
    model.binary_threshold = optimal_binary_threshold
    model.multilabel_thresholds = optimal_multilabel_thresholds

    print("\nRe-running predictions with optimized thresholds...")
    binary_probs, multilabel_probs, binary_targets, multilabel_targets = predict_probabilities(model, dataloader, device=device)

    print("\nRe-Evaluation with Optimized Thresholds:")
    evaluate_model_with_thresholds(
        binary_probs, 
        multilabel_probs, 
        binary_targets, 
        multilabel_targets, 
        binary_threshold=optimal_binary_threshold, 
        multilabel_thresholds=optimal_multilabel_thresholds
    )