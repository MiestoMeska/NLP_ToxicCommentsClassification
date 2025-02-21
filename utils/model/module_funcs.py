import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

def evaluate_model(pl_model, val_loader):
    """
    Evaluate the model using the validation loader, applying class-specific thresholds if available.
    It prints classification metrics for each class and overall performance.
    """
    pl_model.model.to('cuda')
    all_preds, all_targets = [], []

    pl_model.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['ids'].to('cuda')
            attention_mask = batch['mask'].to('cuda')
            targets = batch['targets'].long().to('cuda')

            outputs = pl_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if hasattr(pl_model, 'class_thresholds') and pl_model.class_thresholds is not None:
        binary_preds = torch.empty_like(all_preds)
        for i, threshold in enumerate(pl_model.class_thresholds):
            binary_preds[:, i] = (all_preds[:, i] > threshold).int()
        binary_preds = (all_preds > 0.5).int()

    plot_roc_auc_curve(all_preds, all_targets)
    print_classification_metrics(binary_preds, all_targets)

def plot_loss_curves(train_metrics, val_metrics):
    """
    Plot the training and validation loss, accuracy, and F1 score curves for both binary and multilabel tasks.
    """

    train_binary_loss = train_metrics['binary']['loss']
    val_binary_loss = val_metrics['binary']['loss']
    train_binary_f1 = train_metrics['binary']['f1']
    val_binary_f1 = val_metrics['binary']['f1']
    train_binary_acc = train_metrics['binary']['accuracy']
    val_binary_acc = val_metrics['binary']['accuracy']

    train_multilabel_loss = train_metrics['multilabel']['loss']
    val_multilabel_loss = val_metrics['multilabel']['loss']
    train_multilabel_f1 = train_metrics['multilabel']['f1']
    val_multilabel_f1 = val_metrics['multilabel']['f1']
    train_multilabel_acc = train_metrics['multilabel']['accuracy']
    val_multilabel_acc = val_metrics['multilabel']['accuracy']

    epochs = range(1, len(train_binary_loss) + 1)

    max_binary_loss = max(max(train_binary_loss), max(val_binary_loss))
    max_multilabel_loss = max(max(train_multilabel_loss), max(val_multilabel_loss))

    y_max_binary_loss = max_binary_loss + 0.1 * max_binary_loss
    y_max_multilabel_loss = max_multilabel_loss + 0.1 * max_multilabel_loss

    plt.figure(figsize=(12, 18))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_binary_loss, label='Training Loss')
    plt.plot(epochs, val_binary_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Binary Classification Loss')
    plt.ylim(0, y_max_binary_loss)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_multilabel_loss, label='Training Loss')
    plt.plot(epochs, val_multilabel_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multilabel Classification Loss')
    plt.ylim(0, y_max_multilabel_loss)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_binary_acc, label='Training Accuracy')
    plt.plot(epochs, val_binary_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Binary Classification Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_multilabel_acc, label='Training Accuracy')
    plt.plot(epochs, val_multilabel_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Multilabel Classification Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_binary_f1, label='Training F1 Score')
    plt.plot(epochs, val_binary_f1, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Binary Classification F1 Score')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(epochs, train_multilabel_f1, label='Training F1 Score')
    plt.plot(epochs, val_multilabel_f1, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Multilabel Classification F1 Score')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_roc_auc_curve(all_preds, all_targets, num_labels=6):
    """
    Plot ROC-AUC curves for each class.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(all_targets[:, i].cpu(), all_preds[:, i].cpu())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Label {i+1} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-AUC Curve')
    ax.legend(loc="lower right")
    plt.show()

def print_classification_metrics(binary_preds, all_targets):
    """
    Print full classification metrics for all labels.
    """
    print("\nFull Classification Metrics for All Labels (averaged):")
    print(classification_report(all_targets.cpu(), binary_preds.cpu(), zero_division=0))
