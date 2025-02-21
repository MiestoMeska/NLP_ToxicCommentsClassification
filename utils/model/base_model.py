import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from utils.model.module_funcs import evaluate_model, plot_loss_curves, plot_roc_auc_curve, print_classification_metrics

class Base_pl_Model(pl.LightningModule):
    """
    A PyTorch Lightning model for multi-label classification using DistilBERT.
    
    This model supports class-specific and toxic/non-toxic sample weighting during loss calculation. It tracks 
    training and validation loss and accuracy at the batch and epoch levels and automatically saves the best model 
    based on validation loss.
    
    Attributes:
    - model: The underlying DistilBERT model for encoding inputs.
    - learning_rate: The learning rate for the optimizer.
    - num_labels: Number of output labels.
    - model_name: Name of the model, used for saving checkpoints.
    - class_weights: Tensor of class weights for handling class imbalance.
    - toxic_weights: Weights for toxic vs non-toxic samples to further address class imbalance.
    - best_val_loss: The best validation loss observed during training.
    - train_batch_losses: List of training batch losses for averaging at the end of the epoch.
    - val_batch_losses: List of validation batch losses for averaging at the end of the epoch.
    - train_epoch_losses: List of average training losses across epochs.
    - val_epoch_losses: List of average validation losses across epochs.
    """
    
    def __init__(self, model, num_labels=6, learning_rate=1e-5, model_name="model", class_weights=None):
        """
        Initializes the model with the given hyperparameters and configurations.
        
        Args:
        - model: The DistilBERT model for encoding input sequences.
        - num_labels: Number of output labels (default is 6).
        - learning_rate: Learning rate for the optimizer (default is 1e-5).
        - model_name: Name of the model, used for saving checkpoints (default is "model").
        - class_weights: Tensor of weights for each label (for addressing class imbalance).
        - toxic_weights: Tuple with two values: weight for toxic and non-toxic samples.
        """
        super(Base_pl_Model, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.model_name = model_name
        self.class_weights = class_weights
        # Metrics
        self.train_macro_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        self.val_macro_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        self.test_macro_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        self.train_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)
        self.val_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)
        self.test_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)
        self.val_auroc = MultilabelAUROC(num_labels=self.num_labels)
        self.test_auroc = MultilabelAUROC(num_labels=self.num_labels)

        self.best_val_loss = float('inf')

        self.train_batch_losses = []
        self.val_batch_losses = []
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []

    def forward(self, input_ids, attention_mask):

        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        targets = batch['targets'].float()

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, targets)

        preds = torch.sigmoid(outputs)

        int_targets = targets.long()

        acc = self.train_accuracy(preds, int_targets)
        macro_f1 = self.train_macro_f1(preds, int_targets)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_macro_f1', macro_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.train_batch_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        targets = batch['targets'].float()

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, targets)

        preds = torch.sigmoid(outputs)
        binary_preds = (preds > 0.5).int()

        int_targets = targets.long()

        acc = self.val_accuracy(binary_preds, int_targets)
        auroc = self.val_auroc(preds, int_targets)
        macro_f1 = self.val_macro_f1(binary_preds, int_targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_macro_f1', macro_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.val_batch_losses.append(loss.item())

        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        targets = batch['targets'].float()

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs, targets)

        preds = torch.sigmoid(outputs)
        binary_preds = (preds > 0.5).int()

        int_targets = targets.long()

        acc = self.test_accuracy(binary_preds, int_targets)
        auroc = self.test_auroc(preds, int_targets)
        macro_f1 = self.test_macro_f1(binary_preds, int_targets)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_auroc', auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_macro_f1', macro_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        avg_train_loss = sum(self.train_batch_losses) / len(self.train_batch_losses)
        self.train_epoch_losses.append(avg_train_loss)
        self.train_batch_losses.clear()

        train_f1_score = self.train_macro_f1.compute()
        self.train_f1_scores.append(train_f1_score)
        self.train_macro_f1.reset()

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_batch_losses) / len(self.val_batch_losses)
        self.val_epoch_losses.append(avg_val_loss)
        self.val_batch_losses.clear()

        val_f1_score = self.val_macro_f1.compute()
        self.val_f1_scores.append(val_f1_score)
        self.val_macro_f1.reset()

    def on_fit_end(self):
        plot_loss_curves(self.train_epoch_losses, self.val_epoch_losses, self.train_f1_scores, self.val_f1_scores)

    def evaluate_model(self, val_loader):
        evaluate_model(self, val_loader)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_macro_f1'
            }
        }