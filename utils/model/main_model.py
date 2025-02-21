import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC, BinaryF1Score, BinaryAccuracy, BinaryAUROC
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from utils.model.module_funcs import evaluate_model, plot_loss_curves

class Main_Model(pl.LightningModule):
    """
    PyTorch Lightning module for binary and multi-label classification using the Extended_Model.
    The model predicts both toxic vs. non-toxic (binary classification) and multiple toxicity categories (multi-label classification).
    """
    
    def __init__(self, model, num_labels=6, learning_rate=1e-5, class_weights=None, binary_weights=None):
        """
        Initializes the model with the given hyperparameters and configurations.

        Args:
        - model: The Extended_Model for binary and multi-label classification.
        - num_labels: Number of output labels for multi-label classification (default is 6).
        - learning_rate: Learning rate for the optimizer (default is 1e-5).
        - class_weights: Tensor of weights for each multi-label class (for addressing class imbalance).
        """
        super(Main_Model, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.binary_weights = binary_weights
        self.training_phase = 'default' 

        self.binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.binary_weights)
        self.multilabel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        self.train_binary_f1 = BinaryF1Score()
        self.val_binary_f1 = BinaryF1Score()
        self.test_binary_f1 = BinaryF1Score()

        self.train_binary_accuracy = BinaryAccuracy()
        self.val_binary_accuracy = BinaryAccuracy()
        self.test_binary_accuracy = BinaryAccuracy()

        self.val_binary_auroc = BinaryAUROC()
        self.test_binary_auroc = BinaryAUROC()


        self.train_multilabel_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        self.val_multilabel_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        self.test_multilabel_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')

        self.train_multilabel_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)
        self.val_multilabel_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)
        self.test_multilabel_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)

        self.val_multilabel_auroc = MultilabelAUROC(num_labels=self.num_labels)
        self.test_multilabel_auroc = MultilabelAUROC(num_labels=self.num_labels)

        self.train_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                          'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}
        self.val_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                        'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def compute_loss(self, binary_output, multilabel_output, binary_target, multilabel_target):
        if self.training_phase == 'binary':

            return self.binary_loss_fn(binary_output, binary_target)
        elif self.training_phase == 'multilabel':

            return self.multilabel_loss_fn(multilabel_output, multilabel_target)
        else:

            binary_loss = self.binary_loss_fn(binary_output, binary_target)
            multilabel_loss = self.multilabel_loss_fn(multilabel_output, multilabel_target)
            return binary_loss + multilabel_loss
        
    def on_train_start(self):
        """
        Clears the metric logs for both training and validation before the first epoch starts
        (i.e., after the validation sanity check).
        """

        self.train_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}
        
        self.val_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}

    def training_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        binary_target = batch['binary_targets'].float()
        multilabel_target = batch['multi_targets'].float()

        binary_output, multilabel_output = self(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.compute_loss(binary_output, multilabel_output, binary_target, multilabel_target)

        if self.training_phase == 'binary':
            binary_preds = torch.sigmoid(binary_output) > 0.5
            binary_acc = self.train_binary_accuracy(binary_preds, binary_target.int())
            binary_f1 = self.train_binary_f1(binary_preds, binary_target.int())

            self.log('train_binary_acc', binary_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_binary_f1', binary_f1, on_step=False, on_epoch=True, prog_bar=True)

        elif self.training_phase == 'multilabel':
            multilabel_preds = torch.sigmoid(multilabel_output) > 0.5
            multilabel_acc = self.train_multilabel_accuracy(multilabel_preds, multilabel_target.int())
            multilabel_f1 = self.train_multilabel_f1(multilabel_preds, multilabel_target.int())

            self.log('train_multilabel_acc', multilabel_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_multilabel_f1', multilabel_f1, on_step=False, on_epoch=True, prog_bar=True)

        else:
            binary_preds = torch.sigmoid(binary_output) > 0.5
            binary_acc = self.train_binary_accuracy(binary_preds, binary_target.int())
            binary_f1 = self.train_binary_f1(binary_preds, binary_target.int())

            multilabel_preds = torch.sigmoid(multilabel_output) > 0.5
            multilabel_acc = self.train_multilabel_accuracy(multilabel_preds, multilabel_target.int())
            multilabel_f1 = self.train_multilabel_f1(multilabel_preds, multilabel_target.int())

            self.log('train_binary_acc', binary_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_binary_f1', binary_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_multilabel_acc', multilabel_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_multilabel_f1', multilabel_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        binary_target = batch['binary_targets'].float()
        multilabel_target = batch['multi_targets'].float()

        with torch.no_grad():
            binary_output, multilabel_output = self(input_ids=input_ids, attention_mask=attention_mask)

            loss = self.compute_loss(binary_output, multilabel_output, binary_target, multilabel_target)

            if self.training_phase == 'binary':
                binary_preds = torch.sigmoid(binary_output) > 0.5
                binary_acc = self.val_binary_accuracy(binary_preds, binary_target.int())
                binary_f1 = self.val_binary_f1(binary_preds, binary_target.int())
                binary_auroc = self.val_binary_auroc(binary_output, binary_target.int())

                self.log('val_binary_acc', binary_acc, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_binary_f1', binary_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_binary_auroc', binary_auroc, on_step=False, on_epoch=True, prog_bar=True)

            elif self.training_phase == 'multilabel':
                multilabel_preds = torch.sigmoid(multilabel_output) > 0.5
                multilabel_acc = self.val_multilabel_accuracy(multilabel_preds, multilabel_target.int())
                multilabel_f1 = self.val_multilabel_f1(multilabel_preds, multilabel_target.int())
                multilabel_auroc = self.val_multilabel_auroc(multilabel_output, multilabel_target.int())

                self.log('val_multilabel_acc', multilabel_acc, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_multilabel_f1', multilabel_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_multilabel_auroc', multilabel_auroc, on_step=False, on_epoch=True, prog_bar=True)

            else:
                binary_preds = torch.sigmoid(binary_output) > 0.5
                binary_acc = self.val_binary_accuracy(binary_preds, binary_target.int())
                binary_f1 = self.val_binary_f1(binary_preds, binary_target.int())
                binary_auroc = self.val_binary_auroc(binary_output, binary_target.int())

                multilabel_preds = torch.sigmoid(multilabel_output) > 0.5
                multilabel_acc = self.val_multilabel_accuracy(multilabel_preds, multilabel_target.int())
                multilabel_f1 = self.val_multilabel_f1(multilabel_preds, multilabel_target.int())
                multilabel_auroc = self.val_multilabel_auroc(multilabel_output, multilabel_target.int())

                self.log('val_binary_acc', binary_acc, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_binary_f1', binary_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_binary_auroc', binary_auroc, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_multilabel_acc', multilabel_acc, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_multilabel_f1', multilabel_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_multilabel_auroc', multilabel_auroc, on_step=False, on_epoch=True, prog_bar=True)

            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        binary_acc = self.trainer.callback_metrics.get('train_binary_acc')
        binary_f1 = self.trainer.callback_metrics.get('train_binary_f1')
        multilabel_acc = self.trainer.callback_metrics.get('train_multilabel_acc')
        multilabel_f1 = self.trainer.callback_metrics.get('train_multilabel_f1')
        train_loss = self.trainer.callback_metrics.get('train_loss')

        if self.training_phase == 'binary':
            self.train_metrics['binary']['loss'].append(train_loss.item())
            self.train_metrics['binary']['f1'].append(binary_f1.item())
            self.train_metrics['binary']['accuracy'].append(binary_acc.item())

        elif self.training_phase == 'multilabel':
            self.train_metrics['multilabel']['loss'].append(train_loss.item())
            self.train_metrics['multilabel']['f1'].append(multilabel_f1.item())
            self.train_metrics['multilabel']['accuracy'].append(multilabel_acc.item())

        else:
            self.train_metrics['binary']['loss'].append(train_loss.item())
            self.train_metrics['binary']['f1'].append(binary_f1.item())
            self.train_metrics['binary']['accuracy'].append(binary_acc.item())

            self.train_metrics['multilabel']['loss'].append(train_loss.item())
            self.train_metrics['multilabel']['f1'].append(multilabel_f1.item())
            self.train_metrics['multilabel']['accuracy'].append(multilabel_acc.item())


    def on_validation_epoch_end(self):
        binary_acc = self.trainer.callback_metrics.get('val_binary_acc')
        binary_f1 = self.trainer.callback_metrics.get('val_binary_f1')
        multilabel_acc = self.trainer.callback_metrics.get('val_multilabel_acc')
        multilabel_f1 = self.trainer.callback_metrics.get('val_multilabel_f1')
        val_loss = self.trainer.callback_metrics.get('val_loss')

        if self.training_phase == 'binary':
            self.val_metrics['binary']['loss'].append(val_loss.item())
            self.val_metrics['binary']['f1'].append(binary_f1.item())
            self.val_metrics['binary']['accuracy'].append(binary_acc.item())

        elif self.training_phase == 'multilabel':
            self.val_metrics['multilabel']['loss'].append(val_loss.item())
            self.val_metrics['multilabel']['f1'].append(multilabel_f1.item())
            self.val_metrics['multilabel']['accuracy'].append(multilabel_acc.item())

        else:
            self.val_metrics['binary']['loss'].append(val_loss.item())
            self.val_metrics['binary']['f1'].append(binary_f1.item())
            self.val_metrics['binary']['accuracy'].append(binary_acc.item())

            self.val_metrics['multilabel']['loss'].append(val_loss.item())
            self.val_metrics['multilabel']['f1'].append(multilabel_f1.item())
            self.val_metrics['multilabel']['accuracy'].append(multilabel_acc.item())


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=4)

        if self.training_phase == 'binary':
            monitor_metric = 'val_binary_f1'
        elif self.training_phase == 'multilabel':
            monitor_metric = 'val_multilabel_f1'
        else:
           
            monitor_metric = 'val_multilabel_f1'

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor_metric
            }
        }
 
    def on_fit_end(self):
        if self.training_phase == 'binary':
            train_loss = self.train_metrics['binary']['loss']
            val_loss = self.val_metrics['binary']['loss']
            train_binary_f1 = self.train_metrics['binary']['f1']
            val_binary_f1 = self.val_metrics['binary']['f1']

            print("Binary classification curves") 
            plot_loss_curves(train_loss, val_loss, train_binary_f1, val_binary_f1)

        elif self.training_phase == 'multilabel':
            train_loss = self.train_metrics['multilabel']['loss']
            val_loss = self.val_metrics['multilabel']['loss']
            train_multilabel_f1 = self.train_metrics['multilabel']['f1']
            val_multilabel_f1 = self.val_metrics['multilabel']['f1']

            print("Multilabel classification curves") 
            plot_loss_curves(train_loss, val_loss, train_multilabel_f1, val_multilabel_f1)

        else:
            train_loss = self.train_metrics['binary']['loss']
            val_loss = self.val_metrics['binary']['loss']
            train_binary_f1 = self.train_metrics['binary']['f1']
            val_binary_f1 = self.val_metrics['binary']['f1']

            print("Binary classification curves") 
            plot_loss_curves(train_loss, val_loss, train_binary_f1, val_binary_f1)

            train_multilabel_loss = self.train_metrics['multilabel']['loss']
            val_multilabel_loss = self.val_metrics['multilabel']['loss']
            train_multilabel_f1 = self.train_metrics['multilabel']['f1']
            val_multilabel_f1 = self.val_metrics['multilabel']['f1']

            print("Multilabel classification curves") 
            plot_loss_curves(train_multilabel_loss, val_multilabel_loss, train_multilabel_f1, val_multilabel_f1)