import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelAccuracy,
    BinaryF1Score, BinaryAccuracy
)
from transformers import DistilBertModel
from utils.model.module_funcs import plot_loss_curves

class Main_Model_Conditional(pl.LightningModule):
    def __init__(self, freeze_base=False, num_labels=6, learning_rate=1e-5, class_weights=None, binary_weights=None):
        super(Main_Model_Conditional, self).__init__()

        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.binary_weights = binary_weights
        self.num_labels = num_labels

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.binary_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        self.multilabel_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_labels)
)

        self.binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.binary_weights)
        self.multilabel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        self.train_binary_f1 = BinaryF1Score()
        self.val_binary_f1 = BinaryF1Score()
        self.train_binary_accuracy = BinaryAccuracy()
        self.val_binary_accuracy = BinaryAccuracy()

        self.train_multilabel_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        self.val_multilabel_f1 = MultilabelF1Score(num_labels=self.num_labels, average='macro')
        self.train_multilabel_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)
        self.val_multilabel_accuracy = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5)

        self.train_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                              'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}
        self.val_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}


    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]
        pooled_output = hidden_state[:, 0]

        binary_output = self.binary_classifier(pooled_output)
        binary_preds = torch.sigmoid(binary_output) > 0.5

        multilabel_output = self.multilabel_classifier(pooled_output)

        return binary_output, multilabel_output

    def compute_loss(self, binary_output, multilabel_output, binary_target, multilabel_target):
        binary_loss = self.binary_loss_fn(binary_output, binary_target)

        toxic_mask = (binary_target > 0.5).squeeze()

        if toxic_mask.any():
            toxic_multilabel_output = multilabel_output[toxic_mask, :]
            toxic_multilabel_target = multilabel_target[toxic_mask, :]

            multilabel_loss = self.multilabel_loss_fn(toxic_multilabel_output, toxic_multilabel_target)
        else:
            multilabel_loss = torch.tensor(0.0).to(binary_loss.device)

        total_loss = binary_loss + multilabel_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        binary_target = batch['binary_targets'].float()
        multilabel_target = batch['multi_targets'].float()

        binary_output, multilabel_output = self(input_ids=input_ids, attention_mask=attention_mask)

        binary_preds = torch.sigmoid(binary_output) > 0.5
        toxic_samples = binary_preds.squeeze() > 0.5

        loss = self.compute_loss(binary_output, multilabel_output, binary_target, multilabel_target)

        self.train_binary_accuracy(binary_preds, binary_target.int())
        self.train_binary_f1(binary_preds, binary_target.int())


        if toxic_samples.any():
            multilabel_preds = torch.sigmoid(multilabel_output[toxic_samples]) > 0.5
            self.train_multilabel_accuracy(multilabel_preds, multilabel_target[toxic_samples].int())
            self.train_multilabel_f1(multilabel_preds, multilabel_target[toxic_samples].int())


        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        binary_target = batch['binary_targets'].float()
        multilabel_target = batch['multi_targets'].float()

        with torch.no_grad():
            binary_output, multilabel_output = self(input_ids=input_ids, attention_mask=attention_mask)

            binary_preds = torch.sigmoid(binary_output) > 0.5
            toxic_samples = binary_preds.squeeze() > 0.5

            loss = self.compute_loss(binary_output, multilabel_output, binary_target, multilabel_target)

            self.val_binary_accuracy(binary_preds, binary_target.int())
            self.val_binary_f1(binary_preds, binary_target.int())

            if toxic_samples.any():
                multilabel_preds = torch.sigmoid(multilabel_output[toxic_samples]) > 0.5
                self.val_multilabel_accuracy(multilabel_preds, multilabel_target[toxic_samples].int())
                self.val_multilabel_f1(multilabel_preds, multilabel_target[toxic_samples].int())

            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        binary_acc = self.train_binary_accuracy.compute()
        binary_f1 = self.train_binary_f1.compute()

        self.log('train_binary_acc_epoch', binary_acc, on_epoch=True, prog_bar=True)
        self.log('train_binary_f1_epoch', binary_f1, on_epoch=True, prog_bar=True)

        self.train_metrics['binary']['accuracy'].append(binary_acc.item())
        self.train_metrics['binary']['f1'].append(binary_f1.item())

        train_loss = self.trainer.callback_metrics.get('train_loss')
        if train_loss is not None:
            self.train_metrics['binary']['loss'].append(train_loss.item())

        self.train_binary_accuracy.reset()
        self.train_binary_f1.reset()

        if hasattr(self, 'train_multilabel_accuracy') and hasattr(self, 'train_multilabel_f1'):
            multilabel_acc = self.train_multilabel_accuracy.compute()
            multilabel_f1 = self.train_multilabel_f1.compute()

            self.log('train_multilabel_acc_epoch', multilabel_acc, on_epoch=True, prog_bar=True)
            self.log('train_multilabel_f1_epoch', multilabel_f1, on_epoch=True, prog_bar=True)

            self.train_metrics['multilabel']['accuracy'].append(multilabel_acc.item())
            self.train_metrics['multilabel']['f1'].append(multilabel_f1.item())

            if train_loss is not None:
                self.train_metrics['multilabel']['loss'].append(train_loss.item())

            self.train_multilabel_accuracy.reset()
            self.train_multilabel_f1.reset()


    def on_validation_epoch_end(self):
        binary_acc = self.val_binary_accuracy.compute()
        binary_f1 = self.val_binary_f1.compute()

        self.log('val_binary_acc_epoch', binary_acc, on_epoch=True, prog_bar=True)
        self.log('val_binary_f1_epoch', binary_f1, on_epoch=True, prog_bar=True)

        self.val_metrics['binary']['accuracy'].append(binary_acc.item())
        self.val_metrics['binary']['f1'].append(binary_f1.item())

        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_metrics['binary']['loss'].append(val_loss.item())

        self.val_binary_accuracy.reset()
        self.val_binary_f1.reset()

        if hasattr(self, 'val_multilabel_accuracy') and hasattr(self, 'val_multilabel_f1'):
            multilabel_acc = self.val_multilabel_accuracy.compute()
            multilabel_f1 = self.val_multilabel_f1.compute()

            self.log('val_multilabel_acc_epoch', multilabel_acc, on_epoch=True, prog_bar=True)
            self.log('val_multilabel_f1_epoch', multilabel_f1, on_epoch=True, prog_bar=True)

            self.val_metrics['multilabel']['accuracy'].append(multilabel_acc.item())
            self.val_metrics['multilabel']['f1'].append(multilabel_f1.item())

            if val_loss is not None:
                self.val_metrics['multilabel']['loss'].append(val_loss.item())

            self.val_multilabel_accuracy.reset()
            self.val_multilabel_f1.reset()

    def on_train_start(self):
        """
        Clears the metric logs for both training and validation before the first epoch starts
        (i.e., after the validation sanity check).
        """
        self.train_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}
        
        self.val_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_multilabel_f1_epoch'
            }
        }

    def on_fit_end(self):

        print("Binary classification curves")
        plot_loss_curves(self.train_metrics['binary']['loss'], self.val_metrics['binary']['loss'],
                         self.train_metrics['binary']['f1'], self.val_metrics['binary']['f1'])

        print("Multilabel classification curves")
        plot_loss_curves(self.train_metrics['multilabel']['loss'], self.val_metrics['multilabel']['loss'],
                         self.train_metrics['multilabel']['f1'], self.val_metrics['multilabel']['f1'])
