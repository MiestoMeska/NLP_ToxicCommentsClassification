import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelAccuracy,
    BinaryF1Score, BinaryAccuracy
)
from transformers import DistilBertModel
from utils.model.module_funcs import plot_loss_curves

class Main_Model_Multihead(pl.LightningModule):
    def __init__(self, freeze_base=False, num_labels=6, learning_rate=1e-5, class_weights=None, binary_weights=None, binary_threshold=None, multilabel_thresholds=None):
        super(Main_Model_Multihead, self).__init__()

        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.binary_weights = binary_weights
        self.num_labels = num_labels
        self.binary_threshold = binary_threshold if binary_threshold is not None else 0.5
        self.multilabel_thresholds = multilabel_thresholds if multilabel_thresholds is not None else [0.5] * num_labels

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.binary_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.toxic_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.severe_toxic_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.obscene_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.threat_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.insult_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.identity_hate_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.binary_weights)
        self.multilabel_loss_fns = [
            nn.BCEWithLogitsLoss(pos_weight=self.class_weights[i]) for i in range(self.num_labels)
        ]

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

        toxic_output = self.toxic_classifier(pooled_output)
        severe_toxic_output = self.severe_toxic_classifier(pooled_output)
        obscene_output = self.obscene_classifier(pooled_output)
        threat_output = self.threat_classifier(pooled_output)
        insult_output = self.insult_classifier(pooled_output)
        identity_hate_output = self.identity_hate_classifier(pooled_output)

        return binary_output, [toxic_output, severe_toxic_output, obscene_output, threat_output, insult_output, identity_hate_output]

    def compute_loss(self, binary_output, multilabel_outputs, binary_target, multilabel_target):
        if binary_target.dim() == 3:
            binary_target = binary_target.squeeze(-1)
        
        binary_loss = self.binary_loss_fn(binary_output, binary_target)

        multilabel_losses = []
        for i in range(self.num_labels):
            target_i = multilabel_target[:, i].unsqueeze(1)
            multilabel_losses.append(self.multilabel_loss_fns[i](multilabel_outputs[i], target_i))

        total_loss = binary_loss + sum(multilabel_losses)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        binary_target = batch['binary_targets'].float()
        multilabel_target = batch['multi_targets'].float()

        binary_output, multilabel_output = self(input_ids=input_ids, attention_mask=attention_mask)

        binary_preds = torch.sigmoid(binary_output) > self.binary_threshold
        toxic_samples = binary_preds.squeeze() > self.binary_threshold

        loss = self.compute_loss(binary_output, multilabel_output, binary_target, multilabel_target)

        self.train_binary_accuracy(binary_preds, binary_target.int())
        self.train_binary_f1(binary_preds, binary_target.int())

        if toxic_samples.any():
            toxic_indices = toxic_samples.nonzero(as_tuple=True)[0]
            multilabel_preds = [
                torch.sigmoid(multilabel_output[i][toxic_indices]) > self.multilabel_thresholds[i]
                for i in range(self.num_labels)
            ]
            multilabel_preds = torch.cat(multilabel_preds, dim=1)
            multilabel_targets_selected = multilabel_target[toxic_indices]

            self.train_multilabel_accuracy(multilabel_preds, multilabel_targets_selected.int())
            self.train_multilabel_f1(multilabel_preds, multilabel_targets_selected.int())

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['ids']
        attention_mask = batch['mask']
        binary_target = batch['binary_targets'].float()
        multilabel_target = batch['multi_targets'].float()

        with torch.no_grad():
            binary_output, multilabel_output = self(input_ids=input_ids, attention_mask=attention_mask)

            binary_preds = torch.sigmoid(binary_output) > self.binary_threshold
            toxic_samples = binary_preds.squeeze() > self.binary_threshold

            loss = self.compute_loss(binary_output, multilabel_output, binary_target, multilabel_target)

            self.val_binary_accuracy(binary_preds, binary_target.int())
            self.val_binary_f1(binary_preds, binary_target.int())

            if toxic_samples.any():
                toxic_indices = toxic_samples.nonzero(as_tuple=True)[0]
                multilabel_preds = [
                    torch.sigmoid(multilabel_output[i][toxic_indices]) > self.multilabel_thresholds[i]
                    for i in range(self.num_labels)
                ]
                multilabel_preds = torch.cat(multilabel_preds, dim=1)
                multilabel_targets_selected = multilabel_target[toxic_indices]

                self.val_multilabel_accuracy(multilabel_preds, multilabel_targets_selected.int())
                self.val_multilabel_f1(multilabel_preds, multilabel_targets_selected.int())

            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        binary_acc = self.train_binary_accuracy.compute()
        binary_f1 = self.train_binary_f1.compute()

        self.log('train_binary_acc_epoch', binary_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_binary_f1_epoch', binary_f1, on_epoch=True, prog_bar=True, logger=True)

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

            self.log('train_multilabel_acc_epoch', multilabel_acc, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_multilabel_f1_epoch', multilabel_f1, on_epoch=True, prog_bar=True, logger=True)

            self.train_metrics['multilabel']['accuracy'].append(multilabel_acc.item())
            self.train_metrics['multilabel']['f1'].append(multilabel_f1.item())

            if train_loss is not None:
                self.train_metrics['multilabel']['loss'].append(train_loss.item())

            self.train_multilabel_accuracy.reset()
            self.train_multilabel_f1.reset()

    def on_validation_epoch_end(self):
        binary_acc = self.val_binary_accuracy.compute()
        binary_f1 = self.val_binary_f1.compute()

        self.log('val_binary_acc_epoch', binary_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_binary_f1_epoch', binary_f1, on_epoch=True, prog_bar=True, logger=True)

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

            self.log('val_multilabel_acc_epoch', multilabel_acc, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_multilabel_f1_epoch', multilabel_f1, on_epoch=True, prog_bar=True, logger=True)

            self.val_metrics['multilabel']['accuracy'].append(multilabel_acc.item())
            self.val_metrics['multilabel']['f1'].append(multilabel_f1.item())

            if val_loss is not None:
                self.val_metrics['multilabel']['loss'].append(val_loss.item())

            self.val_multilabel_accuracy.reset()
            self.val_multilabel_f1.reset()

    def on_train_start(self):
        self.train_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}
        
        self.val_metrics = {'binary': {'loss': [], 'f1': [], 'accuracy': []},
                            'multilabel': {'loss': [], 'f1': [], 'accuracy': []}}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_multilabel_f1_epoch'
            }
        }

    def on_fit_end(self):
        plot_loss_curves(self.train_metrics, self.val_metrics)
