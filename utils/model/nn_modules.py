import torch
import torch.nn as nn
from transformers import DistilBertModel

class Base_Model(torch.nn.Module):
    def __init__(self, freeze_base=False):
        super(Base_Model, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 6)
        )

    def forward(self, input_ids, attention_mask):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        
        pooled_output = hidden_state[:, 0]
        
        out = self.classifier(pooled_output)
        return out
    
class Extended_Model(nn.Module):
    def __init__(self, freeze_base=False):
        super(Extended_Model, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.binary_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        self.multilabel_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]

        pooled_output = hidden_state[:, 0]

        binary_output = self.binary_classifier(pooled_output)

        multilabel_output = self.multilabel_classifier(pooled_output)

        return binary_output, multilabel_output