import torch
from torch import nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, **kwargs):
        super(AttentionClassifier, self).__init__(**kwargs)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len, hidden_size]
        attention_weights = self.attention(input_ids).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=-1).unsqueeze(-1)
        # attention_weights shape: [batch_size, seq_len, 1]
        attention_output = input_ids * attention_weights
        # attention_output shape: [batch_size, seq_len, hidden_size]
        sum_embeddings = torch.sum(attention_output, dim=1)
        # sum_embeddings shape: [batch_size, hidden_size]
        logits = self.classifier(sum_embeddings)
        # logits shape: [batch_size, num_labels]
        return logits

net = AttentionClassifier(hidden_size=768, num_labels=2)
devices = try_all_gpus()
lr, num_epochs = 0.001, 30
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
train(net, train_loader, test_loader, loss, trainer, num_epochs, devices)