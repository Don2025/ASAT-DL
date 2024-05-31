from torch import nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_size, num_filters, filter_sizes, hidden_size, num_classes, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(filter_size, embedding_size))
            for filter_size in filter_sizes
        ])
        # self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters * len(filter_sizes), hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x 的大小：(batch_size, sequence_length, embedding_size)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_size)
        # 经过多个卷积层
        conv_output = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch_size, num_filters, seq_len - filter_sizes[n] + 1]
        # 经过多个池化层
        pool_output = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conv_output]  # [batch_size, num_filters]
        # 拼接所有池化结果
        concat_output = torch.cat(pool_output, dim=1) # [batch_size, len(filter_sizes) * num_filters]
        # 经过全连接层和输出层
        hidden_output = F.relu(self.fc(concat_output))
        logits = self.output_layer(hidden_output)
        return logits

net = TextCNN(embedding_size=768, num_filters=100, filter_sizes=[3, 4, 5], hidden_size=256, num_classes=2)
devices = try_all_gpus()
lr, num_epochs = 0.001, 20
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
train(net, train_loader, test_loader, loss, trainer, num_epochs, devices)