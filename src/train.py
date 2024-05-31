import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.model_selection import train_test_split

class Defects4JDataset(Dataset):
    def __init__(self, defects4j_dir, test_size=0.2, random_state=None, tokenizer=None):
        self.project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']
        self.defects4j_dir = defects4j_dir
        self.test_size = test_size
        self.random_state = random_state
        self.df = pd.DataFrame()
        self.train_df, self.test_df = self.load_data()
        self.tokenizer = tokenizer
        self.train_encodings, self.test_encodings = self.encoding(tokenizer)
        self.train_dataset = self.create_dataset(self.train_encodings, self.train_df)
        self.test_dataset = self.create_dataset(self.test_encodings, self.test_df)

        
    def load_data(self):
        for pid in self.project_ids:
            df_t = pd.read_csv(f'{self.defects4j_dir}/csv/{pid}/{pid}-warnings_unique.csv')
            self.df = pd.concat([self.df, df_t], ignore_index=True)
        # Split data into train and test sets
        train_df, test_df = train_test_split(self.df, test_size=self.test_size, random_state=self.random_state)
        # Define text preprocessing function
        def preprocess_text(text):
            # Remove special characters, punctuation, and newlines
            text = text.replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('\n', '')
            # Convert to lowercase
            text = text.lower()
            return text
        # Apply text preprocessing to warning column in train and test sets
        train_df['Warning'] = train_df['Warning'].apply(preprocess_text)
        test_df['Warning'] = test_df['Warning'].apply(preprocess_text)
        return train_df, test_df
    
    def encoding(self, tokenizer):
        train_encodings = tokenizer(list(self.train_df['Warning']), truncation=True, padding=True)
        test_encodings = tokenizer(list(self.test_df['Warning']), truncation=True, padding=True)
        return train_encodings, test_encodings
    
    def create_dataset(self, encodings, df):
        return [self.get_item(idx, encodings, df.iloc[idx]['Label']) for idx in range(len(df))]
    
    def get_item(self, idx, encodings, label):
        item = {key: torch.tensor(val[idx]) for key, val in encodings.items()}
        item['label'] = torch.tensor(label)
        return item
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Extract input and label from row
        warning = row['Warning']
        label = row['Label']
        # Return as dictionary
        sample = {'Warning': warning, 'Label': label}
        return sample


model_name = 'bert-base-uncased' # saved in ~/.cache/huggingface/hub
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
defects4j_dir = os.path.expanduser('~/defects4j')
dataset = Defects4JDataset(defects4j_dir, tokenizer=tokenizer)
print(len(dataset), dataset[0])


import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer, TrainerCallback

device = 'cuda:0'
train_dataset = dataset.train_dataset
test_dataset = dataset.test_dataset
# print(len(train_dataset), train_dataset[0])
num_epochs = 5
training_args = TrainingArguments(
    output_dir='./output1/',  # 输出文件的目录
    num_train_epochs=num_epochs,  # 训练轮数
    per_device_train_batch_size=32,  # 每个设备的训练批次大小
    save_steps=500,  # 每隔多少步保存一次模型
    save_total_limit=3,  # 最多保存的模型数量
    learning_rate=2e-5,  # 学习率
    logging_dir='./logs1/',  # 日志文件的目录
    logging_steps = 100,
    evaluation_strategy='steps'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

class CustomCallback(TrainerCallback):
    def __init__(self, num_epochs):
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.num_epochs = num_epochs
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 在每次记录日志后调用
        # 记录训练损失
        if "loss" in logs:
            self.train_loss.append(logs["loss"])
        # 获取验证损失
        if "eval_loss" in logs:
            self.test_loss.append(logs["eval_loss"])
    
    def on_evaluate(self, args, state, control, **kwargs):
        # Compute Test Accuracy
        train_dataloader = trainer.get_train_dataloader()
        # Get model predictions and ground truth labels
        train_predictions = []
        train_labels = []
        for batch in train_dataloader:
            batch_inputs = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            with torch.no_grad():
                model_outputs = trainer.model(batch_inputs)
            batch_predictions = np.argmax(model_outputs.logits.cpu().detach().numpy(), axis=1)
            train_predictions.extend(batch_predictions)
            train_labels.extend(batch_labels.cpu().numpy())
        train_accuracy = np.mean(np.array(train_predictions) == np.array(train_labels))
        self.train_accuracy.append(train_accuracy)
        print(f"Train Accuracy: {train_accuracy}")
        # Compute Test Accuracy
        eval_dataloader = trainer.get_eval_dataloader()
        # Get model predictions and ground truth labels
        test_predictions = []
        test_labels = []
        for batch in eval_dataloader:
            batch_inputs = batch["input_ids"].to(device)
            batch_labels = batch["labels"].to(device)
            with torch.no_grad():
                model_outputs = trainer.model(batch_inputs)
            batch_predictions = np.argmax(model_outputs.logits.cpu().detach().numpy(), axis=1)
            test_predictions.extend(batch_predictions)
            test_labels.extend(batch_labels.cpu().numpy())
        # Compute accuracy
        test_accuracy = np.mean(np.array(test_predictions) == np.array(test_labels))
        self.test_accuracy.append(test_accuracy)
        print(f"Test Accuracy: {test_accuracy}")

# 创建callback对象并添加到trainer中
custom_callback = CustomCallback(num_epochs)
trainer.add_callback(custom_callback)
# 训练模型
trainer.train()


# 进行训练和评估
print("Model device:", model.device)
#########################
# 使用微调后的BERT模型进行预测
def preprocess_text(text):
    # Remove special characters, punctuation, and newlines
    text = text.replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('\n', '')
    # Convert to lowercase
    text = text.lower()
    return text


# 进行预测
def predict(model, text):
    model.eval() # 切换为评估模式
    model.to('cpu')
    with torch.no_grad():
        text = preprocess_text(text)
        encoding = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        output = model(**encoding)
        probs = torch.softmax(output.logits, dim=-1)
        label = torch.argmax(probs, dim=-1).item()
    return label, probs

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']
df = pd.DataFrame()
for pid in project_ids:
    df_t = pd.read_csv(f'{defects4j_dir}/csv/{pid}/{pid}-warnings.csv')
    df = pd.concat([df, df_t], ignore_index=True)
    texts = df['Warning'].tolist()
# 示例预测
for text in texts: 
    label, probs = predict(model, text)
    print("Text: ", text)
    print("Predicted Label: ", label)
    print("Probabilities: ", probs)