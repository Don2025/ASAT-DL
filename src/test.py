import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
# import nltk
# nltk.set_proxy('http://127.0.0.1:7890')
# nltk.download('punkt') # [nltk_data] Downloading package punkt to /home/tyd/nltk_data...
from nltk.tokenize import word_tokenize

class Defects4JDataset(Dataset):
    def __init__(self, defects4j_dir, model_name, test_size=0.2, random_state=None):
        self.project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']
        self.defects4j_dir = defects4j_dir
        self.model_name = model_name
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # What a fucking rich programmer dare do this! 
        # Please load the dataset onto the CPU
        # Otherwise when executing self.encoding() to over 30000 pieces of data,
        # RTX 3080Ti OutOfMemoryError: CUDA out of memory.
        self.model = AutoModel.from_pretrained(model_name)#.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.test_size = test_size
        self.random_state = random_state
        self.df = pd.DataFrame()
        self.train_df, self.test_df = self.load_data()
        self.train_encodings = self.encoding(self.train_df)
        self.test_encodings = self.encoding(self.test_df)
        self.train_dataset = self.create_dataset(self.train_encodings, self.train_df)
        self.test_dataset = self.create_dataset(self.test_encodings, self.test_df)
        
    def load_data(self):
        csv_folder = f'{self.defects4j_dir}/csv/'
        for pid in self.project_ids:
            df_t = pd.read_csv(f'{csv_folder}/{pid}/{pid}-modified_classes.csv')
            modified_classes = df_t['Modified Classes'].tolist()
            tokens = [x.split('.')[-1] for x in modified_classes]
            df_dw = pd.read_csv(f'{csv_folder}/{pid}/{pid}-detailed-warnings.csv')
            df_dw['nl_input'] = df_dw['Warning'] + " " + df_dw['Warning Detail']
            df_dw['Label'] = df_dw['Warning'].str.contains('|'.join(tokens)).astype(int)
            self.df = pd.concat([self.df, df_dw], ignore_index=True)
        row = set(zip(self.df['nl_input'].tolist(), self.df['Source Code'].tolist(), self.df['Label'].tolist()))
        df_unique = pd.DataFrame(list(row), columns=['nl_input', 'pl_input', 'label'])
        print(f"len(df_unique):{len(df_unique)}")
        # Split data into train and test sets
        train_df, test_df = train_test_split(df_unique, test_size=self.test_size, random_state=self.random_state)
        return train_df, test_df
    
    def encoding(self, df):
        nl_tokens = [word_tokenize(x) for x in list(self.train_df['nl_input'])]
        df['pl_input'].fillna('[UNK]', inplace=True)
        pl_tokens = [self.tokenizer.tokenize(x, padding=True, truncation=True) for x in list(df['pl_input'])]
        tokens = []
        print(len(nl_tokens), len(pl_tokens))
        # tokens = [[self.tokenizer.cls_token] + nl_tokens[i] + [self.tokenizer.sep_token] + pl_tokens[i] + [self.tokenizer.eos_token] for i in range(len(nl_tokens))]
        tokens_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        # tokens_ids = [token[:max_seq_length] + [self.tokenizer.pad_token_id] * (max_seq_length - len(token[:max_seq_length])) for token in tokens_ids] 
        max_seq_length = 512
        context_embeddings = []
        with torch.no_grad():
            for tokens_id in tokens_ids:
                tokens_tensor = torch.tensor(tokens_id[:max_seq_length])[None,:].to(self.model.device)
                context_embedding = self.model(tokens_tensor)[0]
                context_embeddings.append(context_embedding)
        return {'input_ids': tokens_ids, 'attention_mask': [[int(token_id != self.tokenizer.pad_token_id) for token_id in tokens] for tokens in tokens_ids], 'context_embeddings': context_embeddings}
    
    def create_dataset(self, encodings, df):
        class CustomDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['label'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        labels = df['label'].tolist()
        dataset = CustomDataset(encodings, labels)
        return dataset
    
    def get_item(self, inex):
        row = self.df.iloc[index]
        # Extract input and label from row
        warning = row['nl_input']
        code_line = row['Source Code']
        label = row['Label']
        # Return as dictionary
        sample = {'warning': nl_input, 'code': code_line, 'label': label }
        return sample
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[idx]) for key, val in encodings.items()}
        item['label'] = torch.tensor(label)
        return item


# 加载已经训练好的CodeBERT模型和标记器
model_name = "microsoft/codebert-base"
defects4j_dir = os.path.expanduser('~/defects4j')
dataset = Defects4JDataset(defects4j_dir, model_name)
print(len(dataset), dataset[123])
torch.save(dataset, "dataset.pt")