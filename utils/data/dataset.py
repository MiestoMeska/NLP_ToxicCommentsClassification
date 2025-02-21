import torch
from torch.utils.data import Dataset
from torch_lr_finder import TrainDataLoaderIter
from utils.data.preprocessing import clean_text_for_bert

class MultiLabelDataset(Dataset):
    """
    A custom Dataset class for handling multi-label text data.

    Args:
        dataframe (pandas.DataFrame): A DataFrame containing the text data and labels.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used for encoding the text.
        max_len (int): The maximum length of the tokenized sequences.
        label_columns (list of str): The list of column names corresponding to the labels in the DataFrame.
        new_data (bool, optional): If True, indicates that the dataset does not contain labels. Default is False.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used for encoding the text.
        data (pandas.DataFrame): The DataFrame containing the text data.
        text (pandas.Series): The Series containing the text comments from the DataFrame.
        new_data (bool): Indicates if the dataset does not contain labels.
        label_columns (list of str): The list of column names corresponding to the labels in the DataFrame.
        targets (numpy.ndarray): The array of target labels, if available.
        max_len (int): The maximum length of the tokenized sequences.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns a dictionary containing the tokenized input IDs, attention mask,
                            token type IDs, and optionally the targets for the text at the specified index.
    """
    def __init__(self, dataframe, tokenizer, max_len, label_columns, text_column, new_data=False, use_preprocess=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text_column = text_column 
        self.text = dataframe[self.text_column]
        self.new_data = new_data
        self.label_columns = label_columns
        self.use_preprocess = use_preprocess
        
        if not new_data:
            self.targets = self.data[self.label_columns].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        """
        Returns a dictionary containing the tokenized input IDs, attention mask, token type IDs,
        and optionally the targets for the text at the specified index.

        Args:
            index (int): The index of the text to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized input IDs ('ids'), attention mask ('mask'),
                  token type IDs ('token_type_ids'), and optionally the targets ('targets') as torch tensors.
        """
        text = str(self.text[index])
        text = " ".join(text.split())

        if self.use_preprocess:
            text = clean_text_for_bert(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            clean_up_tokenization_spaces=False
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        
        if not self.new_data:
            out['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return out


class CustomDataset(Dataset):
    """
    A custom Dataset class for handling both binary and multi-label text data.

    Args:
        dataframe (pandas.DataFrame): A DataFrame containing the text data and labels.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used for encoding the text.
        max_len (int): The maximum length of the tokenized sequences.
        label_columns (list of str): The list of column names corresponding to the multi-labels in the DataFrame.
        binary_label_column (str): The name of the binary label column (e.g., 'is_toxic').
        text_column (str): The column containing the text data.
        new_data (bool, optional): If True, indicates that the dataset does not contain labels. Default is False.
        use_preprocess (bool, optional): If True, applies preprocessing to the text data.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used for encoding the text.
        data (pandas.DataFrame): The DataFrame containing the text data.
        text (pandas.Series): The Series containing the text comments from the DataFrame.
        label_columns (list of str): The list of column names corresponding to the multi-labels in the DataFrame.
        binary_label_column (str): The name of the binary label column.
        targets (numpy.ndarray): The array of target labels (both binary and multi-label), if available.
        max_len (int): The maximum length of the tokenized sequences.
    """
    def __init__(self, dataframe, tokenizer, max_len, label_columns, binary_label_column, text_column, new_data=False, use_preprocess=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text_column = text_column 
        self.text = dataframe[self.text_column]
        self.new_data = new_data
        self.label_columns = label_columns
        self.binary_label_column = binary_label_column
        self.use_preprocess = use_preprocess
        
        if not new_data:
            self.binary_targets = self.data[self.binary_label_column].values
            self.multi_targets = self.data[self.label_columns].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        """
        Returns a dictionary containing the tokenized input IDs, attention mask, token type IDs,
        and optionally the targets (binary and multi-label) for the text at the specified index.

        Args:
            index (int): The index of the text to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized input IDs ('ids'), attention mask ('mask'),
                  token type IDs ('token_type_ids'), and optionally the binary and multi-label targets as torch tensors.
        """
        text = str(self.text[index])
        text = " ".join(text.split())

        if self.use_preprocess:
            text = clean_text_for_bert(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
        
        if not self.new_data:
            binary_target = torch.tensor(self.binary_targets[index], dtype=torch.float).unsqueeze(0)

            multi_target = torch.tensor(self.multi_targets[index], dtype=torch.float)

            out['binary_targets'] = binary_target
            out['multi_targets'] = multi_target
            print(out)

        return out





class CustomTrainDataLoaderIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        inputs = (batch_data['ids'], batch_data['mask'])
        labels = batch_data['targets']
        return inputs, labels