import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

def load_translation_model(src_lang: str, tgt_lang: str, device):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text_batch(text_list, model, tokenizer, device, max_length=256):
    tokenized_text = tokenizer(text_list, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
    
    with torch.amp.autocast('cuda'):
        with torch.no_grad():
            translation = model.generate(**tokenized_text)
    
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in translation.cpu()]
    
    return translated_texts

def back_translate_batch(text_list, model_en_fr, tokenizer_en_fr, model_fr_en, tokenizer_fr_en, device, max_length=256):
    french_texts = translate_text_batch(text_list, model_en_fr, tokenizer_en_fr, device, max_length)
    
    back_translated_texts = translate_text_batch(french_texts, model_fr_en, tokenizer_fr_en, device, max_length)
    
    return back_translated_texts

def augment_data_in_batches(df, batch_size, model_en_fr, tokenizer_en_fr, model_fr_en, tokenizer_fr_en, device):
    augmented_texts = []
    
    with tqdm(total=len(df), desc="Back Translating", unit="rows") as pbar:
        for i in range(0, len(df), batch_size):
            batch_texts = df['cleaned_text'].iloc[i:i+batch_size].tolist()
            
            torch.cuda.empty_cache()
            
            augmented_batch = back_translate_batch(batch_texts, model_en_fr, tokenizer_en_fr, model_fr_en, tokenizer_fr_en, device)
            augmented_texts.extend(augmented_batch)
            
            pbar.update(len(batch_texts))
    
    df.loc[:, 'augmented_text'] = augmented_texts 
    return df