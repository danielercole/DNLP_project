from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

class GeneratorModel(nn.Module):
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 256,
        truncation: str = "longest_first",
        padding: str = "max_length",
    ):
        super(GeneratorModel, self).__init__()
        
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding
        
        # Usa un modello CausalLM (ad es. GPT‑2 Instruct)
        if pretrained_path is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_path)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}/tokenizer/")
        
        self.tokenizer.padding_side = "left"
        # Se il token di padding non esiste, impostalo uguale a eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ci sono 2 token di eos
        # openai-community/gpt2          usa '<|endoftext|>' (50256)
        # vicgalle/gpt2-open-instruct-v1 usa '### End'       (50257)
        if model_name_or_path == "openai-community/gpt2":
            self.custom_eos_token_id = 50256
        elif model_name_or_path == "vicgalle/gpt2-open-instruct-v1":
            self.custom_eos_token_id = 50257
        else:
            self.custom_eos_token_id = self.tokenizer.eos_token
        
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
    
    def forward(
        self,
        sentences: List[str],
        target_sentences: Optional[List[str]] = None,
        device = None,
    ):
        """
        Se vengono forniti target_sentences, per ciascun esempio:
          0. Tokenizza il prompt (senza troncamento) per ottenere la lunghezza completa.
          1. (loss)
            1. Calcola lo spazio residuo per il target come max_seq_length - len(prompt).
            2. Tokenizza il target con truncation a target_max_length.
            3. Concatena prompt e target
            4. Crea la mask (1 per prompt + target, 0 per pad)
            5. Crea le label (mask prompt + pad a -100).
            6. Calcola la loss con teacher forcing
          2. (generate)
            1. Crea la mask (1 per prompt, 0 per pad)
            2. Genera fino eos

        In modalità inferenza, viene tokenizzato solo il prompt.
        """
      
        # Costruisci il prompt per ogni input
        prompts = []
        for sentence in sentences:
            prompt = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
                "### Instruction:\n"
                "Trasform the following sentence from informal style to formal style:\n"
                f"\"{sentence.strip()}\"\n\n"
                "### Response:\n"
            )
            prompts.append(prompt)
        
        if target_sentences is not None:
            all_prompts_ids = []
            all_prompts_att_masks = []
            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            
            for prompt, target in zip(prompts, target_sentences):
                # Tokenizza il prompt SENZA troncamento per ottenere la lunghezza completa
                tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt")
                prompt_ids = tokenized_prompt["input_ids"].squeeze(0)
                prompt_len = prompt_ids.size(0)

                # Save the prompt and its attention for the generation
                prompt_att_mask = torch.ones_like(prompt_ids)
                if prompt_len < self.max_seq_length:
                    pad_length = self.max_seq_length - prompt_len
                    pad_tensor = torch.full((pad_length,), self.custom_eos_token_id, dtype=prompt_ids.dtype)
                    prompt_ids = torch.cat([prompt_ids, pad_tensor], dim=0)
                    attention_pad = torch.zeros(pad_length, dtype=prompt_att_mask.dtype)
                    prompt_att_mask = torch.cat([prompt_att_mask, attention_pad], dim=0)
                else:
                    prompt_ids = prompt_ids[:self.max_seq_length]
                    prompt_att_mask = prompt_att_mask[:self.max_seq_length]

                all_prompts_ids.append(prompt_ids)
                all_prompts_att_masks.append(prompt_att_mask)
                
                # Calcola lo spazio residuo per il target
                target_max_length = max(self.max_seq_length - prompt_len, 0)

                """ # Mettiamo una max length di 512 e non dovrebbe succedere
                if target_max_length <= 0:
                    # Invece di lanciare un'eccezione, troncate il prompt a metà della lunghezza massima
                    tokenized_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_seq_length // 2, return_tensors="pt")
                    prompt_ids = tokenized_prompt["input_ids"].squeeze(0)
                    prompt_len = prompt_ids.size(0)
                    target_max_length = self.max_seq_length - prompt_len
                    if target_max_length <= 0:
                        # Se ancora insufficiente, assegna almeno 1 token per il target
                        target_max_length = 1
                """

                # Tokenizza il target con truncation a target_max_length
                tokenized_target = self.tokenizer(target, truncation=True, max_length=target_max_length, return_tensors="pt")
                target_ids = tokenized_target["input_ids"].squeeze(0)
                # Concatena prompt e target
                input_ids = torch.cat([prompt_ids, target_ids], dim=0)
                # Crea l'attention mask: 1 per i token presenti
                attention_mask = torch.ones_like(input_ids)
                # Crea le label: maschera la parte del prompt
                labels = input_ids.clone()
                labels[:prompt_len] = -100
                # Se la sequenza risultante è più corta di max_seq_length, effettua il padding
                seq_length = input_ids.size(0)
                if seq_length < self.max_seq_length:
                    pad_length = self.max_seq_length - seq_length
                    pad_tensor = torch.full((pad_length,), self.custom_eos_token_id, dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, pad_tensor], dim=0)
                    attention_pad = torch.zeros(pad_length, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, attention_pad], dim=0)
                    label_pad = torch.full((pad_length,), -100, dtype=labels.dtype)
                    labels = torch.cat([labels, label_pad], dim=0)
                else:
                    input_ids = input_ids[:self.max_seq_length]
                    attention_mask = attention_mask[:self.max_seq_length]
                    labels = labels[:self.max_seq_length]
                
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)
            
            # Convert the lists to torch tensors
            batch_prompt_ids = torch.stack(all_prompts_ids).long()
            batch_prompt_att_mask = torch.stack(all_prompts_att_masks).long()
            batch_input_ids = torch.stack(all_input_ids).long()
            batch_attention_mask = torch.stack(all_attention_masks).long()
            batch_labels = torch.stack(all_labels).long()
            
            # Move to device
            batch_prompt_ids = batch_prompt_ids.to(device)
            batch_prompt_att_mask = batch_prompt_att_mask.to(device)
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            # Compute the loss using TEACHER FORCING
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )
            
            # Generate response
            generated_ids = self.model.generate(
                input_ids=batch_prompt_ids,
                attention_mask=batch_prompt_att_mask,
                max_new_tokens=self.max_seq_length,
                num_beams=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                encoder_repetition_penalty=0.1,
                pad_token_id=self.custom_eos_token_id,
                eos_token_id=self.custom_eos_token_id,
                early_stopping=True
            )
            
            # Cut the prompt from the responses
            transferred_sentences = []
            for s, g in zip(sentences, generated_ids):
                response = self.tokenizer.decode(g, skip_special_tokens=True)
                output = response[response.find(s) + len(s) + 3:]
                transferred_sentences.append(output)
            
            return generated_ids, transferred_sentences, outputs.loss
        else:
            # Inference mode
            all_input_ids = []
            all_attention_masks = []
            
            # Tokenize the prompt and make them all the same length
            for prompt in prompts:
                tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt")
                prompt_ids = tokenized_prompt["input_ids"].squeeze(0)
                prompt_len = prompt_ids.size(0)
                if prompt_len < self.max_seq_length:
                    pad_length = self.max_seq_length - prompt_len
                    pad_tensor = torch.full((pad_length,), self.custom_eos_token_id, dtype=prompt_ids.dtype)
                    input_ids = torch.cat([prompt_ids, pad_tensor], dim=0)
                    attention_mask = torch.cat([torch.ones(prompt_len, dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)], dim=0)
                else:
                    input_ids = prompt_ids[:self.max_seq_length]
                    attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
            
            batch_input_ids = torch.stack(all_input_ids).long()
            batch_attention_mask = torch.stack(all_attention_masks).long()

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            
            # Generate response
            generated_ids = self.model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=self.max_seq_length,
                num_beams=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                encoder_repetition_penalty=0.1,
                pad_token_id=self.custom_eos_token_id,
                eos_token_id=self.custom_eos_token_id,
                early_stopping=True
            )
            
            # Cut the prompt from the responses
            transferred_sentences = []
            for s, g in zip(sentences, generated_ids):
                response = self.tokenizer.decode(g, skip_special_tokens=True)
                output = response[response.find(s) + len(s) + 3:]
                transferred_sentences.append(output)

            return generated_ids, transferred_sentences

    def transfer(
        self,
        sentences: List[str],
        device = None
    ):
        # Chiama direttamente forward
        _, transferred_sentences = self.forward(sentences)
        return transferred_sentences

    def save_model(
        self, 
        path: Union[str]
    ):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer/")
