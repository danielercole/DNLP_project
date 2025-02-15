import numpy as np
import pandas as pd
import pickle

import os
import sys

import torch
import evaluate
from transformers import CTRLTokenizer, CTRLLMHeadModel, AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator():

    def __init__(self, args, device, experiment):
        """ Class for evaluation """
        super(Evaluator, self).__init__()

        self.tokenizer = CTRLTokenizer.from_pretrained("salesforce/ctrl")
        self.model = CTRLLMHeadModel.from_pretrained("salesforce/ctrl").to(device)

        self.args = args
        self.experiment = experiment

        self.bleu = evaluate.load('sacrebleu')
        self.rouge = evaluate.load('rouge')
        if args.bertscore: self.bertscore = evaluate.load('bertscore')
    

    def __compute_metric__(self, predictions, references, metric_name, direction=None):
        # predictions = list | references = list of lists
        scores = []
        if metric_name in ['bleu', 'rouge', 'bertscore']:
            for pred, ref in zip(predictions, references):
                if metric_name == 'bleu':
                    res = self.bleu.compute(predictions=[pred], references=[[ref]])
                    scores.append(res['score'])
                elif metric_name == 'rouge':
                    tmp_rouge1, tmp_rouge2, tmp_rougeL = [], [], []
                    for r in ref:
                        res = self.rouge.compute(predictions=[pred], references=[r], use_aggregator=False)
                        # tmp_rouge1.append(res['rouge1'][0].fmeasure)
                        # tmp_rouge2.append(res['rouge2'][0].fmeasure)
                        # tmp_rougeL.append(res['rougeL'][0].fmeasure)
                        tmp_rouge1.append(res['rouge1'][0])
                        tmp_rouge2.append(res['rouge2'][0])
                        tmp_rougeL.append(res['rougeL'][0])
                    scores.append([max(tmp_rouge1), max(tmp_rouge2), max(tmp_rougeL)])
                elif metric_name == 'bertscore':
                    res = self.bertscore.compute(predictions=[pred], references=[ref], lang=self.args.lang)
                    scores.extend(res['f1'])
        else:
            raise Exception(f"Metric {metric_name} is not supported.")
        return scores

    def __compute_classif_metrics__(self, pred_A, pred_B):
        device = self.model.device
        truncation, padding = 'longest_first', 'max_length'
        if 'lambdas' not in vars(self.args) or self.args.lambdas[4] == 0 or self.args.pretrained_classifier_eval != self.args.pretrained_classifier_model:
            classifier = AutoModelForSequenceClassification.from_pretrained(self.args.pretrained_classifier_eval)
            classifier_tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_classifier_eval)
            classifier.to(device)
        else:
            assert False, "WTF, salesforce/CTRL doesn't have a classifier"
        
        classifier.eval()

        y_pred, y_true = [], np.concatenate((np.full(len(pred_A), 0), np.full(len(pred_B), 1)))

        for i in range(0, len(pred_A), self.args.batch_size):
            batch_a = pred_A[i:i+self.args.batch_size]
            inputs = classifier_tokenizer(batch_a, truncation=truncation, padding=padding, max_length=self.args.max_sequence_length, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                output = classifier(**inputs)
            y_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
        for i in range(0, len(pred_B), self.args.batch_size):
            batch_b = pred_B[i:i+self.args.batch_size]
            inputs = classifier_tokenizer(batch_b, truncation=truncation, padding=padding, max_length=self.args.max_sequence_length, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                output = classifier(**inputs)
            y_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1
    
    def run_eval_ref(self, epoch, current_training_step, phase, parallel_dl_evalAB, parallel_dl_evalBA):
        print(f'Start {phase}...')

        if self.args.comet_logging:
            if phase == 'validation': context = self.experiment.validate
            elif phase == 'test': context = self.experiment.test
        
        real_A, real_B = [], []
        pred_A, pred_B = [], []
        ref_A, ref_B = [], []
        scores_AB_bleu_self, scores_BA_bleu_self = [], []
        scores_AB_bleu_ref, scores_BA_bleu_ref = [], []
        scores_AB_r1_self, scores_BA_r1_self, scores_AB_r2_self, scores_BA_r2_self, scores_AB_rL_self, scores_BA_rL_self = [], [], [], [], [], []
        scores_AB_r1_ref, scores_BA_r1_ref, scores_AB_r2_ref, scores_BA_r2_ref, scores_AB_rL_ref, scores_BA_rL_ref = [], [], [], [], [], []
        scores_AB_bscore, scores_BA_bscore = [], []

        for batch in parallel_dl_evalAB:
            parallel_a = list(batch[0])
            references_b = list(batch[1])
            if self.args.lowercase_ref:
                references_b = [[ref.lower() for ref in refs] for refs in references_b]
            
            transferred = self.CTRLtransfer(sentences=parallel_a, direction='AB')
            
            real_A.extend(parallel_a)
            pred_B.extend(transferred)
            ref_B.extend(references_b)
            parallel_a = [[s] for s in parallel_a]
            scores_AB_bleu_self.extend(self.__compute_metric__(transferred, parallel_a, 'bleu'))
            scores_AB_bleu_ref.extend(self.__compute_metric__(transferred, references_b, 'bleu'))
            scores_rouge_self = np.array(self.__compute_metric__(transferred, parallel_a, 'rouge'))
            scores_AB_r1_self.extend(scores_rouge_self[:, 0].tolist())
            scores_AB_r2_self.extend(scores_rouge_self[:, 1].tolist())
            scores_AB_rL_self.extend(scores_rouge_self[:, 2].tolist())
            scores_rouge_ref = np.array(self.__compute_metric__(transferred, references_b, 'rouge'))
            scores_AB_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            scores_AB_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            scores_AB_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
            if self.args.bertscore: scores_AB_bscore.extend(self.__compute_metric__(transferred, references_b, 'bertscore'))
            else: scores_AB_bscore.extend([0])
        avg_AB_bleu_self, avg_AB_bleu_ref = np.mean(scores_AB_bleu_self), np.mean(scores_AB_bleu_ref)
        avg_AB_bleu_geom = (avg_AB_bleu_self*avg_AB_bleu_ref)**0.5
        avg_AB_r1_self, avg_AB_r2_self, avg_AB_rL_self = np.mean(scores_AB_r1_self), np.mean(scores_AB_r2_self), np.mean(scores_AB_rL_self)
        avg_AB_r1_ref, avg_AB_r2_ref, avg_AB_rL_ref = np.mean(scores_AB_r1_ref), np.mean(scores_AB_r2_ref), np.mean(scores_AB_rL_ref)
        avg_AB_bscore = np.mean(scores_AB_bscore)

        for batch in parallel_dl_evalBA:
            parallel_b = list(batch[0])
            references_a = list(batch[1])
            if self.args.lowercase_ref:
                references_a = [[ref.lower() for ref in refs] for refs in references_a]
            
            transferred = self.CTRLtransfer(sentences=parallel_b, direction='BA')

            real_B.extend(parallel_b)
            pred_A.extend(transferred)
            ref_A.extend(references_a)
            parallel_b = [[s] for s in parallel_b]
            scores_BA_bleu_self.extend(self.__compute_metric__(transferred, parallel_b, 'bleu'))
            scores_BA_bleu_ref.extend(self.__compute_metric__(transferred, references_a, 'bleu'))
            scores_rouge_self = np.array(self.__compute_metric__(transferred, parallel_b, 'rouge'))
            scores_BA_r1_self.extend(scores_rouge_self[:, 0].tolist())
            scores_BA_r2_self.extend(scores_rouge_self[:, 1].tolist())
            scores_BA_rL_self.extend(scores_rouge_self[:, 2].tolist())
            scores_rouge_ref = np.array(self.__compute_metric__(transferred, references_a, 'rouge'))
            scores_BA_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            scores_BA_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            scores_BA_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
            if self.args.bertscore: scores_BA_bscore.extend(self.__compute_metric__(transferred, references_a, 'bertscore'))
            else: scores_BA_bscore.extend([0])
        avg_BA_bleu_self, avg_BA_bleu_ref = np.mean(scores_BA_bleu_self), np.mean(scores_BA_bleu_ref)
        avg_BA_bleu_geom = (avg_BA_bleu_self*avg_BA_bleu_ref)**0.5
        avg_BA_r1_self, avg_BA_r2_self, avg_BA_rL_self = np.mean(scores_BA_r1_self), np.mean(scores_BA_r2_self), np.mean(scores_BA_rL_self)
        avg_BA_r1_ref, avg_BA_r2_ref, avg_BA_rL_ref = np.mean(scores_BA_r1_ref), np.mean(scores_BA_r2_ref), np.mean(scores_BA_rL_ref)
        avg_BA_bscore = np.mean(scores_BA_bscore)
        avg_2dir_bleu_ref = (avg_AB_bleu_ref + avg_BA_bleu_ref) / 2

        metrics = {'epoch':epoch, 'step':current_training_step,
                   'self-BLEU A->B':avg_AB_bleu_self, 'self-BLEU B->A':avg_BA_bleu_self,
                   'ref-BLEU A->B':avg_AB_bleu_ref, 'ref-BLEU B->A':avg_BA_bleu_ref,
                   'ref-BLEU avg':avg_2dir_bleu_ref,
                   'g-BLEU A->B':avg_AB_bleu_geom, 'g-BLEU B->A':avg_BA_bleu_geom,
                   'self-ROUGE-1 A->B':avg_AB_r1_self, 'self-ROUGE-1 B->A':avg_BA_r1_self,
                   'self-ROUGE-2 A->B':avg_AB_r2_self, 'self-ROUGE-2 B->A':avg_BA_r2_self,
                   'self-ROUGE-L A->B':avg_AB_rL_self, 'self-ROUGE-L B->A':avg_BA_rL_self,
                   'ref-ROUGE-1 A->B':avg_AB_r1_ref, 'ref-ROUGE-1 B->A':avg_BA_r1_ref,
                   'ref-ROUGE-2 A->B':avg_AB_r2_ref, 'ref-ROUGE-2 B->A':avg_BA_r2_ref,
                   'ref-ROUGE-L A->B':avg_AB_rL_ref, 'ref-ROUGE-L B->A':avg_BA_rL_ref,
                   'BERTScore A->B':avg_AB_bscore, 'BERTScore B->A':avg_BA_bscore}
        
        if phase == 'test':
            acc, prec, rec, f1 = self.__compute_classif_metrics__(pred_A, pred_B)
            metrics['style accuracy'] = acc
            metrics['style precision'] = prec
            metrics['style recall'] = rec
            metrics['style F1 score'] = f1
        
        if phase == 'validation':
            base_path = f"{self.args.save_base_folder}epoch_{epoch}/"
            if self.args.eval_strategy == 'epochs':
                suffix = f'epoch{epoch}'
                if epoch < self.args.additional_eval:
                    suffix += f'_step{current_training_step}'
            else: suffix = f'step{current_training_step}'
        else:
            if self.args.from_pretrained is not None:
                if self.args.save_base_folder is not None:
                    base_path = f"{self.args.save_base_folder}"
                else:
                    base_path = f"{self.args.from_pretrained}epoch_{epoch}/"
            else:
                base_path = f"{self.args.save_base_folder}test/epoch_{epoch}/"
            suffix = f'epoch{epoch}_test'
            if self.args.from_pretrained and 'GYAFCfm' in self.args.from_pretrained:
                if 'family' in self.args.path_paral_test_ref: ds = 'family'
                elif 'music' in self.args.path_paral_test_ref: ds = 'music'
                suffix += f'_{ds}'
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

        for m, v in metrics.items():
            if m not in ['epoch', 'step']:
                print(f'{m}: {v}')

        df_AB = pd.DataFrame()
        df_AB['A (source)'] = real_A
        df_AB['B (generated)'] = pred_B
        ref_B = np.array(ref_B)
        for i in range(self.args.n_references):
            df_AB[f'ref {i+1}'] = ref_B[:, i]
        df_AB.to_csv(f"{base_path}AB_{suffix}.csv", sep=',', header=True)
        df_BA = pd.DataFrame()
        df_BA['B (source)'] = real_B
        df_BA['A (generated)'] = pred_A
        ref_A = np.array(ref_A)
        for i in range(self.args.n_references):
            df_BA[f'ref {i+1}'] = ref_A[:, i]
        df_BA.to_csv(f"{base_path}BA_{suffix}.csv", sep=',', header=True)
        
        if self.args.comet_logging:    
            with context():
                self.experiment.log_table(f'./AB_{suffix}.csv', tabular_data=df_AB, headers=True)
                self.experiment.log_table(f'./BA_{suffix}.csv', tabular_data=df_BA, headers=True)
                for m, v in metrics.items():
                    if m not in ['epoch', 'step']:
                        self.experiment.log_metric(m, v, step=current_training_step, epoch=epoch)
        del df_AB, df_BA
        print(f'End {phase}...')

    def CTRLtransfer(self, sentences, direction):
        device = self.model.device

        CUSTOM_EOS = 246533  # End of Sequence token id

        prompts = []
        for sentence in sentences:
            if direction == "AB":
                prompts.append(f"Opinion Negative: {sentence.strip()} Positive:")
            else:
                prompts.append(f"Opinion Positive: {sentence.strip()} Negative:")
        
        # Inference mode
        all_prompts = []
        all_input_ids = []
        all_attention_masks = []
        
        # Tokenize the prompt and make them all the same length
        max_seq_length = -1
        for prompt in prompts:
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt")
            prompt_ids = tokenized_prompt["input_ids"].squeeze(0)
            prompt_len = prompt_ids.size(0)
            all_prompts.append((prompt_ids, prompt_len))

            if prompt_len > max_seq_length:
                max_seq_length = prompt_len

        for prompt_ids, prompt_len in all_prompts:
            pad_length = max_seq_length - prompt_len
            pad_tensor = torch.full((pad_length,), CUSTOM_EOS, dtype=prompt_ids.dtype)
            input_ids = torch.cat([prompt_ids, pad_tensor], dim=0)
            attention_mask = torch.cat([torch.ones(prompt_len, dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)], dim=0)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        batch_input_ids = torch.stack(all_input_ids)
        batch_attention_mask = torch.stack(all_attention_masks)

        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        
        # Generate response
        generated_ids = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=int(max_seq_length * 1.5),
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=CUSTOM_EOS,
            eos_token_id=CUSTOM_EOS
        )
        
        # Cut the prompt from the responses
        transferred_sentences = []
        for s, g in zip(prompts, generated_ids):
            response = self.tokenizer.decode(g, skip_special_tokens=True)
            output = response[response.find(s) + len(s):].strip()
            transferred_sentences.append(output)

        return transferred_sentences
