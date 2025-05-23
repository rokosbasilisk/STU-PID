#!/usr/bin/env python
# coding: utf-8

"""
RASPID (Reasoning-Aware Steering with PID control):
A method for dynamic control of language model generation using chunk-level classification
and PID-based steering to optimize reasoning quality while reducing redundancy.

This implementation:
- Trains a chunk-level classifier on labeled reasoning chains
- Builds control vectors from required vs redundant thoughts
- Uses PID control to dynamically steer generation away from redundant patterns
- Evaluates performance on GSM8K mathematical reasoning tasks
"""

import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from repeng import ControlModel, ControlVector
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Suppress warnings
warnings.filterwarnings("ignore", "To copy construct from a tensor", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RASPIDConfig:
    """Configuration class for RASPID hyperparameters"""
    
    # Model configuration
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32
    
    # Data paths
    LABELED_CSV = "gsm8k_chains_labeled_with_tokens.csv"
    CTRL_VEC_PATH = "ctrl_vector.pt"
    
    # Classifier hyperparameters
    EMB_LAYER = 20
    CHUNK_SIZES = [16, 24]
    BATCH_SIZE = 32
    FLUFF_STAR = 0.5  # Target probability for "redundant"
    
    # PID steering hyperparameters
    INIT_FREE = 80      # Let model reason freely first
    STEER_WINDOW = 60   # Duration of steering intervention
    KP, KI, KD = 0.05, 0.001, 0.001  # PID coefficients
    MAX_I = 0.20        # Maximum integral term
    MAX_ALPHA = 0.40    # Maximum steering coefficient
    STEER_MARGIN = 0.20 # Margin above FLUFF_STAR to trigger steering
    
    # Generation parameters
    BASE_TEMP = 0.60    # Base temperature for sampling
    STEER_TEMP = 0.30   # Temperature during steering
    MAX_REPEAT = 8      # Maximum consecutive token repetitions
    MAX_TOKENS = 4096   # Maximum tokens to generate
    MAX_RAW = 50.0      # Clamp value for classifier raw scores


class ChunkDataset(Dataset):
    """Dataset for creating fixed-size chunks from text sequences"""
    
    def __init__(self, texts, label, chunk_size, tokenizer):
        self.chunk_size = chunk_size
        self.chunks = []
        self.labels = []
        
        for text in texts:
            tok_ids = tokenizer.encode(text, add_special_tokens=False)
            # Create overlapping chunks
            for i in range(0, len(tok_ids) - chunk_size + 1, chunk_size):
                chunk = tok_ids[i:i + chunk_size]
                self.chunks.append(chunk)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx], self.labels[idx]


class ChunkClassifier:
    """Chunk-level classifier for identifying redundant vs required reasoning"""
    
    def __init__(self, config, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.best_classifier = None
        self.best_chunk_size = None
        self.best_accuracy = 0.0
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        input_ids, labels = zip(*batch)
        # Pad sequences on CPU for pin_memory efficiency
        seqs = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        padded = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = (padded != self.tokenizer.pad_token_id).long()
        return {"input_ids": padded, "attention_mask": attention_mask}, torch.tensor(labels, dtype=torch.long)
    
    def _extract_embeddings(self, loader):
        """Extract embeddings from model for all chunks"""
        features, labels = [], []
        self.model.eval()
        
        with torch.no_grad():
            for batch_tokens, batch_labels in tqdm(loader, desc="Extracting embeddings"):
                # Move batch to device
                batch_tokens = {
                    k: v.to(self.config.DEVICE, non_blocking=True)
                    for k, v in batch_tokens.items()
                }
                
                # Get model outputs and extract embeddings
                outputs = self.model(**batch_tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.config.EMB_LAYER]
                embeddings = hidden_states.mean(dim=1)  # Average pooling
                
                features.append(embeddings.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        return np.vstack(features), np.concatenate(labels)
    
    def _train_sgd_classifier(self, X_train, y_train, X_val, y_val, chunk_size):
        """Train SGD classifier with early stopping"""
        classifier = SGDClassifier(
            loss="log_loss", random_state=42, warm_start=True, max_iter=1, tol=None
        )
        
        prev_coef = None
        pbar = tqdm(range(500), desc=f"Training classifier (cs={chunk_size})", leave=False)
        
        for iteration in pbar:
            classifier.fit(X_train, y_train)
            current_coef = classifier.coef_
            
            if prev_coef is not None:
                delta = np.max(np.abs(current_coef - prev_coef))
                pbar.set_postfix(delta=delta)
                if delta < 1e-3:  # Convergence criterion
                    break
            prev_coef = current_coef.copy()
        
        pbar.close()
        
        # Evaluate on validation set
        val_accuracy = accuracy_score(y_val, classifier.predict(X_val))
        return classifier, val_accuracy
    
    def train(self, required_texts, redundant_texts):
        """Train classifier on required vs redundant text chunks"""
        print("Training chunk classifier...")
        
        for chunk_size in self.config.CHUNK_SIZES:
            print(f"Testing chunk size: {chunk_size}")
            
            # Create datasets
            dataset_required = ChunkDataset(required_texts, 0, chunk_size, self.tokenizer)
            dataset_redundant = ChunkDataset(redundant_texts, 1, chunk_size, self.tokenizer)
            
            # Create data loader
            combined_dataset = ConcatDataset([dataset_required, dataset_redundant])
            loader = DataLoader(
                combined_dataset,
                batch_size=self.config.BATCH_SIZE,
                collate_fn=self._collate_fn,
                shuffle=False,
                pin_memory=True,
                num_workers=0
            )
            
            # Extract embeddings
            X, y = self._extract_embeddings(loader)
            
            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            classifier, accuracy = self._train_sgd_classifier(
                X_train, y_train, X_val, y_val, chunk_size
            )
            
            print(f"Chunk size {chunk_size} → Validation accuracy: {accuracy:.3f}")
            
            # Keep track of best performer
            if accuracy > self.best_accuracy:
                self.best_chunk_size = chunk_size
                self.best_accuracy = accuracy
                self.best_classifier = classifier
        
        print(f"✅ Best classifier: chunk_size={self.best_chunk_size}, accuracy={self.best_accuracy:.3f}")
        return self.best_classifier, self.best_chunk_size


class ControlVectorBuilder:
    """Builder for control vectors from text contrasts"""
    
    def __init__(self, config, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
    
    def _compute_mean_hidden_state(self, texts):
        """Compute mean hidden state across texts"""
        hidden_states = []
        
        for text in tqdm(texts, desc="Computing hidden states"):
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.config.DEVICE)
            
            with torch.inference_mode():
                outputs = self.model(**tokens, output_hidden_states=True)
                hidden_state = outputs.hidden_states[self.config.EMB_LAYER][0]
                mean_hidden = hidden_state.mean(dim=0)
                hidden_states.append(mean_hidden.cpu())
        
        return torch.stack(hidden_states).mean(dim=0)
    
    def build_control_vector(self, required_texts, redundant_texts):
        """Build control vector from required vs redundant text contrasts"""
        print("Building control vector...")
        
        # Compute mean hidden states
        v_required = self._compute_mean_hidden_state(required_texts)
        v_redundant = self._compute_mean_hidden_state(redundant_texts)
        
        # Create control vector
        model_type = self.model.config.model_type
        control_vector = ControlVector(
            model_type=model_type,
            directions={self.config.EMB_LAYER: (v_required - v_redundant).to(self.config.DEVICE)}
        )
        
        # Save control vector
        torch.save(control_vector, self.config.CTRL_VEC_PATH)
        print("✅ Control vector saved")
        
        return control_vector


class RASPIDGenerator:
    """RASPID generator with PID-controlled steering"""
    
    def __init__(self, config, tokenizer, base_model, control_model, classifier, chunk_size, control_vector):
        self.config = config
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.control_model = control_model
        self.classifier = classifier
        self.chunk_size = chunk_size
        self.control_vector = control_vector
        self.stop_pattern = re.compile(r"\\boxed\{[^{}]{1,12}\}")
    
    def _handle_numerical_stability(self, tensor, name="tensor"):
        """Handle NaN values and extreme values in tensors"""
        if torch.isnan(tensor).any():
            print(f"[WARNING] NaN detected in {name}, replacing with zeros")
            tensor = torch.nan_to_num(tensor)
        return tensor
    
    def _scale_extreme_classifier_output(self, raw_score):
        """Scale down extreme classifier outputs for stability"""
        if abs(raw_score) > self.config.MAX_RAW:
            scaled = self.config.MAX_RAW * np.sign(raw_score) * np.log(1 + abs(raw_score) / self.config.MAX_RAW)
            return scaled
        return max(-self.config.MAX_RAW, min(self.config.MAX_RAW, raw_score))
    
    def _should_apply_steering(self, generation_length, steering_active, steering_start):
        """Determine if steering should be applied"""
        if not steering_active and generation_length >= self.config.INIT_FREE:
            return True, generation_length
        elif steering_active and generation_length - steering_start > self.config.STEER_WINDOW:
            return False, steering_start
        return steering_active, steering_start
    
    def _update_pid_controller(self, p_redundant, alpha, integral, derivative, prev_error):
        """Update PID controller based on redundancy probability"""
        error = 0.0
        
        if p_redundant > self.config.FLUFF_STAR + self.config.STEER_MARGIN:
            error = p_redundant - self.config.FLUFF_STAR
            
            # Update PID terms
            integral = max(-self.config.MAX_I, min(self.config.MAX_I, integral + self.config.KI * error))
            derivative = self.config.KD * (error - prev_error) + (1 - self.config.KD) * derivative
            
            # Calculate new alpha
            pid_output = self.config.KP * error + integral + derivative
            alpha = max(0.0, min(self.config.MAX_ALPHA, alpha + pid_output))
        
        return alpha, integral, derivative, error
    
    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=None, debug=False):
        """Generate text using RASPID with PID-controlled steering"""
        if max_new_tokens is None:
            max_new_tokens = self.config.MAX_TOKENS
        
        if debug:
            print(f"\n=== RASPID GENERATION ===")
            print(f"Prompt: '{prompt}'")
            print(f"Max tokens: {max_new_tokens}")
            print("--- GENERATION TRACE ---")
            print("step | steering | p_red |  err  |   α   |   I   |   D   | temp | token")
        
        # Initialize generation state
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE).input_ids[0]
        output_ids = input_ids.clone()
        past_key_values = None
        
        # Initialize PID controller state
        alpha = integral = derivative = prev_error = 0.0
        chunk_hidden = None
        tokens_in_chunk = 0
        steering_active = False
        steering_start = 0
        
        # Initialize repetition detection
        last_token = None
        repetition_count = 0
        generated_text = ""
        
        # Generation loop
        pbar = tqdm(range(max_new_tokens), desc="RASPID generation", leave=False)
        
        for step in pbar:
            generation_length = output_ids.size(0) - input_ids.size(0)
            
            # Update steering state
            steering_active, steering_start = self._should_apply_steering(
                generation_length, steering_active, steering_start
            )
            
            if debug and step == 0:
                print(f"[INFO] Steering will activate at generation length {self.config.INIT_FREE}")
            
            # Apply control vector
            coefficient = alpha if steering_active else 0.0
            self.control_model.set_control(self.control_vector, coeff=coefficient)
            
            # Forward pass
            if generation_length == 0:
                # First step: process entire prompt
                outputs = self.control_model(
                    input_ids=output_ids.unsqueeze(0),
                    use_cache=True,
                    output_hidden_states=True
                )
            else:
                # Subsequent steps: process only new token
                outputs = self.control_model(
                    input_ids=output_ids[-1:].unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1]
            hidden_state = outputs.hidden_states[self.config.EMB_LAYER][0, -1]
            
            # Handle numerical stability
            logits = self._handle_numerical_stability(logits, "logits")
            hidden_state = self._handle_numerical_stability(hidden_state, "hidden_state")
            
            # Check for token repetition
            current_token = output_ids[-1].item()
            if current_token == last_token:
                repetition_count += 1
                if repetition_count >= self.config.MAX_REPEAT:
                    if debug:
                        print(f"[INFO] Maximum repetition reached, stopping generation")
                    break
            else:
                repetition_count = 0
                last_token = current_token
            
            # Update chunk tracking
            hidden_normalized = hidden_state / (torch.norm(hidden_state) + 1e-8)
            chunk_hidden = hidden_normalized if chunk_hidden is None else chunk_hidden + hidden_normalized
            tokens_in_chunk += 1
            
            # Classifier prediction and PID update
            p_redundant = error = 0.0
            if tokens_in_chunk >= self.chunk_size:
                try:
                    # Prepare classifier input
                    classifier_input = (chunk_hidden / self.chunk_size).cpu().unsqueeze(0).numpy()
                    classifier_input = np.nan_to_num(classifier_input)
                    
                    # Get classifier prediction
                    raw_score = self.classifier.decision_function(classifier_input)[0]
                    raw_score = self._scale_extreme_classifier_output(raw_score)
                    p_redundant = 1.0 / (1.0 + math.exp(-raw_score))
                    
                    # Update PID controller
                    alpha, integral, derivative, error = self._update_pid_controller(
                        p_redundant, alpha, integral, derivative, prev_error
                    )
                    prev_error = error
                    
                    # Reset chunk
                    chunk_hidden = None
                    tokens_in_chunk = 0
                    
                except Exception as e:
                    if debug:
                        print(f"[ERROR] Classifier/PID error: {e}")
                    chunk_hidden = hidden_normalized
                    tokens_in_chunk = 1
            
            # Temperature adjustment and sampling
            temp_ratio = coefficient / self.config.MAX_ALPHA
            temperature = self.config.BASE_TEMP * (1 - temp_ratio) + self.config.STEER_TEMP * temp_ratio
            
            try:
                # Safe sampling
                logits_clamped = logits.clamp(-100, 100)
                probabilities = torch.softmax(logits_clamped / temperature, dim=-1)
                
                if torch.isnan(probabilities).any():
                    probabilities = torch.ones_like(probabilities) / probabilities.size(0)
                
                next_token = torch.multinomial(probabilities, 1).item()
                
            except Exception as e:
                if debug:
                    print(f"[ERROR] Sampling error: {e}, using argmax")
                next_token = torch.argmax(logits).item()
            
            # Update output
            token_str = self.tokenizer.decode([next_token], skip_special_tokens=True).replace("\n", "\\n")
            generated_text += token_str
            output_ids = torch.cat([output_ids, torch.tensor([next_token], device=self.config.DEVICE)])
            
            # Debug output
            if debug:
                print(f"{generation_length:4d} | {int(steering_active):8d} | {p_redundant:5.3f} | {error:5.3f} | "
                      f"{alpha:5.3f} | {integral:5.3f} | {derivative:5.3f} | {temperature:5.3f} | '{token_str}'")
            
            # Check stopping conditions
            if self.stop_pattern.search(generated_text) or "Final answer:" in generated_text:
                if debug:
                    print(f"[INFO] Stop condition met")
                break
        
        pbar.close()
        
        if debug:
            print("--- END TRACE ---")
            print(f"Total tokens generated: {output_ids.size(0) - input_ids.size(0)}")
        
        final_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        tokens_generated = output_ids.size(0) - input_ids.size(0)
        
        return final_text, tokens_generated


class BaselineGenerator:
    """Baseline generator without steering"""
    
    def __init__(self, config, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
    
    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=None):
        """Generate text using baseline model"""
        if max_new_tokens is None:
            max_new_tokens = self.config.MAX_TOKENS
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, tokens_generated


class AnswerExtractor:
    """Utility for extracting numerical answers from generated text"""
    
    @staticmethod
    def extract_boxed_answer(text):
        """Extract the content of the last \\boxed{} occurrence"""
        matches = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
        return matches[-1].group(1).strip() if matches else ""
    
    @staticmethod
    def extract_reference_answer(gsm8k_answer):
        """Extract reference answer from GSM8K format"""
        return gsm8k_answer.split('#### ')[1] if '#### ' in gsm8k_answer else gsm8k_answer


class GSM8KEvaluator:
    """Evaluator for GSM8K mathematical reasoning tasks"""
    
    def __init__(self, config, raspid_generator, baseline_generator):
        self.config = config
        self.raspid_generator = raspid_generator
        self.baseline_generator = baseline_generator
        self.answer_extractor = AnswerExtractor()
    
    def evaluate(self, n_problems, max_tokens=None, debug=False):
        """Evaluate both RASPID and baseline on GSM8K problems"""
        if max_tokens is None:
            max_tokens = self.config.MAX_TOKENS
        
        # Load GSM8K test set (starting from problem 1000)
        gsm8k_test = load_dataset("gsm8k", "main")["test"].select(range(1000, 1000 + n_problems))
        
        results = []
        baseline_total_tokens = 0
        raspid_total_tokens = 0
        
        for example in tqdm(gsm8k_test, desc="Evaluating GSM8K"):
            question = example["question"].strip()
            prompt = f"{question}\n\nAnswer step by step and end with: Final answer: \\boxed{{numeric_value}}"
            
            # Generate with RASPID
            raspid_text, raspid_tokens = self.raspid_generator.generate(prompt, max_tokens, debug)
            
            # Generate with baseline
            baseline_text, baseline_tokens = self.baseline_generator.generate(prompt, max_tokens)
            
            # Extract answers
            baseline_answer = self.answer_extractor.extract_boxed_answer(baseline_text)
            raspid_answer = self.answer_extractor.extract_boxed_answer(raspid_text)
            reference_answer = self.answer_extractor.extract_reference_answer(example["answer"])
            
            results.append({
                "reference_answer": example["answer"],
                "reference_correct": reference_answer,
                "baseline_correct": baseline_answer,
                "raspid_correct": raspid_answer,
                "baseline_tokens": baseline_tokens,
                "raspid_tokens": raspid_tokens,
                "baseline_txt": baseline_text,
                "raspid_txt": raspid_text,
            })
            
            baseline_total_tokens += baseline_tokens
            raspid_total_tokens += raspid_tokens
            
            print(f'Total token usage - Baseline: {baseline_total_tokens}, RASPID: {raspid_total_tokens}')
        
        return pd.DataFrame(results)


def main():
    """Main execution function"""
    print("🚀 Initializing RASPID...")
    
    # Initialize configuration
    config = RASPIDConfig()
    
    # Load models and tokenizer
    print(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=config.DTYPE,
        device_map="auto" if config.DEVICE == "cuda" else None
    ).eval()
    control_model = ControlModel(base_model, [config.EMB_LAYER])
    
    # Load and split labeled data
    print("Loading labeled data...")
    df_all = pd.read_csv(config.LABELED_CSV)
    df_ctrl = df_all.iloc[:1000]  # First 1000 for training
    df_eval = df_all.iloc[1000:1200]  # Last 200 reserved for evaluation
    
    required_thoughts = df_ctrl["required_thoughts"].fillna("")
    redundant_thoughts = df_ctrl["redundant_thoughts"].fillna("")
    
    # Train chunk classifier
    print("Training chunk classifier ...")
    classifier_trainer = ChunkClassifier(config, tokenizer, base_model)
    best_classifier, best_chunk_size = classifier_trainer.train(required_thoughts, redundant_thoughts)
    
    # Build control vector
    print("Building control vector ...")
    vector_builder = ControlVectorBuilder(config, tokenizer, base_model)
    control_vector = vector_builder.build_control_vector(required_thoughts, redundant_thoughts)
    
    # Initialize generators
    raspid_generator = RASPIDGenerator(
        config, tokenizer, base_model, control_model,
        best_classifier, best_chunk_size, control_vector
    )
    baseline_generator = BaselineGenerator(config, tokenizer, base_model)
    
    # Evaluate on GSM8K
    print("🧮 Starting GSM8K evaluation...")
    evaluator = GSM8KEvaluator(config, raspid_generator, baseline_generator)
    results_df = evaluator.evaluate(n_problems=100, debug=False)
    
    # Save results
    results_df.to_csv('results_df_100.csv', index=False)
    print("✅ Results saved to results_df_100.csv")
    
    # Calculate and display metrics
    baseline_accuracy = (results_df['baseline_correct'] == results_df['reference_correct']).mean()
    raspid_accuracy = (results_df['raspid_correct'] == results_df['reference_correct']).mean()
    token_efficiency = (results_df['baseline_tokens'].mean() - results_df['raspid_tokens'].mean()) / results_df['baseline_tokens'].mean()
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"Baseline Accuracy: {baseline_accuracy:.3f}")
    print(f"RASPID Accuracy: {raspid_accuracy:.3f}")
    print(f"Token Savings: {token_efficiency:.3f} ({token_efficiency*100:.1f}%)")


if __name__ == "__main__":
    main()
