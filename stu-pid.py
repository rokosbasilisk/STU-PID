#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install repeng accelerate datasets matplotlib seaborn


# In[2]:


import warnings
warnings.filterwarnings("ignore", "To copy construct from a tensor", UserWarning)


# In[3]:


#!/usr/bin/env python3
"""
RASPID with dynamic PID-steering:
- Train chunk-level classifier and control vector on the first 1000 labeled chains
- Hold out the last 200 chains for final GSM8K evaluation only
"""

import os, re, math, warnings
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from repeng import ControlModel
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── 1) CONFIG & LOAD ──────────────────────────────────────────────────────

MODEL_NAME    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float32

LABELED_CSV   = "gsm8k_chains_labeled_with_tokens.csv"
CTRL_VEC_PATH = "ctrl_vector.pt"

# classifier hyperparams
EMB_LAYER     = 20
CHUNK_SIZES   = [16,24]
BATCH_SIZE    = 32
FLUFF_STAR    = 0.5   # target probability for “redundant”

# PID steering hyperparams
INIT_FREE     = 40
STEER_WINDOW  = 60
KP, KI, KD    = 0.05, 0.001, 0.001
MAX_I, DERIV  = 0.20, 0.01
MAX_ALPHA     = 0.40
BASE_TEMP     = 0.70
STEER_TEMP    = 0.20
MAX_REPEAT    = 8

# load tokenizer & models
tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model   = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE,
    device_map="auto" if DEVICE=="cuda" else None
).eval()
control_model = ControlModel(base_model, [EMB_LAYER])


# In[4]:


# ─── 2) SPLIT LABELED DATA ─────────────────────────────────────────────────

df_all = pd.read_csv(LABELED_CSV)

# first 1000 for training classifier & control vector
df_ctrl = df_all.iloc[:1000]
# last 200 reserved for later (but not used to train classifier)
df_eval = df_all.iloc[1000:1200]

required_ctrl  = df_ctrl["required_thoughts"].fillna("")
redundant_ctrl = df_ctrl["redundant_thoughts"].fillna("")


# In[5]:


import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np

# Disable HuggingFace tokenizer parallel warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ChunkDataset(Dataset):
    def __init__(self, texts, label, cs):
        self.cs = cs
        self.chunks = []
        self.labels = []
        for t in texts:
            tok_ids = tokenizer.encode(t, add_special_tokens=False)
            for i in range(0, len(tok_ids) - cs + 1, cs):
                self.chunks.append(tok_ids[i : i + cs])
                self.labels.append(label)
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        return self.chunks[idx], self.labels[idx]

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    # pad on CPU so pin_memory works
    seqs = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
    padded = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = (padded != tokenizer.pad_token_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask}, torch.tensor(labels, dtype=torch.long)

best_cs, best_acc, best_clf = None, 0.0, None

for cs in CHUNK_SIZES:
    # build datasets
    ds0 = ChunkDataset(required_ctrl, 0, cs)
    ds1 = ChunkDataset(redundant_ctrl, 1, cs)
    loader = DataLoader(
        ConcatDataset([ds0, ds1]),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=0,  # no multiprocessing to avoid fork issues
    )

    # embed all chunks
    feats, labels = [], []
    base_model.eval()
    with torch.no_grad():
        for batch_tokens, batch_labels in tqdm(loader):
            # move to GPU
            batch_tokens = {
                k: v.to(DEVICE, non_blocking=True)
                for k, v in batch_tokens.items()
            }
            out = base_model(**batch_tokens, output_hidden_states=True)
            h = out.hidden_states[EMB_LAYER].mean(1)
            feats.append(h.cpu().numpy())
            labels.append(batch_labels.numpy())
    X = np.vstack(feats)
    y = np.concatenate(labels)

    # train/validation split
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SGD training with tqdm progress
    clf = SGDClassifier(
        loss="log_loss", random_state=42, warm_start=True, max_iter=1, tol=None
    )
    prev_coef = None
    pbar = tqdm(range(500), desc=f"Training clf @ cs={cs}", leave=False)
    for _ in pbar:
        clf.fit(Xtr, ytr)
        coef = clf.coef_
        if prev_coef is not None:
            delta = np.max(np.abs(coef - prev_coef))
            pbar.set_postfix(delta=delta)
            if delta < 1e-3:
                break
        prev_coef = coef.copy()
    pbar.close()

    acc = accuracy_score(yval, clf.predict(Xval))
    print(f"chunk_size={cs} → val_acc={acc:.3f}")
    if acc > best_acc:
        best_cs, best_acc, best_clf = cs, acc, clf

print(f"✔ Selected chunk_size={best_cs}, val_acc={best_acc:.3f}")


# In[6]:


# ─── 4) BUILD CONTROL VECTOR FROM CTRL SET ────────────────────────────────

def mean_hidden(texts):
    vs = []
    for t in tqdm(texts):
        toks = tokenizer(t, return_tensors="pt", truncation=True).to(DEVICE)
        with torch.inference_mode():
            h = base_model(**toks, output_hidden_states=True).hidden_states[EMB_LAYER][0]
        vs.append(h.mean(0).cpu())
    return torch.stack(vs).mean(0)

v_req = mean_hidden(required_ctrl)
v_red = mean_hidden(redundant_ctrl)
ctrl_vec = {EMB_LAYER: (v_req - v_red).to(DEVICE)}
torch.save(ctrl_vec, CTRL_VEC_PATH)
print("✅ Saved control vector")


# In[7]:


from repeng import ControlVector

# model_type is a short string identifying your model, e.g. "qwen" or whatever base_model.config.model_type gives
model_type = base_model.config.model_type  # e.g. "DeepSeek-R1-Distill-Qwen-1.5B"

ctrl_vec = ControlVector(
    model_type = model_type,
    directions={EMB_LAYER: (v_req - v_red).to(DEVICE)}
)

# now you can save it
torch.save(ctrl_vec, "ctrl_vector.pt")


# In[8]:


MAX_TOKENS = 4096


# In[9]:


@torch.inference_mode()
def generate_baseline(prompt, max_new_tokens=MAX_TOKENS):
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = base_model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.6,
        top_p=0.9, repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )
    toks = out.shape[1] - inp.input_ids.shape[1]
    return tokenizer.decode(out[0], skip_special_tokens=True), toks


# In[10]:


# tuned hyper-params
INIT_FREE      = 80          # let the model reason first
STEER_WINDOW   = 60
BASE_TEMP      = 0.60        # same as baseline
STEER_TEMP     = 0.30
STEER_MARGIN   = 0.20        # p_red must exceed 0.5+0.2
MAX_ALPHA      = 0.40
MAX_RAW        = 50.0        # clamp for exp()


# In[11]:


@torch.inference_mode()
def generate_raspid(prompt, max_new_tokens=MAX_TOKENS, debug=True):
    """
    Fixed RASPID generation function that properly applies the control vector
    and handles numerical stability issues.
    """
    if debug:
        print(f"\n=== RASPID GENERATION ===")
        print(f"Original prompt: '{prompt}'")
        print(f"MAX_TOKENS: {MAX_TOKENS}, INIT_FREE: {INIT_FREE}, STEER_WINDOW: {STEER_WINDOW}")
        print(f"Temperatures: BASE_TEMP: {BASE_TEMP}, STEER_TEMP: {STEER_TEMP}")
        print(f"PID: KP={KP}, KI={KI}, KD={KD}, MAX_I={MAX_I}, MAX_ALPHA={MAX_ALPHA}")
        print(f"FLUFF_STAR: {FLUFF_STAR}, EMB_LAYER: {EMB_LAYER}")

    # For structured control vectors (like ControlVector with directions dict),
    # we should leave the structure intact
    ctrl_vec_normalized = ctrl_vec

    # Setup generation
    stop_re = re.compile(r"\\boxed\{[^{}]{1,12}\}")
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE).input_ids[0]
    out_ids = ids.clone()
    past = None
    alpha = I = D = prev_err = 0.0
    chunk_h = None
    tok_in_chunk = 0
    steering = False
    steer_start = 0
    last_tok = None
    rep_ctr = 0
    generated_text = ""

    if debug:
        print("\n--- RASPID TRACE ---")
        print("step | on | p_red |  err  |   α   |   I   |   D   | temp | token")

    pbar = tqdm(range(max_new_tokens), desc="RASPID gen", leave=False)
    for step in pbar:
        gen_len = out_ids.size(0) - ids.size(0)

        # Check if steering should be activated/deactivated
        if not steering and gen_len >= INIT_FREE:
            steering, steer_start = True, gen_len
            if debug:
                print(f"[INFO] Activating steering at gen_len={gen_len}")
        if steering and gen_len - steer_start > STEER_WINDOW:
            if debug:
                print(f"[INFO] Deactivating steering at gen_len={gen_len}")
            steering, alpha, I, D = False, 0.0, 0.0, 0.0

        # Get coefficient and apply control
        coeff = alpha if steering else 0.0
        control_model.set_control(ctrl_vec_normalized, coeff=coeff)

        # CRITICAL FIX: Process the entire prompt on the first step
        if gen_len == 0 and step == 0:
            # First step - process the entire prompt
            out = control_model(
                input_ids=out_ids.unsqueeze(0),  # Use the full prompt
                use_cache=True,
                output_hidden_states=True
            )
        else:
            # Subsequent steps - process only the new token with the cached state
            out = control_model(
                input_ids=out_ids[-1:].unsqueeze(0),  # Only the last token
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True
            )

        past, logits = out.past_key_values, out.logits[0, -1]
        h_last = out.hidden_states[EMB_LAYER][0, -1]

        # Check for NaN values
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits)
        if torch.isnan(h_last).any():
            h_last = torch.nan_to_num(h_last)

        # Check for token repetition
        tok = out_ids[-1].item()
        if tok == last_tok:
            rep_ctr += 1
            if rep_ctr >= MAX_REPEAT:
                if debug:
                    print(f"[INFO] Hit MAX_REPEAT={MAX_REPEAT}, stopping generation")
                break
        else:
            rep_ctr, last_tok = 0, tok

        # Normalize hidden states for stability
        h_last_norm = h_last / (torch.norm(h_last) + 1e-8)

        # Track hidden states with normalized values
        chunk_h = h_last_norm if chunk_h is None else chunk_h + h_last_norm
        tok_in_chunk += 1

        # Classifier and PID controller
        p_red = err = 0.0
        if tok_in_chunk >= best_cs:
            try:
                # Use normalized chunk_h for classifier
                classifier_input = (chunk_h / best_cs).cpu().unsqueeze(0).numpy()

                # Check for NaN values
                if np.isnan(classifier_input).any():
                    classifier_input = np.nan_to_num(classifier_input)

                # Get raw classifier output
                raw = best_clf.decision_function(classifier_input)[0]

                # Scale down extreme classifier values
                if abs(raw) > 50.0:
                    scaled_raw = 50.0 * (np.sign(raw) * np.log(1 + abs(raw) / 50.0) / np.log(1 + abs(raw) / 50.0 * 20))
                    if debug:
                        print(f"[INFO] Scaling down extreme classifier value: {raw:.2f} -> {scaled_raw:.2f}")
                    raw = scaled_raw
                else:
                    raw = max(-50.0, min(50.0, raw))

                p_red = 1.0 / (1.0 + math.exp(-raw))

                if p_red > FLUFF_STAR + 0.20:
                    err = p_red - FLUFF_STAR

                    # Update PID controller
                    I = max(-MAX_I, min(MAX_I, I + KI * err))
                    D = KD * (err - prev_err) + (1 - KD) * D
                    prev_err = err
                    alpha = max(0.0, min(MAX_ALPHA, alpha + KP * err + I + D))

                # Reset chunk
                chunk_h = None
                tok_in_chunk = 0

            except Exception as e:
                if debug:
                    print(f"[ERROR] Error in classifier/PID section: {e}")
                chunk_h = h_last_norm
                tok_in_chunk = 1

        # Temperature and sampling
        temp = BASE_TEMP * (1 - coeff / MAX_ALPHA) + STEER_TEMP * (coeff / MAX_ALPHA)

        # Apply temperature and get probabilities
        try:
            logits_safe = logits.clamp(-100, 100)
            probs = torch.softmax(logits_safe / temp, dim=-1)

            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.size(0)

            # Sample token
            nxt = torch.multinomial(probs, 1).item()
            token_str = tokenizer.decode([nxt], skip_special_tokens=True).replace("\n","\\n")

            # Add token to output
            generated_text += token_str
            out_ids = torch.cat([out_ids, torch.tensor([nxt], device=DEVICE)])

        except Exception as e:
            if debug:
                print(f"[ERROR] Error in sampling: {e}")
            # Fallback to argmax sampling
            nxt = torch.argmax(logits).item()
            token_str = tokenizer.decode([nxt], skip_special_tokens=True).replace("\n","\\n")
            generated_text += token_str
            out_ids = torch.cat([out_ids, torch.tensor([nxt], device=DEVICE)])

        # Print trace line if in debug mode
        if debug:
            print(f"{gen_len:4d} | {int(steering)} | {p_red:5.3f} | {err:5.3f} | "
                  f"{alpha:5.3f} | {I:5.3f} | {D:5.3f} | {temp:5.3f} | '{token_str}'")

        # Check stop condition
        if stop_re.search(generated_text) or "Final answer:" in generated_text:
            if debug:
                print(f"[INFO] Stop condition met, ending generation")
            break

    pbar.close()
    if debug:
        print("--- END TRACE ---\n")
        print(f"=== GENERATION RESULT ===")
        print(f"Total tokens generated: {out_ids.size(0) - ids.size(0)}")
        print(f"Final text: {tokenizer.decode(out_ids, skip_special_tokens=True)}")

    return tokenizer.decode(out_ids, skip_special_tokens=True), out_ids.size(0) - ids.size(0)


# In[12]:


def norm_answer(s: str) -> str:
    """
    Extracts the content of the last \boxed{} occurrence in the string.
    This handles cases where the model might have multiple boxed answers,
    ensuring we get the final one.

    Args:
        s: The string to extract the answer from

    Returns:
        The content inside the last \boxed{} occurrence, or empty string if none found
    """
    # Find all occurrences of \boxed{...}
    matches = list(re.finditer(r"\\boxed\{([^}]+)\}", s))

    # Return the last match, if any
    if matches:
        return matches[-1].group(1).strip()
    else:
        return ""


# In[13]:


def run_gsm8k(n_probs, max_tokens, debug = False):
    gsm = load_dataset("gsm8k", "main")["test"].select(range(1000,1000+n_probs))
    rec = []
    baseline_total = 0
    raspid_total = 0
    for ex in tqdm(gsm):
        q   = ex["question"].strip()
        prompt = f"{q}\n\nAnswer step by step and end with: Final answer: \\boxed{{numeric_value}}"
        r_txt,r_tok = generate_raspid(prompt, max_tokens, debug = debug)
        b_txt,b_tok = generate_baseline(prompt, max_tokens)


        rec.append({
            "reference_answer":ex["answer"],
            "baseline_correct": norm_answer(b_txt),
            "raspid_correct":  norm_answer(r_txt),
            "baseline_tokens": b_tok,
            "raspid_tokens":  r_tok,
            "baseline_txt": b_txt,
            "raspid_txt":  r_txt,

        })
        baseline_total += int(b_tok)
        raspid_total += int(r_tok)
        print(f'total-token-usage for baseline: {baseline_total} raspid: {raspid_total}')

    df = pd.DataFrame(rec)
    return df


# In[14]:


seal_results = pd.read_csv('seal_results.csv')    
seal_results.head()


# In[17]:


seal_df = pd.read_csv('seal_results.csv')

# baseline_tokens = []
# seal_tokens = []

# for i in tqdm(range(len(seal_df)), desc="Counting tokens"):
#     baseline_tokens.append(len(tokenizer(seal_df.iloc[i]['baseline_generation'])['input_ids']))
#     seal_tokens.append(len(tokenizer(seal_df.iloc[i]['seal_generation'])['input_ids']))

# seal_df['baseline_tokens'] = baseline_tokens
# seal_df['seal_tokens'] = seal_tokens


# In[18]:


# seal_df.to_csv('seal_results.csv')
seal_df['baseline_tokens'].median(),seal_df['seal_tokens'].median()


# In[19]:


def run_seal(n_probs, max_tokens, debug = False):

    seal_results = seal_df.iloc[:n_probs]

    extract_prompt = lambda q: q.split('<｜User｜>')[1].split('<｜Assistant｜>')[0].strip()

    rec = []
    baseline_total = 0
    raspid_total = 0

    for idx,ex in tqdm(seal_results.iterrows()):

        prompt = extract_prompt(ex["prompt"])
        ref_answer = ex["answer"]

        baseline_txt = str(eval(ex['baseline_generation'])[0])
        seal_txt = str(eval(ex['seal_generation'])[0])
        raspid_txt,raspid_tokens = generate_raspid(prompt, max_tokens, debug = debug)

        baseline_ans = eval(ex['all_pred_baseline'])[0]
        seal_ans = eval(ex['all_pred_seal'])[0]
        raspid_ans = norm_answer(raspid_txt)


        baseline_tokens = ex['baseline_tokens']
        seal_tokens = ex['seal_tokens']


        rec.append({
            "prompt":prompt,
            "reference_answer":ref_answer,


            "baseline_txt": baseline_txt,
            "seal_txt":  seal_txt,
            "raspid_txt":  raspid_txt,


            "baseline_answer": baseline_ans,
            "seal_answer":  seal_ans,
            "raspid_answer":  raspid_ans,


            "baseline_tokens": baseline_tokens,
            "seal_tokens": seal_tokens,
            "raspid_tokens":  raspid_tokens,

        })
        print(f"baseline_tokens: {baseline_tokens}, seal_tokens: {seal_tokens}, raspid_tokens: {raspid_tokens}")

    df = pd.DataFrame(rec)
    return df


# In[21]:


# results_df = run_seal(5, 4096*4)


# In[22]:


results_df = run_gsm8k(50, 4096)
results_df.to_csv('results_df_50.csv')


# In[33]:


gsm_subset = load_dataset("gsm8k", "main")["test"].select(range(1000,1000+50))


# In[41]:


reference_answers = [gsm_subset['answer'][idx].split('#### ')[1] for idx in range(50)]


# In[42]:


results_df['reference_correct'] = reference_answers


# In[63]:


baseline_acc = len(results_df[results_df['baseline_correct'] == results_df['reference_correct']])/len(results_df)
raspid_acc = len(results_df[results_df['raspid_correct'] == results_df['reference_correct']])/len(results_df)
token_eff = (results_df['baseline_tokens'].mean() - results_df['raspid_tokens'].mean())/results_df['baseline_tokens'].mean()


# In[64]:


print(f'baseline-acc: {baseline_acc} raspid-acc: {raspid_acc} token-saving: {token_eff}')


# In[25]:


## raspid works better than rasped fixed v2 in 66% of the cases with 52% token saving.

