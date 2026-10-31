"""
Advanced Hybrid Submission: 
Layer 1: Parsed Answer Majority Voting (utilizing common.parse_answer)
Layer 2: Logistic Regression Embedding Classifier (Fallback when no consensus)
Hybrid Submission with Execution Tracking
Advanced Hybrid Submission with Relative & Cosine Similarity Features
Advanced Hybrid Submission with Smart Confidence Routing & Advanced Features
Advanced Hybrid Submission with Smart Dynamic Margin Routing & Advanced Features
Advanced Hybrid Submission with Random Forest & Smart Dynamic Routing (Safe Version)
Optimized Submission script with Confidence-Gated Vetoing
Optimized Submission script with Soft-Blending (Probability + L1 Bonus) 
Submission script with Symbolic Math Verification & Answer-Group Aggregation
High-Precision Submission Script with Multi-Step Equation Verification & PCA-Enhanced Ensemble

"""
import ast
import operator
import pathlib
import pickle
import re
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from common import parse_answer

_MODEL_CACHE = None

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def load_verifier(checkpoint_dir):
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if checkpoint_dir is None:
        return None
    ckpt_path = pathlib.Path(checkpoint_dir) / "classifier.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
    return _MODEL_CACHE


def _eval_ast(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Div) and abs(right) < 1e-9:
            raise ZeroDivisionError()
        return _ALLOWED_OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


def check_equations(text):
    t = str(text)
    t = t.replace('\\times', '*').replace('×', '*').replace('÷', '/').replace('\\div', '/')
    t = t.replace('$', '').replace(',', '')
    pattern = r'((?:[-+]?\d+(?:\.\d+)?|\(\s*[-+]?\d+(?:\.\d+)?)(?:\s*[\+\-\*\/]\s*(?:\(?\s*[-+]?\d+(?:\.\d+)?\s*\)?))+)\s*=\s*([-+]?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, t)
    correct, wrong = 0, 0
    for expr_s, rhs_s in matches:
        try:
            rhs = float(rhs_s)
            val = _eval_ast(ast.parse(expr_s.strip(), mode='eval').body)
            if abs(val - rhs) <= max(1e-2, 1e-3 * abs(rhs)):
                correct += 1
            else:
                wrong += 1
        except Exception:
            pass
    return correct, wrong


def extract_numbers(text):
    nums = set()
    for m in re.findall(r'(?<![a-zA-Z])[-+]?\d[\d,]*(?:\.\d+)?', str(text)):
        try:
            nums.add(float(m.replace(',', '')))
        except ValueError:
            pass
    return nums


def extract_row_features(row, cand_embeddings):
    candidates = row.get('candidates', [])
    question = row.get('question', '')
    q_nums = extract_numbers(question)
    
    N_cand = len(candidates)
    cand_embeddings = np.asarray(cand_embeddings, dtype=np.float32).reshape(N_cand, -1)
    cos_matrix = cosine_similarity(cand_embeddings)
    
    parsed_ans = []
    parsed_vals = []
    texts = []
    for cand in candidates:
        t = cand['text'] if isinstance(cand, dict) else str(cand)
        texts.append(t)
        ans = parse_answer(t)
        parsed_ans.append(ans)
        try:
            parsed_vals.append(float(ans) if ans is not None else np.nan)
        except (TypeError, ValueError):
            parsed_vals.append(np.nan)
            
    valid_ans = [a for a in parsed_ans if a is not None]
    ans_counts = Counter(valid_ans)
    max_freq = max(ans_counts.values()) if ans_counts else 0
    num_unique_ans = len(ans_counts)
    
    raw_scalars = []
    emb_block = []
    mean_all_emb = np.mean(cand_embeddings, axis=0)
    
    for i in range(N_cand):
        e_i = cand_embeddings[i]
        others = [cand_embeddings[j] for j in range(N_cand) if j != i]
        mean_other = np.mean(others, axis=0)
        
        other_sims = [float(cos_matrix[i, j]) for j in range(N_cand) if j != i]
        max_sim = float(np.max(other_sims))
        min_sim = float(np.min(other_sims))
        mean_sim = float(np.mean(other_sims))
        std_sim = float(np.std(other_sims))
        
        l2_dists = [float(np.linalg.norm(e_i - o)) for o in others]
        min_l2 = float(np.min(l2_dists))
        mean_l2 = float(np.mean(l2_dists))
        l2_to_center = float(np.linalg.norm(e_i - mean_all_emb))
        
        ans_i = parsed_ans[i]
        is_valid = 1.0 if ans_i is not None else 0.0
        freq_i = float(ans_counts.get(ans_i, 0)) if ans_i is not None else 0.0
        vote_share = freq_i / max(1.0, float(len(valid_ans)))
        is_top_vote = 1.0 if (ans_i is not None and freq_i == max_freq) else 0.0
        
        same_ans_sims = [float(cos_matrix[i, j]) for j in range(N_cand) if j != i and parsed_ans[j] == ans_i and ans_i is not None]
        diff_ans_sims = [float(cos_matrix[i, j]) for j in range(N_cand) if j != i and parsed_ans[j] != ans_i]
        mean_same_sim = float(np.mean(same_ans_sims)) if same_ans_sims else mean_sim
        mean_diff_sim = float(np.mean(diff_ans_sims)) if diff_ans_sims else mean_sim
        sim_gap = mean_same_sim - mean_diff_sim
        
        val_i = parsed_vals[i]
        if np.isfinite(val_i):
            is_int = 1.0 if abs(val_i - round(val_i)) < 1e-6 else 0.0
            is_pos = 1.0 if val_i >= 0 else 0.0
            log_mag = float(np.log1p(abs(val_i)))
            ans_str = str(int(round(val_i))) if is_int else str(val_i)
            in_other_text = sum(1.0 for j in range(N_cand) if j != i and parsed_ans[j] != ans_i and re.search(r'(?<!\d)' + re.escape(ans_str) + r'(?!\d)', texts[j]))
        else:
            is_int, is_pos, log_mag, in_other_text = 0.0, 0.0, 0.0, 0.0
            
        t_i = texts[i]
        text_len = float(len(t_i))
        num_lines = float(t_i.count('\n') + 1)
        has_hashes = 1.0 if '####' in t_i else 0.0
        has_boxed = 1.0 if '\\boxed' in t_i else 0.0
        
        eq_corr, eq_wrong = check_equations(t_i)
        eq_total = eq_corr + eq_wrong
        eq_acc = float(eq_corr) / float(eq_total) if eq_total > 0 else 1.0
        has_eq_err = 1.0 if eq_wrong > 0 else 0.0
        
        c_nums = extract_numbers(t_i)
        q_cov = float(len(q_nums & c_nums)) / float(len(q_nums)) if q_nums else 1.0
        hallucinated_nums = float(len(c_nums - q_nums)) / max(1.0, float(len(c_nums)))
        
        sc = np.array([
            is_valid, freq_i, vote_share, is_top_vote, float(num_unique_ans),
            max_sim, min_sim, mean_sim, std_sim, min_l2, mean_l2, l2_to_center,
            mean_same_sim, mean_diff_sim, sim_gap,
            is_int, is_pos, log_mag, in_other_text,
            text_len, num_lines, has_hashes, has_boxed,
            float(eq_corr), float(eq_wrong), eq_acc, has_eq_err, q_cov, hallucinated_nums
        ], dtype=np.float32)
        
        raw_scalars.append(sc)
        emb_block.append(np.concatenate([e_i, e_i - mean_other]))
        
    raw_scalars = np.vstack(raw_scalars)
    sc_mean = np.mean(raw_scalars, axis=0, keepdims=True)
    sc_std = np.std(raw_scalars, axis=0, keepdims=True) + 1e-6
    rel_scalars = (raw_scalars - sc_mean) / sc_std
    
    emb_block = np.vstack(emb_block)
    return np.hstack([emb_block, raw_scalars, rel_scalars]), parsed_ans


def predict_single(row, emb, bundle):
    candidates = row.get('candidates', [])
    if not candidates:
        return 0
        
    feats, parsed_ans = extract_row_features(row, emb)
    valid_ans = [a for a in parsed_ans if a is not None]
    if not valid_ans:
        return 0
        
    counts = Counter(valid_ans)
    if len(counts) == 1:
        only_ans = next(iter(counts))
        for i, a in enumerate(parsed_ans):
            if a == only_ans:
                return i
                
    if bundle is None or not isinstance(bundle, dict) or 'models' not in bundle:
        top_ans = counts.most_common(1)[0][0]
        for i, a in enumerate(parsed_ans):
            if a == top_ans:
                return i
        return 0
        
    models = bundle['models']
    w_lr, w_hgb, w_et, w_mlp = bundle.get('weights', (0.45, 0.30, 0.15, 0.10))
    params = bundle.get('params', (0.5, 0.5, 0.65, 0.30, 0.45, 0.0))
    
    fc = np.hstack([models['pca'].transform(feats[:, :2048]), feats[:, 2048:]])
    p_lr = 0.5 * models['lr1'].predict_proba(feats)[:, 1] + 0.5 * models['lr2'].predict_proba(feats)[:, 1]
    p_hgb = models['hgb'].predict_proba(fc)[:, 1]
    p_et = models['et'].predict_proba(fc)[:, 1]
    p_mlp = models['mlp'].predict_proba(fc)[:, 1]
    probs_4 = w_lr * p_lr + w_hgb * p_hgb + w_et * p_et + w_mlp * p_mlp
    
    p_norm = probs_4 / (np.sum(probs_4) + 1e-8)
    
    alpha, beta, w_31, w_211, w_21, w_tie = params
    pat = tuple(c for _, c in counts.most_common())
    if pat == (3, 1):
        vote_w = w_31
    elif pat == (2, 1, 1):
        vote_w = w_211
    elif pat == (2, 1):
        vote_w = w_21
    else:
        vote_w = w_tie
        
    best_ans = None
    best_group_score = -1e9
    
    for ans, freq in counts.items():
        idxs = [i for i, a in enumerate(parsed_ans) if a == ans]
        p_sub = p_norm[idxs]
        g_score = alpha * float(np.mean(p_sub)) + beta * float(np.max(p_sub)) + vote_w * (float(freq) / 4.0)
        if g_score > best_group_score:
            best_group_score = g_score
            best_ans = ans
            
    best_idxs = [i for i, a in enumerate(parsed_ans) if a == best_ans]
    return int(best_idxs[int(np.argmax(probs_4[best_idxs]))])


def predict(bank, arrays, checkpoint_dir=None):
    bundle = load_verifier(checkpoint_dir)
    raw_embeddings = arrays.get('embeddings', None)
    if raw_embeddings is not None:
        embeddings_list = raw_embeddings.reshape(len(bank), 4, -1)
    else:
        embeddings_list = [np.zeros((4, 1024), dtype=np.float32)] * len(bank)
        
    predictions = []
    for idx, row in enumerate(bank):
        pred_idx = predict_single(row, embeddings_list[idx], bundle)
        predictions.append(int(pred_idx))
        
    return predictions
