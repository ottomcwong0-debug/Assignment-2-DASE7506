"""
High-Precision Disagreement Verifier Training (Target >80% on Hard Test)
"""
import ast
import operator
import pathlib
import pickle
import re
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from common import DATA, load_inputs, read_jsonl, parse_answer

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


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
    """升級版算式檢查：支援複合四則運算 (如 30 + 2 * 20 = 70) 與貨幣符號清理"""
    t = str(text)
    t = t.replace('\\times', '*').replace('×', '*').replace('÷', '/').replace('\\div', '/')
    t = t.replace('$', '').replace(',', '')
    
    # 匹配 形如 expr = num 的算式片段
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
            # 最終答案是否也有出現在其他不同答案候選人的中間推理過程裡
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


def build_dataset(split):
    bank, arrays = load_inputs(split, DATA)
    gold_rows = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    embeddings = arrays['embeddings']
    
    X_all, y_all = [], []
    disagree_records = []
    
    for idx, (row, gold_row) in enumerate(zip(bank, gold_rows)):
        gold_ans = parse_answer(gold_row['answer'])
        feats, parsed_ans = extract_row_features(row, embeddings[idx])
        valid_ans = [a for a in parsed_ans if a is not None]
        unique_valid = set(valid_ans)
        
        is_disagree = (len(unique_valid) > 1) or (len(valid_ans) < 4)
        labels = [1 if (a is not None and a == gold_ans) else 0 for a in parsed_ans]
        
        if is_disagree and sum(labels) > 0:
            for c_idx in range(4):
                X_all.append(feats[c_idx])
                y_all.append(labels[c_idx])
            disagree_records.append((feats, parsed_ans, gold_ans))
            
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int32), disagree_records


def make_compact_features(X, pca):
    """提取 PCA 降維後的語意特徵 + 所有純量與相對特徵，供樹模型與神經網路高效學習"""
    emb_part = X[:, :2048]
    scalar_part = X[:, 2048:]
    pca_part = pca.transform(emb_part)
    return np.hstack([pca_part, scalar_part])


def predict_with_params(probs_4, parsed_ans, params):
    valid_ans = [a for a in parsed_ans if a is not None]
    if not valid_ans:
        return int(np.argmax(probs_4))
        
    counts = Counter(valid_ans)
    if len(counts) == 1:
        only_ans = next(iter(counts))
        for i, a in enumerate(parsed_ans):
            if a == only_ans:
                return i
                
    # 組內機率正規化 (使每題 4 個候選人的機率總和一致，提升跨題型穩定度)
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


def fit_models(X, y):
    pca = PCA(n_components=48, random_state=42)
    pca.fit(X[:, :2048])
    X_compact = make_compact_features(X, pca)
    
    lr1 = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=2000, class_weight='balanced', random_state=42)
    )
    lr2 = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.10, max_iter=2000, class_weight='balanced', random_state=42)
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.04, l2_regularization=3.0, min_samples_leaf=25, random_state=42
    )
    et = ExtraTreesClassifier(
        n_estimators=250, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.5, max_iter=500, early_stopping=True, random_state=42)
    )
    
    lr1.fit(X, y)
    lr2.fit(X, y)
    hgb.fit(X_compact, y)
    et.fit(X_compact, y)
    mlp.fit(X_compact, y)
    
    return {'pca': pca, 'lr1': lr1, 'lr2': lr2, 'hgb': hgb, 'et': et, 'mlp': mlp}


def get_ensemble_probs(models, feats, weights):
    w_lr, w_hgb, w_et, w_mlp = weights
    feats_compact = make_compact_features(feats, models['pca'])
    p_lr = 0.5 * models['lr1'].predict_proba(feats)[:, 1] + 0.5 * models['lr2'].predict_proba(feats)[:, 1]
    p_hgb = models['hgb'].predict_proba(feats_compact)[:, 1]
    p_et = models['et'].predict_proba(feats_compact)[:, 1]
    p_mlp = models['mlp'].predict_proba(feats_compact)[:, 1]
    return w_lr * p_lr + w_hgb * p_hgb + w_et * p_et + w_mlp * p_mlp


def main():
    print("1. 正在載入並提取 Train 與 Validation 的歧異樣本特徵...")
    X_tr, y_tr, rec_tr = build_dataset('train')
    X_val, y_val, rec_val = build_dataset('validation')
    print(f"Train 歧異題數: {len(rec_tr)} | Val 歧異題數: {len(rec_val)}")
    
    print("2. 訓練驗證模型並搜尋最佳混合超參數...")
    val_models = fit_models(X_tr, y_tr)
    
    # 預先計算各基礎模型在 Validation 上的輸出
    cached_val = []
    for feats, _, _ in rec_val:
        fc = make_compact_features(feats, val_models['pca'])
        p_lr = 0.5 * val_models['lr1'].predict_proba(feats)[:, 1] + 0.5 * val_models['lr2'].predict_proba(feats)[:, 1]
        p_hgb = val_models['hgb'].predict_proba(fc)[:, 1]
        p_et = val_models['et'].predict_proba(fc)[:, 1]
        p_mlp = val_models['mlp'].predict_proba(fc)[:, 1]
        cached_val.append((p_lr, p_hgb, p_et, p_mlp))
        
    weight_candidates = [
        (0.45, 0.30, 0.15, 0.10),
        (0.50, 0.25, 0.15, 0.10),
        (0.35, 0.35, 0.15, 0.15),
        (0.60, 0.25, 0.15, 0.00),
    ]
    
    best_acc = -1.0
    best_weights = weight_candidates[0]
    best_params = (0.5, 0.5, 0.65, 0.30, 0.45, 0.0)
    
    for w_tuple in weight_candidates:
        w_lr, w_hgb, w_et, w_mlp = w_tuple
        for w_31 in [0.45, 0.60, 0.75, 0.95]:
            for w_211 in [0.18, 0.28, 0.38]:
                for w_21 in [0.30, 0.45, 0.60]:
                    params = (0.5, 0.5, w_31, w_211, w_21, 0.0)
                    corr = 0
                    for (p_lr, p_hgb, p_et, p_mlp), (_, parsed_ans, gold_ans) in zip(cached_val, rec_val):
                        p_blend = w_lr * p_lr + w_hgb * p_hgb + w_et * p_et + w_mlp * p_mlp
                        idx = predict_with_params(p_blend, parsed_ans, params)
                        if parsed_ans[idx] == gold_ans:
                            corr += 1
                    acc = corr / len(rec_val)
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = w_tuple
                        best_params = params
                        
    print(f"Validation 歧異子集最佳準確率: {best_acc*100:.2f}% | Weights: {best_weights} | Params: {best_params}")
    
    print("3. 合併 Train + Validation 所有歧異樣本訓練最終模型套件...")
    X_full = np.vstack([X_tr, X_val])
    y_full = np.concatenate([y_tr, y_val])
    full_models = fit_models(X_full, y_full)
    
    bundle = {
        'models': full_models,
        'weights': best_weights,
        'params': best_params
    }
    
    output_dir = pathlib.Path("runs/model_bundle")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"最終模型套件已儲存至: {model_path}")


if __name__ == "__main__":
    main()
