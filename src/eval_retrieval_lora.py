"""
驗證一個具體假設:eval_retrieval.py 裡「TSDAE全參數微調讓檢索大幅變差」(MRR 0.555->0.218),
理論上符合 catastrophic forgetting(災難性遺忘)——在只有139條語料的小資料集上做全參數微調,
訓練目標(去噪重建)跟下游任務(語義檢索)不一致,足以蓋掉/破壞原本在大規模語料上學到的
通用多語語義空間。

這裡用 LoRA(Low-Rank Adaptation)重做同一個TSDAE微調實驗:凍結原本的BERT權重,只在
attention 的 query/value 投影矩陣插入小型低秩adapter做訓練,可訓練參數量大幅減少。
如果 catastrophic forgetting 假設成立,LoRA版本應該比全參數微調保留更多原本的檢索能力
(不一定會贏過baseline,但應該明顯優於全參數微調的0.218)。

這個實驗方法對應到目前機器學習裡真實、當紅的研究方向:LoRA/PEFT(參數效率微調)之所以
被廣泛採用,其中一個核心理由正是避免全參數微調在小資料下的災難性遺忘問題。
"""
import os
os.environ["USE_TF"] = "0"
import transformers.integrations.integration_utils as _integration_utils
_integration_utils.is_tensorboard_available = lambda: False
from sentence_transformers.trainer import SentenceTransformerTrainer as _STTrainer
_orig_compute_loss = _STTrainer.compute_loss
def _compute_loss_compat(self, model, inputs, return_outputs=False, **kwargs):
    return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)
_STTrainer.compute_loss = _compute_loss_compat

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

with open("../eval/articles_clean.json", encoding="utf-8") as f:
    articles = json.load(f)
with open("../eval/eval_questions.json", encoding="utf-8") as f:
    eval_qs = json.load(f)

corpus_ids = [a["id"] for a in articles]
corpus_texts = [a["text"] for a in articles]
id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
queries = [q["question"] for q in eval_qs]
gold_ids = [q["gold_id"] for q in eval_qs]


def evaluate(model, name):
    corpus_emb = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, corpus_emb, top_k=10)
    recall_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    n = len(queries)
    for i, hit_list in enumerate(hits):
        gold_idx = id_to_idx.get(gold_ids[i])
        ranks = [h["corpus_id"] for h in hit_list]
        if gold_idx in ranks:
            rank_pos = ranks.index(gold_idx) + 1
            rr_sum += 1.0 / rank_pos
            for k in recall_at:
                if rank_pos <= k:
                    recall_at[k] += 1
    print(f"\n=== {name} ===")
    for k in recall_at:
        print(f"Recall@{k}: {recall_at[k]}/{n} = {recall_at[k]/n:.3f}")
    print(f"MRR: {rr_sum/n:.3f}")
    return {"recall@1": recall_at[1]/n, "recall@3": recall_at[3]/n, "recall@5": recall_at[5]/n, "mrr": rr_sum/n}


print("載入 baseline 模型...")
base_model = SentenceTransformer(BASE_MODEL)
baseline_result = evaluate(base_model, "Baseline (未微調)")

print("\n套用 LoRA(凍結原本權重,只在attention query/value插入低秩adapter)...")
ft_model = SentenceTransformer(BASE_MODEL)
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
)
ft_model[0].auto_model = get_peft_model(ft_model[0].auto_model, lora_config)
trainable = sum(p.numel() for p in ft_model[0].auto_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in ft_model[0].auto_model.parameters())
print(f"可訓練參數: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import sentence_transformers.losses as losses

train_dataset = DenoisingAutoEncoderDataset(corpus_texts)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
train_loss = losses.DenoisingAutoEncoderLoss(ft_model, decoder_name_or_path=BASE_MODEL, tie_encoder_decoder=False)

ft_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},  # 第一次嘗試用 3e-4(LoRA常見的較高學習率)導致embedding整個崩潰
    # (任兩篇不同法規條文的cosine相似度高達0.9996,等於失去所有語義區分能力)——
    # 這說明「LoRA可以用更高學習率」的經驗法則在極小資料集(僅139筆)的無監督重建任務上不成立,
    # 改用跟全參數微調一樣的學習率重跑,隔離出「學習率過高」和「LoRA本身」兩個變因
    show_progress_bar=True,
)

debug_emb = ft_model.encode(corpus_texts[:5], convert_to_numpy=True, show_progress_bar=False)
print(f"\n[除錯] 微調後embedding是否含NaN: {np.isnan(debug_emb).any()}, 是否含Inf: {np.isinf(debug_emb).any()}")
print(f"[除錯] embedding norm(前5筆): {np.linalg.norm(debug_emb, axis=1)}")
print(f"[除錯] 前兩筆embedding的cosine相似度: {np.dot(debug_emb[0], debug_emb[1]) / (np.linalg.norm(debug_emb[0]) * np.linalg.norm(debug_emb[1]) + 1e-9):.4f}")

lora_result = evaluate(ft_model, "TSDAE + LoRA 微調後")

print("\n=== 三方結果比較 ===")
try:
    with open("../eval/results.json", encoding="utf-8") as f:
        full_ft_result = json.load(f)["tsdae_finetuned"]
except FileNotFoundError:
    full_ft_result = None

for k in ["recall@1", "recall@3", "recall@5", "mrr"]:
    line = f"{k}: baseline={baseline_result[k]:.3f}  LoRA微調={lora_result[k]:.3f}"
    if full_ft_result:
        line += f"  全參數微調={full_ft_result[k]:.3f}"
    print(line)

with open("../eval/results_lora.json", "w", encoding="utf-8") as f:
    json.dump({
        "baseline": baseline_result,
        "lora_finetuned": lora_result,
        "full_finetuned_for_reference": full_ft_result,
        "lora_trainable_params": trainable,
        "lora_total_params": total,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_lora.json")
