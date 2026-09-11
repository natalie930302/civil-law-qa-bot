"""
驗證「資料量門檻」假設的另一半:之前用 LoRA(只調0.13%參數)解決了全參數微調
在139條語料上的災難性遺忘(MRR 0.555->0.218 全參數微調 vs 0.544 LoRA)。但還沒
測過:如果單純把語料量放大,全參數微調本身會不會就不再崩壞?這支腳本用
872條語料(原本139條 + 從 Hugging Face lianghsun/tw-law 資料集抓來的23部相關
公務人員法規,共733條新增,官方同步自法務部全國法規資料庫)重跑一次全參數
TSDAE微調,看資料量從139條增加到872條(6.3倍),是否足以讓全參數微調不再破壞
檢索品質。

評估問題集不變(24題),但檢索的語料庫改成872條(確保24題的gold_id全部都在
擴充後的語料庫裡,不會因為換語料庫而讓某些題目變成不可能答對)。
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

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

with open("../eval/expanded_articles.json", encoding="utf-8") as f:
    articles = json.load(f)
with open("../eval/eval_questions.json", encoding="utf-8") as f:
    eval_qs = json.load(f)

corpus_ids = [a["id"] for a in articles]
corpus_texts = [a["text"] for a in articles]
id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
queries = [q["question"] for q in eval_qs]
gold_ids = [q["gold_id"] for q in eval_qs]

print(f"語料庫大小: {len(articles)} 條(原本139條 + 擴充語料)")


def evaluate(model, name):
    corpus_emb = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)
    query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, corpus_emb, top_k=10)

    recall_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    for i, hit_list in enumerate(hits):
        gold_idx = id_to_idx.get(gold_ids[i])
        ranks = [h["corpus_id"] for h in hit_list]
        if gold_idx in ranks:
            rank_pos = ranks.index(gold_idx) + 1
            rr_sum += 1.0 / rank_pos
            for k in recall_at:
                if rank_pos <= k:
                    recall_at[k] += 1
    n = len(queries)
    print(f"\n=== {name} ===")
    for k in recall_at:
        print(f"Recall@{k}: {recall_at[k]}/{n} = {recall_at[k]/n:.3f}")
    print(f"MRR: {rr_sum/n:.3f}")
    return {"recall@1": recall_at[1]/n, "recall@3": recall_at[3]/n, "recall@5": recall_at[5]/n, "mrr": rr_sum/n}


print("載入 baseline 模型...")
base_model = SentenceTransformer(BASE_MODEL)
baseline_result = evaluate(base_model, f"Baseline(未微調,語料庫{len(articles)}條)")

print(f"\n開始全參數 TSDAE 微調(用{len(articles)}條語料,原本是139條的6.3倍)...")
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import sentence_transformers.losses as losses

train_dataset = DenoisingAutoEncoderDataset(corpus_texts)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
ft_model = SentenceTransformer(BASE_MODEL)
train_loss = losses.DenoisingAutoEncoderLoss(ft_model, decoder_name_or_path=BASE_MODEL, tie_encoder_decoder=True)

ft_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

expanded_result = evaluate(ft_model, f"全參數TSDAE微調後(語料庫{len(articles)}條)")

print("\n=== 三方結果比較(原本139條全參數微調 vs. 872條全參數微調 vs. baseline) ===")
try:
    with open("../eval/results.json", encoding="utf-8") as f:
        original_139_result = json.load(f)["tsdae_finetuned"]
except FileNotFoundError:
    original_139_result = None

for k in ["recall@1", "recall@3", "recall@5", "mrr"]:
    line = f"{k}: baseline={baseline_result[k]:.3f}  872條全參數微調={expanded_result[k]:.3f}"
    if original_139_result:
        line += f"  139條全參數微調(舊)={original_139_result[k]:.3f}"
    print(line)

with open("../eval/results_expanded_corpus.json", "w", encoding="utf-8") as f:
    json.dump({
        "corpus_size": len(articles),
        "baseline": baseline_result,
        "full_finetuned_expanded_corpus": expanded_result,
        "full_finetuned_original_139_for_reference": original_139_result,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_expanded_corpus.json")
