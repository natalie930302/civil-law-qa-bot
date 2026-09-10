"""
評估 RAG 檢索品質:baseline pretrained embedding vs. TSDAE 領域微調後的 embedding。
指標: Recall@1 / Recall@3 / Recall@5 / MRR

TSDAE 是無監督微調法(只需要語料本身,不需要人工標注的問答配對),
用 139 條清理過的法規條文當語料微調,24 題人工撰寫的問題當測試集(完全不重疊,避免資料洩漏)。
"""
import os
os.environ["USE_TF"] = "0"
# 環境內的 tensorboard 會誤觸發到壞掉的 tensorflow(AttributeError: module 'tensorflow' has no attribute 'io'),
# 而 transformers Trainer 預設會自動偵測並加入 TensorBoardCallback,與 USE_TF=0 無關,需另外關閉
import transformers.integrations.integration_utils as _integration_utils
_integration_utils.is_tensorboard_available = lambda: False
# 安裝的 sentence-transformers(3.2.1)跟 transformers(4.46.3)版本沒對齊:
# transformers 的 Trainer.training_step 會多傳 num_items_in_batch 給 compute_loss,
# 但 sentence-transformers 這個版本的 compute_loss 還不接受這個參數,直接呼叫會 TypeError
from sentence_transformers.trainer import SentenceTransformerTrainer as _STTrainer
_orig_compute_loss = _STTrainer.compute_loss
def _compute_loss_compat(self, model, inputs, return_outputs=False, **kwargs):
    return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)
_STTrainer.compute_loss = _compute_loss_compat
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util, InputExample
import sys

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

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
baseline_result = evaluate(base_model, "Baseline (未微調)")

print("\n開始 TSDAE 領域微調(只用139條法規條文語料,無監督)...")
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

tsdae_result = evaluate(ft_model, "TSDAE 領域微調後")

print("\n=== 結果比較 ===")
for k in ["recall@1", "recall@3", "recall@5", "mrr"]:
    b, t = baseline_result[k], tsdae_result[k]
    diff = t - b
    print(f"{k}: baseline={b:.3f}  微調後={t:.3f}  差異={diff:+.3f}")

ft_model.save("../eval/tsdae_finetuned_model")
print("\n已儲存微調後模型至 eval/tsdae_finetuned_model")

with open("../eval/results.json", "w", encoding="utf-8") as f:
    json.dump({"baseline": baseline_result, "tsdae_finetuned": tsdae_result}, f, ensure_ascii=False, indent=2)
