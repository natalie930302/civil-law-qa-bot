"""
消融實驗:把「語料量」跟「訓練step數」這兩個混淆變數拆開。

上一支腳本(eval_retrieval_expanded_corpus.py)把語料從139條擴大到872條,
同時維持3個epoch不變,結果總訓練step數也跟著從約54步變成327步(約6倍),
MRR不升反降(0.218 -> 0.048)。但無法判斷「語料本身」跟「訓練跑太多步」
哪個才是真正原因,因為兩個變數是綁在一起變動的。

這支腳本維持872條語料不變,但用 steps_per_epoch 把總訓練step數直接鎖定在
跟原本139條、3 epoch時完全一樣的54步。如果MRR回升到接近LoRA/139條全參數
微調的水準,代表問題主要是「step數太多」;如果還是很差,代表語料本身
(例如23個新增類別語料主題發散、跟24題評估集的4部法規語意距離較遠)
也是原因之一。
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
from sentence_transformers import SentenceTransformer, util

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FIXED_STEPS = 54  # 跟原本139條語料、3 epoch、batch_size=8 的總step數一致

with open("../eval/expanded_articles.json", encoding="utf-8") as f:
    articles = json.load(f)
with open("../eval/eval_questions.json", encoding="utf-8") as f:
    eval_qs = json.load(f)

corpus_ids = [a["id"] for a in articles]
corpus_texts = [a["text"] for a in articles]
id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
queries = [q["question"] for q in eval_qs]
gold_ids = [q["gold_id"] for q in eval_qs]

print(f"語料庫大小: {len(articles)} 條,但訓練step數鎖定為 {FIXED_STEPS}(跟139條語料時一致)")


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

print(f"\n開始全參數 TSDAE 微調(872條語料,鎖定總step數={FIXED_STEPS})...")
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import sentence_transformers.losses as losses

train_dataset = DenoisingAutoEncoderDataset(corpus_texts)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
ft_model = SentenceTransformer(BASE_MODEL)
train_loss = losses.DenoisingAutoEncoderLoss(ft_model, decoder_name_or_path=BASE_MODEL, tie_encoder_decoder=True)

ft_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    steps_per_epoch=FIXED_STEPS,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

fixed_step_result = evaluate(ft_model, f"872條語料 + 鎖定{FIXED_STEPS}step微調後")

print("\n=== 四方結果比較(baseline vs. 139條/54step vs. 872條/327step vs. 872條/54step) ===")
try:
    with open("../eval/results.json", encoding="utf-8") as f:
        original_139_result = json.load(f)["tsdae_finetuned"]
except FileNotFoundError:
    original_139_result = None
try:
    with open("../eval/results_expanded_corpus.json", encoding="utf-8") as f:
        expanded_327step_result = json.load(f)["full_finetuned_expanded_corpus"]
except FileNotFoundError:
    expanded_327step_result = None

for k in ["recall@1", "recall@3", "recall@5", "mrr"]:
    line = f"{k}: baseline(872條)={baseline_result[k]:.3f}  872條/54step={fixed_step_result[k]:.3f}"
    if expanded_327step_result:
        line += f"  872條/327step={expanded_327step_result[k]:.3f}"
    if original_139_result:
        line += f"  139條/54step(舊)={original_139_result[k]:.3f}"
    print(line)

with open("../eval/results_expanded_corpus_fixed_steps.json", "w", encoding="utf-8") as f:
    json.dump({
        "corpus_size": len(articles),
        "fixed_steps": FIXED_STEPS,
        "baseline": baseline_result,
        "full_finetuned_872_corpus_54_steps": fixed_step_result,
        "full_finetuned_872_corpus_327_steps_for_reference": expanded_327step_result,
        "full_finetuned_139_corpus_54_steps_for_reference": original_139_result,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_expanded_corpus_fixed_steps.json")
