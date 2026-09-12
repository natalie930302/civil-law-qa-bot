"""
幫正式上線的 LINE Bot(app_langchain.py,走 OpenAI text-embedding-3-small +
Chroma,跟 eval/ 裡驗證 LoRA/TSDAE 的本地 sentence-transformers + FAISS 是
完全不同的兩條線)校準一個信心閾值,讓 Bot 在檢索結果不夠相關時,誠實回報
「目前資料不足以確認」,而不是硬答。

做法比照 food-rag 的 tune_confidence_threshold.py:用24題真實in-domain問題
(eval_questions.json,跟這個Chroma DB用同一組4部法規PDF建的,問題文字可以
直接重用)跟一組明顯無關的out-of-domain問題,量測 Chroma 的
similarity_search_with_score() 回傳的距離分數(L2距離,越小越相似)分布,
看兩組有沒有清楚間隔。

注意:Chroma回傳的是「距離」不是「相似度」,跟food-rag的reranker分數方向
相反——這裡是「distance越小越有信心」,food-rag是「score越大越有信心」。
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "chroma_db_langchain")
COLLECTION_NAME = "civil_service_laws"
EMBEDDING_MODEL = "text-embedding-3-small"

OUT_OF_DOMAIN_QUESTIONS = [
    "台灣的所得稅申報期限是什麼時候?",
    "Unity 遊戲引擎裡要怎麼設定角色的動畫過渡?",
    "颱風來的時候要注意哪些居家安全事項?",
    "貓咪大概多久需要施打一次疫苗?",
    "台北到高雄搭高鐵大概要多久時間?",
    "護照到期了要怎麼申請換發?",
]

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    in_domain_qs = [q["question"] for q in json.load(f)]

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)


def top1_distances(questions):
    out = []
    for q in questions:
        results = vectorstore.similarity_search_with_score(q, k=1)
        out.append(results[0][1] if results else None)
    return out


in_domain_dist = top1_distances(in_domain_qs)
out_domain_dist = top1_distances(OUT_OF_DOMAIN_QUESTIONS)

print("=== In-domain(24題,跟Chroma DB同一批4部法規)top-1 距離 ===")
for q, d in zip(in_domain_qs, in_domain_dist):
    print(f"  {d:.4f}  {q}")
print(f"\n  min={min(in_domain_dist):.4f}  max={max(in_domain_dist):.4f}  "
      f"avg={sum(in_domain_dist)/len(in_domain_dist):.4f}")

print("\n=== Out-of-domain(6題明顯無關)top-1 距離 ===")
for q, d in zip(OUT_OF_DOMAIN_QUESTIONS, out_domain_dist):
    print(f"  {d:.4f}  {q}")
print(f"\n  min={min(out_domain_dist):.4f}  max={max(out_domain_dist):.4f}  "
      f"avg={sum(out_domain_dist)/len(out_domain_dist):.4f}")

# 距離越小越相關,所以間隔 = out-of-domain最小距離 - in-domain最大距離
gap = min(out_domain_dist) - max(in_domain_dist)
print(f"\n=== 間隔分析 ===")
print(f"in-domain 最大距離(最不confident的正常問題): {max(in_domain_dist):.4f}")
print(f"out-of-domain 最小距離(最像有關的無關問題): {min(out_domain_dist):.4f}")
if gap > 0:
    suggested = (max(in_domain_dist) + min(out_domain_dist)) / 2
    print(f"兩組距離有清楚間隔(gap={gap:.4f}),建議閾值(取中點,距離小於此值才算confident): {suggested:.4f}")
else:
    print(f"⚠️ 兩組距離有重疊(gap={gap:.4f} < 0),沒有一個閾值能完美切開兩組,"
          f"這是誠實的實測結果,不是bug——見README討論")

with open(Path(__file__).parent / "bot_threshold_tuning_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "in_domain_distances": in_domain_dist,
        "out_domain_distances": out_domain_dist,
        "in_domain_max": max(in_domain_dist),
        "out_domain_min": min(out_domain_dist),
        "gap": gap,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/bot_threshold_tuning_results.json")
