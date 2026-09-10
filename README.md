# LangChain_Linebot — 公務人員法規問答系統

用 LangChain 框架整合 LINE Bot 的法規問答系統,讓使用者透過 LINE 對話查詢公務人員相關法規,並附上法規來源摘錄。

## 運作流程

1. 透過 LINE Bot 接收使用者問題
2. 用 LangChain 在向量資料庫(`chroma_db_langchain`)中檢索相關法規內容
3. 透過 OpenAI GPT 產生法規說明回答
4. 自動附上法規來源摘錄回傳給使用者

## 檔案說明

- `setup_database.py` — 讀取 `datasets/` 底下的法規文件,建立 Chroma 向量資料庫(需先執行)
- `app_langchain.py` — 主程式,處理 LINE Webhook、LangChain 檢索與 GPT 回答生成
- `chroma_db_langchain/` — 向量資料庫檔案

## 技術棧

Python + LangChain + Chroma(向量資料庫)+ OpenAI GPT + LINE Messaging API

## 環境設定

複製 `.env.example` 為 `.env`,並填入以下金鑰:

```
OPENAI_API_KEY=你的OpenAI金鑰
LINE_CHANNEL_ACCESS_TOKEN=你的LINE頻道存取權杖
LINE_CHANNEL_SECRET=你的LINE頻道密鑰
```

```bash
pip install -r requirements.txt
python setup_database.py   # 建立向量資料庫
python app_langchain.py    # 啟動服務
```

## 檢索品質評估(2026/09 新增)

原本的系統只做到「能回答」,沒有量化過檢索品質。這輪補上人工標註的評估集,量測 RAG 系統最關鍵的一環:「檢索有沒有真的撈到對的法條」。

- `src/parse_laws.py`:把 `datasets/` 裡的 4 部法規 PDF 切成逐條 JSON(`eval/articles_clean.json`,139條乾淨資料)
- `eval/eval_questions.json`:24 題手動撰寫、人工核對過 `gold_id` 的真實問題,涵蓋全部 4 部法規,跟訓練/建庫語料完全不重疊(避免資料洩漏)
- `src/eval_retrieval.py`:評估 baseline embedding(`paraphrase-multilingual-MiniLM-L12-v2`)vs. TSDAE 領域微調後的 embedding

### 結果

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| **Baseline(未微調)** | **0.458** | **0.625** | **0.667** | **0.555** |
| TSDAE 領域微調後 | 0.167 | 0.250 | 0.250 | 0.218 |

### 重要發現:TSDAE 微調反而讓檢索變差,不是變好

這是繼 `food-violation-severity-model` 的 BERT 負向結果後,第二個誠實記錄的「直覺上應該更好、但實際更差」案例。TSDAE 是無監督領域微調法,理論上只用 139 條法規條文語料訓練,不需要人工標註問答配對,應該能讓 embedding 更貼近法規領域的語言風格。但實際結果全面下滑(MRR 0.555 → 0.218)。

推測原因:
- **語料量太小**:139 條條文對一個要重新調整整個 embedding 空間的模型來說遠遠不夠,3 epoch、54 個訓練 step 就讓模型偏離了原本通用語義空間的品質(`train_loss` 收斂在 10.6,明顯沒有學到穩定的重建能力)
- **TSDAE 的訓練目標(去噪重建句子)跟下游任務(語義檢索排序)不一致**,語料量不足時,這種目標不一致比訓練帶來的領域適應效果影響更大,等於是用一個訓練不充分的重建任務把原本檢索能力很好的通用多語embedding「洗壞」了
- 呼應 `food-violation-severity-model` 得到的同一個原則:**模型複雜度/訓練策略要跟資料量匹配**,資料量不夠時貿然微調反而有害,不是保底也能持平

## 用 LoRA 驗證「災難性遺忘」假設(2026/09 新增)

上面的負向結果,理論上符合機器學習裡的 **Catastrophic Forgetting(災難性遺忘)**:全參數微調在只有139條語料的小資料集上,訓練目標(去噪重建)跟下游任務(語義檢索)不一致,足以蓋掉/破壞原本在大規模語料上學到的通用多語語義空間。這不只是憑直覺推測,而是可以直接驗證的假設——如果原因真的是「全參數微調動到了太多不該動的權重」,那麼凍結大部分原始權重、只微調一小部分參數(LoRA / PEFT,近年參數效率微調的主流做法,原因正是為了避免這個問題),應該就能大幅緩解。

`src/eval_retrieval_lora.py` 用同樣的139條語料、同樣的TSDAE去噪重建訓練目標,但只在 attention 的 query/value 插入低秩 adapter,凍結 99.87% 的原始參數(可訓練參數 147,456 / 117,801,216)。

### 結果:LoRA 幾乎完整保留了原本的檢索能力

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR | 可訓練參數比例 |
|---|---|---|---|---|---|
| **Baseline(未微調)** | 0.458 | 0.625 | 0.667 | 0.555 | — |
| **LoRA 微調後** | **0.458** | **0.625** | 0.625 | **0.544** | 0.13% |
| 全參數微調 | 0.167 | 0.250 | 0.250 | 0.218 | 100% |

LoRA 版本的 MRR(0.544)幾乎跟 baseline(0.555)打平,Recall@1/3 完全相同,只有 Recall@5 略降一點點——跟全參數微調的全面崩盤(MRR掉到0.218)是天壤之別。**這個對照直接驗證了「小資料下全參數微調的問題出在動到了太多權重」這個假設**:同一份139條語料、同一個訓練目標,只是把可訓練參數從100%降到0.13%,檢索品質的破壞幾乎完全消失。

### 一個意外插曲:LoRA 不是自動免疫,超參數沒調好一樣會壞

第一次嘗試時,依照「LoRA通常可以用比全參數微調更高的學習率」這個常見經驗法則,把學習率設成 3e-4(比全參數微調的 3e-5 高10倍),結果**比全參數微調還要更差**——檢索完全失效(MRR=0.024),而且是一種不同的失效模式:不是遺忘,而是**表徵坍縮(representation collapse)**,任兩篇完全不同法規條文算出來的embedding,cosine相似度高達0.9996,等於模型不管輸入什麼都輸出幾乎一樣的向量。這是靠診斷embedding本身(檢查是否NaN、算相似度)才抓到的,不是單純看最終分數判斷。把學習率調回跟全參數微調一致的 3e-5 之後,才得到上面那組乾淨的正向結果。

這個插曲本身也是一個值得記錄的發現:**「LoRA 對超參數不敏感/更穩健」不是可以無腦套用的通則**,在極小資料集(139筆)的無監督重建任務上,學習率選擇仍然是決定成敗的關鍵變因,LoRA 降低的是「訓練動到的權重量」,不是「訓練不穩定的風險」。

### 如何重現

```bash
cd src
python parse_laws.py            # 從PDF切出逐條法規JSON
python eval_retrieval.py        # 跑baseline+全參數TSDAE微調評估,結果存到 eval/results.json
python eval_retrieval_lora.py   # 跑baseline+LoRA版TSDAE微調評估,結果存到 eval/results_lora.json
```

> 微調後的模型檔(`eval/tsdae_finetuned_model/`,約 458MB)已排除在 `.gitignore`,超過 GitHub 單檔限制;對應腳本重新執行即可再產生。
