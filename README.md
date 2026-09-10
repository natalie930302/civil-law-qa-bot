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
