# ===================================================================
# 檔名: app_langchain.py
# 目的: 整合LangChain與LINE Bot的公務人員法規問答系統
# 功能: 
#   1. 透過LINE Bot接收使用者問題
#   2. 使用LangChain檢索相關法規內容
#   3. 透過OpenAI GPT產生法規說明回答
#   4. 自動發送法規來源摘錄
# 先決條件: 需要先執行setup_database.py建立向量資料庫
# ===================================================================

# === 引入標準函式庫 ===
import os              # 作業系統相關功能，主要用於環境變數
import logging         # 日誌記錄系統，用於追蹤程式執行狀況
from dotenv import load_dotenv    # 載入.env環境變數檔案

# === 引入Flask網頁框架相關 ===
from flask import Flask, request, abort    # Flask基本元件

# === 引入LINE Bot SDK相關元件 ===
from linebot.v3 import WebhookHandler                          # Webhook處理器
from linebot.v3.exceptions import InvalidSignatureError        # 簽章驗證例外
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest    # 訊息API相關
from linebot.v3.messaging import TextMessage, PushMessageRequest                  # 訊息類型
from linebot.v3.webhooks import MessageEvent, TextMessageContent                                # 事件類型

# === 引入LangChain相關元件 ===
from langchain_community.vectorstores import Chroma    # 向量資料庫
from langchain.embeddings import OpenAIEmbeddings      # OpenAI嵌入模型
from langchain_openai import ChatOpenAI                # OpenAI聊天模型
from langchain.chains import RetrievalQA               # 檢索問答鏈
from langchain.prompts import PromptTemplate           # 提示模板

# ===================================================================
# 環境變數載入與系統設定
# ===================================================================

# 載入.env檔案中的環境變數（如API金鑰等敏感資訊）
load_dotenv()

# 設定日誌格式，方便追蹤程式執行狀況與除錯
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================================================================
# 系統組態設定類別
# ===================================================================
class Config:
    """
    系統組態設定檔
    集中管理所有系統參數，方便維護與修改
    """
    
    # === LINE Bot 相關憑證 ===
    # 從環境變數取得LINE Bot的存取權杖，用於發送訊息
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
    # 從環境變數取得LINE Bot的頻道密鑰，用於驗證訊息來源
    LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
    
    # === OpenAI 相關設定 ===
    # OpenAI API金鑰，用於呼叫GPT模型與嵌入模型
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # 指定使用的GPT模型版本，gpt-4o-mini為高性價比選擇
    LLM_MODEL = "gpt-4o-mini"

    # === ChromaDB 向量資料庫設定 ===
    # 向量資料庫的儲存目錄位置
    CHROMA_PERSIST_DIR = './chroma_db_langchain'
    # 資料庫集合名稱，用於區分不同類型的資料
    COLLECTION_NAME = 'civil_service_laws'

    # === 嵌入模型設定 ===
    # OpenAI的文字嵌入模型，用於將文字轉換為向量
    EMBEDDING_MODEL = 'text-embedding-3-small'  # 最新且效率高的嵌入模型
    
    # === 檢索參數設定 ===
    # 每次搜尋時回傳最相關的文件數量，用於GPT分析
    SEARCH_K = 5
    # 實際發送給使用者的法規來源段落數量
    REFERENCES_TO_SEND = 3

# ===================================================================
# LangChain 核心元件初始化
# ===================================================================

# 開始載入OpenAI嵌入模型
logging.info(f"正在載入 OpenAI 嵌入模型: {Config.EMBEDDING_MODEL}...")

# === 環境變數檢查與驗證 ===
# 確保必要的API金鑰與憑證都已正確設定

# 檢查OpenAI API金鑰是否存在
if not Config.OPENAI_API_KEY:
    logging.warning("警告：OPENAI_API_KEY 環境變數未設定，某些功能可能無法使用")
    logging.warning("請設定 OPENAI_API_KEY 環境變數以使用完整功能")

# 檢查LINE Bot存取權杖是否存在
if not Config.LINE_CHANNEL_ACCESS_TOKEN:
    logging.warning("警告：LINE_CHANNEL_ACCESS_TOKEN 環境變數未設定，LINE Bot 功能將無法使用")
    Config.LINE_CHANNEL_ACCESS_TOKEN = "test_token"  # 提供測試用預設值

# 檢查LINE Bot頻道密鑰是否存在
if not Config.LINE_CHANNEL_SECRET:
    logging.warning("警告：LINE_CHANNEL_SECRET 環境變數未設定，LINE Bot 功能將無法使用")
    Config.LINE_CHANNEL_SECRET = "test_secret"  # 提供測試用預設值

# === 核心元件變數初始化 ===
# 先將所有核心元件設為None，稍後會根據API金鑰可用性進行初始化
embeddings = None      # OpenAI嵌入模型實例
vectorstore = None     # ChromaDB向量資料庫實例
llm = None             # OpenAI聊天模型實例
qa_chain = None        # LangChain問答鏈實例

# === 條件式元件初始化 ===
# 只有在OpenAI API金鑰存在時才進行完整初始化
if Config.OPENAI_API_KEY:
    try:
        # 1. 初始化OpenAI嵌入模型
        # 用於將使用者問題與法規內容轉換為向量，以便進行相似度比較
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        logging.info("OpenAI 嵌入模型載入成功。")

        # 2. 載入預先建立的向量資料庫
        # 這個資料庫包含公務人員法規的向量化內容，由setup_database.py建立
        logging.info(f"正在從 '{Config.CHROMA_PERSIST_DIR}' 載入向量資料庫...")
        vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME
        )
        logging.info("向量資料庫載入成功。")

        # 3. 初始化OpenAI聊天模型（GPT）
        # temperature=0 確保回答的一致性與準確性
        llm = ChatOpenAI(
            model_name=Config.LLM_MODEL, 
            temperature=0,  # 設為0確保回答穩定，不會有隨機性
            openai_api_key=Config.OPENAI_API_KEY
        )

        # 4. 建立檢索器
        # 將向量資料庫包裝成檢索器，用於根據問題找出相關法規內容
        retriever = vectorstore.as_retriever(search_kwargs={"k": Config.SEARCH_K})

        # 5. 建立提示模板
        # 定義GPT如何處理檢索到的法規內容並回答使用者問題
        prompt_template = f"""
你是一位公務人員法規助理。請根據以下提供的「相關法規內容」回答使用者問題。

重要指示：
1. 回答需以繁體中文撰寫，內容清楚精簡
2. 僅能依據提供的法規內容回答，不要編造不存在的條文
3. 若有對應依據，請整理重點並引用法規名稱與頁碼
4. 若內容不足，請明確說明「目前提供的法規內容不足以確認」
5. 回答不超過220字

相關法規內容:
{{context}}

問題: {{question}}

回答:
"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # 6. 建立完整的問答鏈
        # 整合檢索器、GPT模型與提示模板，形成完整的問答系統
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,                                    # 使用的語言模型
            # 決定如何處理從檢索器 (retriever) 拿到的文件
            # "stuff" 的意思是：把檢索到的所有文件內容，全部「塞 (stuff)」進同一個提示模板 (Prompt) 的上下文 (Context) 中，然後一次性地發送給 LLM
            # 這種方式簡單直接，但有其物理限制。如果檢索到的文件總長度超過了 LLM 的上下文視窗 (Context Window) 上限，程式就會報錯。對於需要處理大量文件的場景，就需要改用其他 chain_type，例如 "map_reduce" 或 "refine"，這些策略會用更複雜的方式分批處理文件
            chain_type="stuff",                         # 將所有相關文件一次性傳給模型
            retriever=retriever,                        # 檢索器
            # 設定為 True 後，當你執行這個 qa_chain 時，回傳的結果不僅僅是模型的回答字串，而是一個包含兩項內容的字典：
            # 'result': 模型的最終回答。
            # 'source_documents': 模型在回答時參考的所有原始文件。
            return_source_documents=True,               # 回傳原始文件以便後續處理
            # 這個 chain_type_kwargs 字典用於設定問答鏈的運行參數。
            # 其中 prompt 參數用於指定問答鏈的提示模板，這個模板定義了模型在回答時應該參考的上下文資訊。
            # 這個提示模板會被插入到問答鏈的運行過程中，幫助模型理解如何結合檢索到的文件內容來回答使用者的問題。
            chain_type_kwargs={"prompt": PROMPT}        # 使用自訂提示模板
        )
        logging.info("LangChain QA 系統初始化成功。")
    
    except Exception as e:
        logging.error(f"初始化 LangChain 組件時發生錯誤: {e}")
        logging.warning("將以基本模式運行，某些功能可能無法使用")
else:
    # 如果沒有OpenAI API金鑰，系統將以基本模式運行
    logging.warning("未設定 OPENAI_API_KEY，將以基本模式運行")

# ===================================================================
# Flask 網頁伺服器與 LINE Bot 應用初始化
# ===================================================================

# 建立Flask應用程式實例
app = Flask(__name__)

# === LINE Bot API 初始化 ===
# 使用前面設定的憑證初始化LINE Bot相關服務
configuration = Configuration(access_token=Config.LINE_CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)               # 用於發送訊息的API
handler = WebhookHandler(Config.LINE_CHANNEL_SECRET)  # 用於處理接收到的訊息

# === 重複處理防護機制 ===
# 追蹤已處理的reply token，避免同一訊息被處理多次
processed_tokens = set()

# === 請求頻率限制機制 ===
import time  # 引入時間模組用於計算時間間隔
last_request_time = {}      # 記錄每個使用者最後一次請求的時間
REQUEST_COOLDOWN = 2        # 設定冷卻時間為2秒，防止使用者過於頻繁發送請求

# ===================================================================
# 輔助功能函數
# ===================================================================

def format_reference_info(doc, index=1):
    """
    將檢索到的法規文件片段格式化為可讀訊息
    """
    metadata = getattr(doc, 'metadata', {}) or {}
    law_name = metadata.get('law_name', '未知法規')
    page_number = metadata.get('page')
    page_text = f"{int(page_number) + 1}" if isinstance(page_number, int) else "未知"
    excerpt = (doc.page_content or '').strip().replace('\n', ' ')
    excerpt = excerpt[:220] + "..." if len(excerpt) > 220 else excerpt
    return f"法規依據 {index}\n法規名稱：{law_name}\n頁碼：{page_text}\n摘錄：{excerpt}"

def get_push_target_id(event):
    """
    取得可用於 push_message 的目標 ID
    
    優先順序：
    1. group_id（群組）
    2. room_id（多人聊天室）
    3. user_id（一對一）
    """
    source = getattr(event, 'source', None)
    if not source:
        return None
    return (
        getattr(source, 'group_id', None)
        or getattr(source, 'room_id', None)
        or getattr(source, 'user_id', None)
    )

def send_reference_text(target_id, doc, index=1):
    """發送法規來源摘錄到LINE對話目標。"""
    try:
        reference_info = format_reference_info(doc, index)
        line_bot_api.push_message(
            PushMessageRequest(
                to=target_id,
                messages=[TextMessage(text=reference_info)]
            )
        )
        logging.info(f"法規摘錄訊息已發送: {reference_info[:60]}...")
        return True
    except Exception as e:
        logging.error(f"發送法規摘錄失敗: {e}")
        return False

# ===================================================================
# Flask 路由定義
# ===================================================================

@app.route("/", methods=['GET'])
def index():
    """
    系統首頁路由
    提供系統狀態資訊與可用端點說明
    
    Returns:
        str: HTML格式的系統資訊頁面
    """
    return """
    <h1>LangChain LINE Bot 公務人員法規問答系統</h1>
    <p>系統運行中！</p>
    <p>請透過 LINE Bot 進行法規查詢。</p>
    <hr>
    <p><strong>可用端點：</strong></p>
    <ul>
        <li><code>/</code> - 系統狀態頁面</li>
        <li><code>/callback</code> - LINE Bot Webhook</li>
        <li><code>/health</code> - 健康檢查</li>
    </ul>
    """

@app.route("/health", methods=['GET'])
def health_check():
    """
    系統健康檢查端點
    檢查各個核心元件的運作狀況
    
    Returns:
        dict: 包含系統狀態資訊的JSON回應
        tuple: 如果系統異常，會回傳錯誤訊息與HTTP 500狀態碼
    """
    try:
        if vectorstore:
            # 檢查向量資料庫是否正常運作
            # 執行一個簡單的搜尋來驗證連線狀態
            test_docs = vectorstore.similarity_search("test", k=1)
            return {
                "status": "healthy",
                "message": "系統運行正常",
                "vectorstore_status": "connected",           # 向量資料庫已連接
                "documents_count": len(test_docs),            # 測試查詢回傳的文件數量
                "openai_status": "configured" if Config.OPENAI_API_KEY else "not_configured"
            }
        else:
            # 如果向量資料庫不可用，回傳部分功能狀態
            return {
                "status": "partial",
                "message": "系統運行中（基本模式）",
                "vectorstore_status": "not_available",       # 向量資料庫不可用
                "openai_status": "not_configured",
                "note": "請設定 OPENAI_API_KEY 以使用完整功能"
            }
    except Exception as e:
        # 如果健康檢查過程中發生錯誤，回傳錯誤狀態
        return {
            "status": "unhealthy",
            "message": f"系統異常: {str(e)}",
            "vectorstore_status": "error"
        }, 500

@app.route("/callback", methods=['POST'])
def callback():
    """
    LINE Bot Webhook 回調端點
    接收並處理來自LINE平台的所有訊息與事件
    
    Returns:
        str: 處理成功回傳'OK'，失敗會回傳HTTP錯誤碼
    """
    # 1. 獲取LINE平台的數位簽章，用於驗證請求來源的真實性
    signature = request.headers.get('X-Line-Signature', '')
    if not signature:
        logging.error("缺少 X-Line-Signature 標頭")
        abort(400)  # 回傳HTTP 400 Bad Request
    
    # 2. 獲取請求主體內容（包含使用者的訊息資料）
    body = request.get_data(as_text=True)
    logging.info(f"收到 webhook 請求，內容長度: {len(body)}")
    
    try:
        # 3. 使用LINE Bot SDK的handler處理請求
        # handler會驗證簽章並將事件分派給對應的處理函數
        handler.handle(body, signature)
    except InvalidSignatureError:
        # 如果簽章驗證失敗，表示請求來源不可信
        logging.error("簽章驗證失敗")
        abort(400)
    except Exception as e:
        # 處理其他未預期的錯誤
        logging.error(f"Webhook 處理錯誤: {e}")
        abort(500)  # 回傳HTTP 500 Internal Server Error
    
    return 'OK'  # 成功處理請求

# ===================================================================
# LINE Bot 訊息事件處理器
# ===================================================================

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    處理來自LINE使用者的文字訊息
    這是系統的核心函數，整合了LangChain檢索與OpenAI問答功能
    
    Args:
        event: LINE平台發送的訊息事件物件
    """
    # === 1. 事件驗證與檢查 ===
    logging.info(f"收到事件: {type(event).__name__}")
    
    # 檢查事件是否包含必要的reply_token
    if not hasattr(event, 'reply_token') or not event.reply_token:
        logging.warning("無效事件: 缺少 reply_token")
        return
    
    # 檢查是否為文字訊息
    if not hasattr(event, 'message') or not hasattr(event.message, 'text'):
        logging.warning("無效事件: 非文字訊息")
        return
    
    # === 2. 重複處理防護 ===
    # 檢查此reply_token是否已經處理過，避免重複回應
    if event.reply_token in processed_tokens:
        logging.warning(f"Reply token {event.reply_token} 已處理過，跳過")
        return
    
    # 標記此token為已處理
    processed_tokens.add(event.reply_token)
    
    # 維護processed_tokens集合大小，避免記憶體洩漏
    if len(processed_tokens) > 1000:
        processed_tokens.clear()
    
    # === 3. 提取使用者查詢內容 ===
    user_query = event.message.text.strip()
    logging.info(f"接收到使用者查詢: {user_query} (reply_token: {event.reply_token})")
    
    # 過濾空訊息或系統訊息
    if not user_query or user_query.startswith('LineBot'):
        logging.info("忽略空訊息或系統訊息")
        return
    
    # === 4. 請求頻率限制檢查 ===
    # 防止使用者過於頻繁發送請求，保護系統資源
    user_id = getattr(event.source, 'user_id', 'unknown')
    current_time = time.time()
    if user_id in last_request_time:
        time_since_last = current_time - last_request_time[user_id]
        if time_since_last < REQUEST_COOLDOWN:
            logging.info(f"請求太頻繁，忽略 (用戶: {user_id}, 間隔: {time_since_last:.1f}秒)")
            return
    
    # 更新此使用者的最後請求時間
    last_request_time[user_id] = current_time

    # === 5. 核心查詢處理邏輯 ===
    try:
        if qa_chain and vectorstore:
            # === 使用LangChain完整問答系統 ===
            # 呼叫問答鏈，整合檢索、GPT回答與來源文件
            result = qa_chain.invoke({"query": user_query})
            
            # 提取GPT生成的回答文字
            llm_answer = result.get('result', "抱歉，我無法處理您的請求。").strip()
            # 提取檢索到的相關法規文件
            source_documents = result.get('source_documents', [])
            
            # === 備用檢索機制 ===
            # 如果QA鏈沒有找到相關文件，嘗試直接搜尋向量資料庫
            if not source_documents or len(source_documents) == 0:
                try:
                    docs = vectorstore.similarity_search(user_query, k=Config.SEARCH_K)
                    if docs:
                        source_documents = docs
                        logging.info(f"備用搜尋找到 {len(docs)} 個相關文件")
                except Exception as e:
                    logging.warning(f"備用搜索失敗: {e}")

            # === 6. 發送GPT回答 ===
            # 先回覆使用者GPT生成的總結性回答
            try:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=llm_answer)]
                    )
                )
                logging.info("成功回覆 LLM 訊息")
            except Exception as reply_error:
                logging.error(f"回覆 LLM 訊息失敗: {reply_error}")
            
            # === 7. 發送法規來源摘錄 ===
            # 如果找到相關法規內容，逐一發送來源摘錄
            if source_documents:
                target_id = get_push_target_id(event)
                if target_id:
                    for i, doc in enumerate(source_documents[:Config.REFERENCES_TO_SEND]):
                        send_reference_text(target_id, doc, i + 1)
                else:
                    logging.warning("無法獲取有效推播目標 ID")
        else:
            # === 基本模式回應處理 ===
            # 當LangChain元件未正確初始化時的降級處理
            response_text = (
                f"您好！我收到了您的訊息：「{user_query}」\n\n"
                "目前系統正在基本模式運行中。\n"
                "要使用完整的法規問答功能，請管理員設定以下環境變數：\n"
                "• OPENAI_API_KEY\n"
                "• 重新建立向量資料庫\n\n"
                "感謝您的使用！"
            )
            try:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=response_text)]
                    )
                )
                logging.info("成功回覆基本模式訊息")
            except Exception as reply_error:
                logging.error(f"回覆基本模式訊息失敗: {reply_error}")

    except Exception as e:
        # === 全域錯誤處理 ===
        # 捕獲任何未預期的錯誤，確保系統穩定運行
        logging.error(f"處理訊息時發生錯誤: {e}")
        try:
            # 嘗試發送使用者友善的錯誤訊息
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="系統發生一點問題，請稍後再試。")]
                )
            )
        except Exception as error_reply_error:
            # 如果連錯誤訊息都無法發送，只能記錄到日誌
            logging.error(f"發送錯誤訊息失敗: {error_reply_error}")

# ===================================================================
# 應用程式啟動點
# ===================================================================

if __name__ == "__main__":
    # 從環境變數取得Port號，預設為5000
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"啟動 Flask 應用程式於 port {port}")
    
    # 啟動Flask開發伺服器
    # host='0.0.0.0' 允許外部連線（適用於雲端部署）
    app.run(host='127.0.0.1', port=port)
