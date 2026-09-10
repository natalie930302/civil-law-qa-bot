import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    DATA_FILE_PATHS = [
        'datasets\\公務人員考績法.pdf',
        'datasets\\公務人員考績法施行細則.pdf',
        'datasets\\公務員懲戒法.pdf',
        'datasets\\公務員服務法.pdf',
    ]

    CHROMA_PERSIST_DIR = './chroma_db_langchain'
    COLLECTION_NAME = 'civil_service_laws'

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL = 'text-embedding-3-small'

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 120


def load_law_documents():
    all_documents = []

    for file_path in Config.DATA_FILE_PATHS:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到資料檔案: {file_path}")

        law_name = Path(file_path).stem
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata['law_name'] = law_name

        all_documents.extend(docs)
        logging.info(f"✅ 載入完成：{law_name}，共 {len(docs)} 頁")

    return all_documents


def main():
    logging.info("=== 開始建置公務人員法規向量資料庫 ===")

    try:
        documents = load_law_documents()
        logging.info(f"✅ 成功載入全部法規文件，共 {len(documents)} 份頁面文件")
    except FileNotFoundError as e:
        logging.error(f"❌ {e}")
        return
    except Exception as e:
        logging.error(f"❌ 載入法規文件時發生錯誤: {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"✅ 已完成切割，共 {len(split_docs)} 個文字片段")

    if not Config.OPENAI_API_KEY:
        logging.error("❌ 錯誤：請設定 OPENAI_API_KEY 環境變數")
        return

    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )
    logging.info("✅ OpenAI 嵌入模型載入成功")

    try:
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
            collection_name=Config.COLLECTION_NAME
        )

        vector_count = vectorstore._collection.count()
        logging.info("🎉 向量資料庫建置完成！")
        logging.info(f"📊 共 {vector_count} 個向量")
        logging.info(f"💾 資料庫位置: {Config.CHROMA_PERSIST_DIR}")
        logging.info("=== 公務人員法規向量資料庫建置完畢 ===")
    except Exception as e:
        logging.error(f"❌ 建立向量資料庫時發生錯誤: {e}")


if __name__ == "__main__":
    main()
