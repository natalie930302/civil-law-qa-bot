"""
把 datasets/ 底下的法規PDF切成逐條文chunk,存成 JSON。
每條文格式: {id, law_name, article_no, text}
"""
import fitz
import re
import json
import glob
import os

OUT_PATH = "../eval/articles.json"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_articles(text, law_name):
    # 依「第 N 條」切分(N可能是中文數字或阿拉伯數字)
    pattern = re.compile(r"第\s*([一二三四五六七八九十百零〇\d]+)\s*條(?:之([一二三四五六七八九十\d]+))?")
    matches = list(pattern.finditer(text))
    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        body = re.sub(r"\s+", " ", body)
        no = m.group(1) + (f"之{m.group(2)}" if m.group(2) else "")
        if len(body) > 5:
            articles.append({
                "id": f"{law_name}_第{no}條",
                "law_name": law_name,
                "article_no": no,
                "text": body,
            })
    return articles

all_articles = []
for pdf_path in glob.glob("../datasets/*.pdf"):
    law_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text = extract_text(pdf_path)
    arts = split_articles(text, law_name)
    print(f"{law_name}: 擷取到 {len(arts)} 條")
    all_articles.extend(arts)

print(f"總條文數: {len(all_articles)}")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_articles, f, ensure_ascii=False, indent=2)
print(f"已存至 {OUT_PATH}")
