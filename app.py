"""
Naive Bayes sentiment (star rating 1-5) on Wongnai reviews
Steps follow the whiteboard:
  1. training set / test set
  2. tokenize -> stop words removing + negation/intensifier binding (Thai normalization)
  3. TF table per class + Laplace smoothing (+1)
  4. P(class|doc) ∝ P(class) * Π P(word|class)   (OOV words are ignored)
  5. heat map, word cloud (รวม / 1 / 2 / 3 / 4 / 5), performance test (accuracy)
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import io
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Tahoma"  # Thai-capable font on Windows
THAI_FONT = r"C:\Windows\Fonts\tahoma.ttf"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 2. preprocessing setup
raw_stop = set(thai_stopwords())

# Keep sentiment-bearing words that PyThaiNLP includes in default stopwords
SENTIMENT_KEEP = {
    "ไม่", "ไม่ได้", "ไม่มี", "ไม่ค่อย", "ไม่เคย", "ไม่น่า",
    "ดี", "ดีมาก", "สุด", "สุดๆ", "มาก", "มากๆ", "น้อย", "นิด", "นิดเดียว",
    "เกิน", "เกินไป", "ค่อย", "ค่อนข้าง", "พอ", "พอใช้", "พอได้", "เฉย", "เฉยๆ",
    "ธรรมดา", "แย่", "เลว", "ร้าย", "เสีย", "ขาด", "ช้า", "เร็ว", "ไว",
    "แพง", "ถูก", "คุ้ม", "ชอบ", "รัก", "เกลียด",
    "ประทับใจ", "ผิดหวัง", "เสียดาย", "เสียใจ", "ห่วย", "พัง", "ซ้ำ",
}

# Generic restaurant/domain nouns that appear across all 1-5 star reviews
DOMAIN_STOP = {
    "ร้าน", "อาหาร", "ทาน", "กิน", "เมนู", "สั่ง", "จาน", "โต๊ะ", "คน", "บาท",
    "สาขา", "ซอย", "ถนน", "ชั้น", "ห้าง", "กก", "ตัว", "ชิ้น", "ถ้วย", "แก้ว",
    "ขวด", "จอด", "รถ", "เพื่อน", "พี่", "น้อง", "แฟน", "บ้าน", "ผม", "หนู",
    "เค้า", "นะคะ", "นะครับ", "ค่ะ", "ครับ", "เลย", "ตอน", "เวลา", "นาที",
    "ชั่วโมง", "รอบ", "เรื่อง", "แบบ", "อัน", "อย่าง", "ทีนี่", "ที่นี่", "สำหรับ",
    "เลือก", "ลอง", "ทำ", "ใส่", "ดู", "เดิน", "นั่ง", "ขาย", "ซื้อ", "ชื่อ",
    "หน้า", "แถว", "ข้าง", "ติด", "ใกล้", "ไกล", "เปิด", "ปิด", "น้ำ", "ข้าว",
    "หมู", "ไก่", "ปลา", "เนื้อ", "กุ้ง", "เส้น", "ผัด", "ทอด", "ต้ม", "ยำ",
    "ซุป", "กาแฟ", "เค้ก", "ชา", "นม", "ขนม", "ผัก", "พริก", "ไข่", "ชีส",
    "ซอส", "ก๋วยเตี๋ยว", "ส้มตำ", "ขนมปัง", "ไอศกรีม", "ปู", "หอย", "เป็ด",
    "ลูกชิ้น", "น้ำจิ้ม", "เครื่องดื่ม", "ของหวาน", "สลัด", "พิซซ่า", "ซูชิ",
    "รสชาติ", "ราคา", "บริการ", "พนักงาน", "ลูกค้า", "บรรยากาศ", "รส",
    "บอ", "ถาม", "อ้าว", "อึ้ง", "ผู้หญิง", "ผู้ชาย", "หก", "เหรอ", "เห้อ", "เฮ้อ",
    "มอเตอร์ไซค์", "ปีกนก", "แปลงผัก", "สวนผัก", "ฟักทอง", "กระเบื้อง",
    "ธนบุรี", "ปทุม", "สุโขทัย", "เชียงใหม่", "ออสเตรเลีย", "ญี่ปุ่น", "ไทย", "จีน",
    "โอวัลติน", "เยนตาโฟ", "หมูกรอบ", "หมูแดง", "คอหมูย่าง", "ซี่โครง", "สลัดผลไม้",
    "ชาเขียว", "ขนมหวาน", "ไอศครีม", "บิงซู", "เจ้าของร้าน", "แม่ค้า", "พ่อค้า",
    "จอดรถ", "ที่จอดรถ", "ร้านอาหาร", "เครื่อง", "แถม", "แวะ", "เข้ามา", "เข้าไป",
    "ตั้งอยู่", "ค่อนข้าง", "เหมือน", "ขนาด", "รีวิว", "กลิ่น", "สี", "สอง",
}

STOP = raw_stop - SENTIMENT_KEEP

# Words to exclude from WordCloud/Heatmap display so only clear sentiment/descriptive words appear
DISPLAY_EXCLUDE = DOMAIN_STOP | {
    "มาก", "พลาด", "ห้าม", "ขอโทษ", "พอ", "ไม่มี", "ไม่ได้", "ไม่มา", "เงิน", "ยืน",
    "อุ่น", "เช็คบิล", "แทน", "ขาด", "สรุป", "เฉย", "ไม่โอ", "ที่นั่ง", "แอร์",
    "ร้อน", "เย็น", "ชิม", "หา", "แอบ", "นิด", "นิดนึง", "ไก่ทอด", "ต้มยำ", "พิกัด",
    "โซน", "ตลาด", "มื้อ", "ย่าง", "ราด", "ไส้", "ห้อง", "ตกแต่ง",
    "หุ่นยนต์", "อ้างว่า", "พิพ", "พวกผม", "ลักค์", "บาต", "ก่", "จู๋", "ด้วยประการทั้งปวง",
    "สำเพ็ง", "เส", "อ่ะ", "อะ", "บริเวณ", "ตรงข้าม", "ฝั่ง", "ด้านใน", "ชาม",
}
NEGATIONS = {"ไม่", "ไม่ได้", "ไม่มี", "ไม่ค่อย", "ไม่น่า", "ไม่เคย"}
INTENSIFIERS = {"มาก", "มากๆ", "สุด", "สุดๆ", "เกิน", "เกินไป"}
RE_REPEAT = re.compile(r"(.)\1{2,}")
RE_THAI = re.compile(r"^[\u0E00-\u0E7F]{2,}$")


def tokenize(text: str) -> list[str]:
    text = RE_REPEAT.sub(r"\1", text.lower())
    raw_toks = word_tokenize(text, engine="newmm", keep_whitespace=False)
    toks = [t.strip() for t in raw_toks if t.strip() and RE_THAI.match(t.strip())]
    out = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        # Bind negation prefix: e.g. "ไม่" + "อร่อย" -> "ไม่อร่อย", "ไม่" + "ประทับใจ" -> "ไม่ประทับใจ"
        if t in NEGATIONS and i + 1 < n:
            nxt = toks[i + 1]
            if nxt not in DOMAIN_STOP:
                out.append(f"{t}{nxt}")
                i += 2
                continue
        # Bind sentiment word + intensifier: e.g. "อร่อย" + "มาก" -> "อร่อยมาก", "แย่" + "มาก" -> "แย่มาก"
        if t not in STOP and i + 1 < n and toks[i + 1] in INTENSIFIERS:
            out.append(f"{t}{toks[i + 1]}")
            i += 2
            continue
        if t not in STOP:
            out.append(t)
        i += 1
    return out


class TeeOutput:
    """Write stdout to both console and a string buffer for saving output.txt."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, data):
        self.stream.write(data)
        self.buffer.write(data)

    def flush(self):
        self.stream.flush()


if __name__ == "__main__":
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    # ==========================================================================
    # PART A: โจทย์ตัวอย่างบนกระดาน (Whiteboard Toy Example)
    # ==========================================================================
    print("=" * 72)
    print("PART A: โจทย์คำนวณ Naive Bayes บนกระดาน (Whiteboard Example)")
    print("=" * 72)
    wb_train = [
        ("I love this product", "Y"),
        ("He loves this product", "Y"),
        ("She hates this product", "N"),
    ]
    wb_test = "John love this product"
    wb_vocab = ["I", "love", "product", "He", "She", "hate"]
    wb_tf = pd.DataFrame(
        [[1, 2, 2, 1, 0, 0], [0, 0, 1, 0, 1, 1]],
        index=["Y", "N"],
        columns=wb_vocab,
    )
    print("1) Training Set:")
    for doc, lbl in wb_train:
        print(f"   - {doc}. -> {lbl}")
    print(f"   Test Set: {wb_test} -> Y/N  ('John' = OOV, 'this' = stop word)")
    print("\n2) Vocabulary (หลังตัด Stop word 'this' และ Stemming loves->love, hates->hate):")
    print("  ", wb_vocab, f"(|V| = {len(wb_vocab)})")
    print("\n3) ตาราง TF + Laplace Smoothing (+1):")
    wb_prob = pd.DataFrame(
        {
            w: [f"{wb_tf.loc['Y', w]}+1 / 8 = {(wb_tf.loc['Y', w]+1)/8:.3f}",
                f"{wb_tf.loc['N', w]}+1 / 7 = {(wb_tf.loc['N', w]+1)/7:.3f}"]
            for w in wb_vocab
        },
        index=["Y", "N"],
    )
    print(wb_prob.to_string())
    p_y = (2 / 3) * (3 / 8) * (3 / 8)
    p_n = (1 / 3) * (1 / 7) * (2 / 7)
    print("\n4) คำนวณความน่าจะเป็นของประโยคทดสอบ 'John love this product':")
    print(f"   P(Y | test) ∝ P(Y) * P(love|Y) * P(product|Y) = (2/3)*(3/8)*(3/8) = {p_y:.3f}")
    print(f"   P(N | test) ∝ P(N) * P(love|N) * P(product|N) = (1/3)*(1/7)*(2/7) = {p_n:.3f}")
    print("   => สรุปผลทำนาย: ตอบ Y (เพราะ 0.094 > 0.014)\n")

    # ==========================================================================
    # PART B: Wongnai Reviews (1-5 Stars)
    # ==========================================================================
    print("=" * 72)
    print("PART B: Wongnai Reviews Sentiment Analysis (1-5 Stars)")
    print("=" * 72)

    # ---------------------------------------------------------------- 1. data
    try:
        ds = load_dataset("Highgroundbkk/wongnai_reviews")
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        ds = load_dataset("Highgroundbkk/wongnai_reviews")

    df = ds["train"].to_pandas()
    df["star"] = df["star_rating"] + 1          # 0-4 -> 1-5 stars
    print("จำนวนรีวิวแต่ละระดับดาว (1-5):")
    print(df["star"].value_counts().sort_index())

    # Use cached tokens if available
    CACHE_PATH = os.path.join(OUT, ".tokens_cache.pkl")
    if os.path.exists(CACHE_PATH):
        cached_series = pd.read_pickle(CACHE_PATH)
        df["tokens_str"] = [
            " ".join(w for w in s.split() if RE_THAI.match(w) and w not in STOP)
            for s in cached_series
        ]
    else:
        tokens_list = Parallel(n_jobs=-1, backend="loky", batch_size=500)(
            delayed(tokenize)(txt) for txt in df["review_body"].tolist()
        )
        df["tokens_str"] = [" ".join(t) for t in tokens_list]
        df["tokens_str"].to_pickle(CACHE_PATH)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["star"]
    )

    # ---------------------------------------------------------------- 3. TF table
    vec = CountVectorizer(
        token_pattern=r"\S+",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.5,
        max_features=30000,
        binary=True,
    )
    X_train = vec.fit_transform(train_df["tokens_str"])
    X_test = vec.transform(test_df["tokens_str"])          # OOV words simply dropped
    y_train, y_test = train_df["star"].values, test_df["star"].values
    print("Vocabulary size:", len(vec.vocabulary_))

    # ---------------------------------------------------------------- 4. Naive Bayes (alpha=1 -> +1 smoothing)
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)

    # ---------------------------------------------------------------- 5a. performance test (Accuracy)
    pred = nb.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(classification_report(y_test, pred, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=[1, 2, 3, 4, 5])
    print("Confusion Matrix (Rows = True star 1-5, Cols = Predicted star 1-5):")
    print(pd.DataFrame(cm, index=[f"True {i}" for i in range(1, 6)], columns=[f"Pred {i}" for i in range(1, 6)]))

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(1, 6), yticklabels=range(1, 6)
    )
    plt.xlabel("Predicted star")
    plt.ylabel("True star")
    plt.title(f"Confusion Matrix (Accuracy = {acc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120)
    plt.close()

    # ---------------------------------------------------------------- 5b. heat map of P(word|class)
    vocab = np.array(vec.get_feature_names_out())
    prob = np.exp(nb.feature_log_prob_)                     # smoothed (count+1)/(total+|V|)
    counts = np.asarray(nb.feature_count_)
    total_counts = counts.sum(axis=0)

    # Count how many UNIQUE original reviews in each star contain each word (prevents single-review artifacts)
    X_orig_bin = (vec.transform(df["tokens_str"]) > 0).astype(int)
    orig_doc_counts = np.vstack([
        np.asarray(X_orig_bin[df["star"].values == c].sum(axis=0)).ravel()
        for c in nb.classes_
    ])

    # Use single-token Thai sentiment words for clean visualization in Heatmap & WordCloud
    is_display_word = np.array([
        " " not in w and len(w) >= 2 and RE_THAI.match(w) is not None and w not in DISPLAY_EXCLUDE
        for w in vocab
    ])

    # Compute class distinctiveness so each star shows its own signature words
    top_idx_list = []
    for c_idx, c in enumerate(nb.classes_):
        other_mean = np.delete(prob, c_idx, axis=0).mean(axis=0)
        lift = prob[c_idx] / (other_mean + 1e-12)
        min_docs = 8 if c == 1 else (18 if c == 2 else 45)
        valid = is_display_word & (orig_doc_counts[c_idx] >= min_docs) & (prob.argmax(axis=0) == c_idx)
        score = np.where(valid, (prob[c_idx] ** 0.65) * (lift ** 1.35), 0.0)
        best_for_c = [i for i in np.argsort(score)[::-1] if score[i] > 0][:4]
        for idx in best_for_c:
            if idx not in top_idx_list:
                top_idx_list.append(idx)

    top_idx = np.array(top_idx_list)
    heat_raw = prob[:, top_idx]
    heat_norm = heat_raw / (heat_raw.max(axis=0, keepdims=True) + 1e-12)
    heat_df = pd.DataFrame(
        heat_norm,
        index=[f"{c} star" for c in nb.classes_],
        columns=vocab[top_idx],
    )
    annot_df = pd.DataFrame(
        heat_raw * 1000,  # ×10⁻³ for readable numbers in cells
        index=[f"{c} star" for c in nb.classes_],
        columns=vocab[top_idx],
    )

    plt.figure(figsize=(16, 5))
    sns.heatmap(
        heat_df, cmap="YlOrRd", annot=annot_df, fmt=".2f",
        cbar_kws={"label": "Relative P(word | star) intensity"}
    )
    plt.title("Heat Map: P(word | star) [×10⁻³] – คำเด่นจำแนกตามระดับดาว (Laplace smoothed)")
    plt.xticks(rotation=25, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "heatmap.png"), dpi=120)
    plt.close()

    # ---------------------------------------------------------------- 5c. word cloud (รวม / 1 / 2 / 3 / 4 / 5)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    # Panel 0: รวม (Overall top sentiment/descriptive words across all stars)
    orig_total_docs = orig_doc_counts.sum(axis=0)
    overall_mask = is_display_word & (orig_total_docs >= 20)
    overall_idx = np.argsort(np.where(overall_mask, orig_total_docs, 0))[-100:]
    overall_freqs = {
        vocab[i]: float(orig_total_docs[i])
        for i in overall_idx
        if overall_mask[i] and orig_total_docs[i] > 0
    }
    wc_all = WordCloud(
        font_path=THAI_FONT, width=650, height=430, background_color="white",
        colormap="plasma", regexp=r"[\u0E00-\u0E7F]+", max_words=80
    ).generate_from_frequencies(overall_freqs)
    axes[0].imshow(wc_all)
    axes[0].axis("off")
    axes[0].set_title("รวม (All Stars)", fontsize=20, fontweight="bold")

    # Panels 1-5: Mutually exclusive signature words for each star rating (1 to 5)
    star_colormaps = {
        1: "Reds",
        2: "Oranges",
        3: "YlOrBr",
        4: "YlGnBu",
        5: "Greens",
    }
    star_labels = {
        1: "1 star (แย่ / ไม่ประทับใจ)",
        2: "2 star (ผิดหวัง / ไม่คุ้ม)",
        3: "3 star (ปานกลาง / พอใช้)",
        4: "4 star (ดี / อร่อย)",
        5: "5 star (ดีมาก / ประทับใจสุดๆ)",
    }

    print("\nคำเด่นจำแนกตามระดับดาว (Top Distinctive Words per Star):")
    for panel_idx, (c_idx, c) in enumerate(enumerate(nb.classes_), start=1):
        other_mean = np.delete(prob, c_idx, axis=0).mean(axis=0)
        lift = prob[c_idx] / (other_mean + 1e-12)
        min_docs = 6 if c == 1 else (14 if c == 2 else 35)
        # Strict exclusivity: each word appears only in the star class where its P(w|star) is highest
        exclusive_mask = (
            is_display_word
            & (orig_doc_counts[c_idx] >= min_docs)
            & (prob.argmax(axis=0) == c_idx)
        )
        distinct_weight = np.where(
            exclusive_mask,
            (prob[c_idx] ** 0.65) * (lift ** 1.35) * 10000.0,
            0.0,
        )
        top_c_idx = [i for i in np.argsort(distinct_weight)[::-1] if distinct_weight[i] > 0][:70]
        print(f"  {c} star:", [vocab[i] for i in top_c_idx[:10]])
        freqs = {vocab[i]: float(distinct_weight[i]) for i in top_c_idx}
        wc = WordCloud(
            font_path=THAI_FONT, width=650, height=430, background_color="white",
            colormap=star_colormaps[c], regexp=r"[\u0E00-\u0E7F]+", max_words=60
        ).generate_from_frequencies(freqs)
        axes[panel_idx].imshow(wc)
        axes[panel_idx].axis("off")
        axes[panel_idx].set_title(star_labels[c], fontsize=18, fontweight="bold")

    plt.suptitle("Word Cloud: รวม / 1 / 2 / 3 / 4 / 5 ดาว (จำแนกคำเด่นแต่ละระดับดาว)", fontsize=24, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "wordcloud.png"), dpi=120)
    plt.close()

    # ---------------------------------------------------------------- demo
    print("\nทดสอบทำนายประโยคตัวอย่าง (Demo Predictions):")
    for s in [
        "อาหารอร่อยมาก บริการดีมาก ประทับใจสุดๆ จะมาซ้ำแน่นอน",
        "รสชาติแย่มาก รอนาน พนักงานไม่สุภาพ ไม่ประทับใจเลย",
    ]:
        s_tok = " ".join(tokenize(s))
        print(f"  '{s}' -> {nb.predict(vec.transform([s_tok]))[0]} star")

    print("\nSaved: heatmap.png, wordcloud.png, confusion_matrix.png, output.txt")

    sys.stdout = tee.stream
    with open(os.path.join(OUT, "output.txt"), "w", encoding="utf-8") as f:
        f.write(tee.buffer.getvalue())
