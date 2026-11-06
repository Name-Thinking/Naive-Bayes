"""
Naive Bayes sentiment (star rating 1-5) on Wongnai reviews
Steps follow the whiteboard:
  1. training set / test set
  2. tokenize -> stop words removing (Thai has no stemming, so we normalize instead)
  3. TF table per class + Laplace smoothing (+1)
  4. P(class|doc) ∝ P(class) * Π P(word|class)   (OOV words are ignored)
  5. heat map, word cloud, performance test (accuracy)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Tahoma"  # Thai-capable font on Windows
THAI_FONT = r"C:\Windows\Fonts\tahoma.ttf"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 1. data
ds = load_dataset("Highgroundbkk/wongnai_reviews")
df = ds["train"].to_pandas()
df["star"] = df["star_rating"] + 1          # 0-4 -> 1-5 stars
print(df["star"].value_counts().sort_index())

# The official test split may be unlabeled, so evaluate on a held-out part of train
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["star"])

# ---------------------------------------------------------------- 2. preprocessing
STOP = set(thai_stopwords())

def tokenize(text: str) -> list[str]:
    toks = word_tokenize(normalize(text.lower()), engine="newmm", keep_whitespace=False)
    return [t for t in toks if t.strip() and t not in STOP and not t.isdigit() and len(t) > 1]

# ---------------------------------------------------------------- 3. TF table
vec = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, min_df=3)
X_train = vec.fit_transform(train_df["review_body"])
X_test = vec.transform(test_df["review_body"])          # OOV words simply dropped
y_train, y_test = train_df["star"].values, test_df["star"].values
print("Vocabulary size:", len(vec.vocabulary_))

# ---------------------------------------------------------------- 4. Naive Bayes (alpha=1 -> +1 smoothing)
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)

# ---------------------------------------------------------------- 5a. performance test
pred = nb.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred, zero_division=0))

cm = confusion_matrix(y_test, pred, labels=[1, 2, 3, 4, 5])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel("Predicted star"); plt.ylabel("True star"); plt.title("Confusion matrix")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120); plt.close()

# ---------------------------------------------------------------- 5b. heat map of P(word|class)
vocab = np.array(vec.get_feature_names_out())
prob = np.exp(nb.feature_log_prob_)                     # smoothed (count+1)/(total+|V|)
top_idx = np.unique(np.concatenate([np.argsort(p)[-6:] for p in prob]))
heat = pd.DataFrame(prob[:, top_idx], index=[f"{c} star" for c in nb.classes_], columns=vocab[top_idx])
plt.figure(figsize=(14, 4))
sns.heatmap(heat, cmap="YlOrRd", annot=False)
plt.title("P(word | star) – top words per class (Laplace smoothed)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "heatmap.png"), dpi=120); plt.close()

# ---------------------------------------------------------------- 5c. word cloud per star
counts = np.asarray(nb.feature_count_)
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, c, row in zip(axes, nb.classes_, counts):
    freqs = {vocab[i]: row[i] for i in np.argsort(row)[-150:] if row[i] > 0}
    wc = WordCloud(font_path=THAI_FONT, width=500, height=400, background_color="white",
                   regexp=r"[\u0E00-\u0E7Fa-zA-Z]+").generate_from_frequencies(freqs)
    ax.imshow(wc); ax.axis("off"); ax.set_title(f"{c} star", fontsize=18)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "wordcloud.png"), dpi=100); plt.close()

# ---------------------------------------------------------------- demo
for s in ["อาหารอร่อยมาก บริการดี จะมาอีก", "รสชาติแย่ รอนาน พนักงานไม่สุภาพ"]:
    print(s, "->", nb.predict(vec.transform([s]))[0], "star")
print("\nSaved: heatmap.png, wordcloud.png, confusion_matrix.png")
