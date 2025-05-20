!pip install numpy pandas matplotlib scikit-learn nltk tqdm gensim torch transformers


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import torch
from google.colab import files

uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])



import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
import torch
import gensim.downloader as api
from google.colab import files

uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
# ==== Tokenisasi + Case Folding + Stopword Removal ====
def tokenize(sentence):
    sentence = re.sub(r"[^\w\s]", "", sentence.lower())
    tokens = sentence.split()
    filtered = [word for word in tokens if word not in stop_words]
    return filtered

# ==== Untuk BERT (cleaned string, bukan token list) ====
def clean_for_bert(sentence):
    sentence = re.sub(r"[^\w\s]", "", sentence.lower())
    tokens = sentence.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ==== Load DataFrame dan Penyesuaian Kolom ====
col_map = {}
if "premise" in df.columns and "hypothesis" in df.columns:
    col_map = {"premise": "question1", "hypothesis": "question2"}
elif "question1" not in df.columns or "question2" not in df.columns:
    raise ValueError("File CSV harus memiliki kolom: question1 & question2, atau premise & hypothesis")

df.rename(columns=col_map, inplace=True)

if "label" not in df.columns:
    print("⚠ Kolom 'label' tidak ditemukan. Evaluasi klasifikasi akan dilewati.")
    df["label"] = np.nan

# ==== Load Model ====
print("📥 Loading Word2Vec...")
w2v_model = api.load("word2vec-google-news-300")

print("📥 Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# ==== Word2Vec Embedding ====
def sentence_to_avg_vector(sentence, model):
    tokens = tokenize(sentence)
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# ==== BERT Embedding ====
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden = outputs.last_hidden_state.squeeze(0)
    mask = inputs["attention_mask"].squeeze(0).unsqueeze(1).expand(last_hidden.size()).float()
    masked_embeddings = last_hidden * mask
    summed = torch.sum(masked_embeddings, 0)
    counted = torch.clamp(mask.sum(0), min=1e-9)
    return (summed / counted).numpy()

# ==== Proses Embedding dan Cosine Similarity ====
w2v_similarities = []
bert_similarities = []

print("🔄 Menghitung Cosine Similarity untuk setiap pasangan...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    vec1_w2v = sentence_to_avg_vector(row["question1"], w2v_model)
    vec2_w2v = sentence_to_avg_vector(row["question2"], w2v_model)
    sim_w2v = min(cosine_similarity([vec1_w2v], [vec2_w2v])[0][0], 1.0)
    w2v_similarities.append(sim_w2v)

    vec1_bert = get_bert_embedding(clean_for_bert(row["question1"]))
    vec2_bert = get_bert_embedding(clean_for_bert(row["question2"]))
    sim_bert = min(cosine_similarity([vec1_bert], [vec2_bert])[0][0], 1.0)
    bert_similarities.append(sim_bert)

df["cosine_similarity_word2vec"] = w2v_similarities
df["cosine_similarity_bert"] = bert_similarities

# ==== Threshold & Prediksi ====
threshold = 0.85
df["pred_w2v"] = df["cosine_similarity_word2vec"] >= threshold
df["pred_bert"] = df["cosine_similarity_bert"] >= threshold
df["pred_both"] = df["pred_w2v"] & df["pred_bert"]



col_map = {}
if "premise" in df.columns and "hypothesis" in df.columns:
    col_map = {"premise": "question1", "hypothesis": "question2"}
elif "question1" not in df.columns or "question2" not in df.columns:
    raise ValueError("File CSV harus memiliki kolom: question1 & question2, atau premise & hypothesis")

df.rename(columns=col_map, inplace=True)

if "label" not in df.columns:
    print("⚠ Kolom 'label' tidak ditemukan. Evaluasi klasifikasi akan dilewati.")
    df["label"] = np.nan

# ==== Load Model ====
print("📥 Loading Word2Vec...")
w2v_model = api.load("word2vec-google-news-300")

print("📥 Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# ==== Tokenisasi ====
def tokenize(sentence):
    sentence = re.sub(r"[^\w\s]", "", sentence.lower())
    return sentence.split()

# ==== Word2Vec Embedding ====
def sentence_to_avg_vector(sentence, model):
    tokens = tokenize(sentence)
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# ==== BERT Embedding ====
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden = outputs.last_hidden_state.squeeze(0)
    mask = inputs["attention_mask"].squeeze(0).unsqueeze(1).expand(last_hidden.size()).float()
    masked_embeddings = last_hidden * mask
    summed = torch.sum(masked_embeddings, 0)
    counted = torch.clamp(mask.sum(0), min=1e-9)
    return (summed / counted).numpy()

# ==== Proses Embedding dan Cosine Similarity ====
w2v_similarities = []
bert_similarities = []

print("🔄 Menghitung Cosine Similarity untuk setiap pasangan...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    vec1_w2v = sentence_to_avg_vector(row["question1"], w2v_model)
    vec2_w2v = sentence_to_avg_vector(row["question2"], w2v_model)
    sim_w2v = min(cosine_similarity([vec1_w2v], [vec2_w2v])[0][0], 1.0)
    w2v_similarities.append(sim_w2v)

    vec1_bert = get_bert_embedding(row["question1"])
    vec2_bert = get_bert_embedding(row["question2"])
    sim_bert = min(cosine_similarity([vec1_bert], [vec2_bert])[0][0], 1.0)
    bert_similarities.append(sim_bert)


df["cosine_similarity_word2vec"] = w2v_similarities
df["cosine_similarity_bert"] = bert_similarities

# ==== Threshold & Prediksi ====
threshold = 0.85
df["pred_w2v"] = df["cosine_similarity_word2vec"] >= threshold
df["pred_bert"] = df["cosine_similarity_bert"] >= threshold
df["pred_both"] = df["pred_w2v"] & df["pred_bert"]





# ==== Evaluasi ====
if df["label"].notnull().all():
    df["label"] = df["label"].astype(bool)

    def evaluate_model(y_true, y_pred, method_name):
        print(f"\n📊 Evaluasi {method_name}")
        print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
        print("Precision:", round(precision_score(y_true, y_pred), 4))
        print("Recall   :", round(recall_score(y_true, y_pred), 4))
        print("F1 Score :", round(f1_score(y_true, y_pred), 4))

    evaluate_model(df["label"], df["pred_w2v"], "Word2Vec")
    evaluate_model(df["label"], df["pred_bert"], "BERT")
else:
    print("⚠ Kolom 'label' tidak lengkap, evaluasi klasifikasi dilewati.")

print("✅ Proses selesai. File hasil telah disimpan.")


# Histogram
bins = np.linspace(0, 1, 11)
hist_w2v, _ = np.histogram(df["cosine_similarity_word2vec"], bins=bins)
hist_bert, _ = np.histogram(df["cosine_similarity_bert"], bins=bins)

plt.figure(figsize=(10, 5))
x = np.arange(len(hist_w2v))
plt.bar(x - 0.2, hist_w2v, width=0.4, label="Word2Vec")
plt.bar(x + 0.2, hist_bert, width=0.4, label="BERT")
plt.xticks(x, [f"{round(bins[i],1)}-{round(bins[i+1],1)}" for i in range(len(bins)-1)])
plt.xlabel("Range Cosine Similarity")
plt.ylabel("Jumlah Pasangan Pertanyaan")
plt.title("Distribusi Cosine Similarity Word2Vec vs BERT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gunakan average='binary' → evaluasi hanya pada kelas positif (label = 1)
acc_w2v = accuracy_score(df["label"], df["pred_w2v"])
prec_w2v = precision_score(df["label"], df["pred_w2v"], average='binary', zero_division=0)
rec_w2v = recall_score(df["label"], df["pred_w2v"], average='binary', zero_division=0)
f1_w2v = f1_score(df["label"], df["pred_w2v"], average='binary', zero_division=0)

acc_bert = accuracy_score(df["label"], df["pred_bert"])
prec_bert = precision_score(df["label"], df["pred_bert"], average='binary', zero_division=0)
rec_bert = recall_score(df["label"], df["pred_bert"], average='binary', zero_division=0)
f1_bert = f1_score(df["label"], df["pred_bert"], average='binary', zero_division=0)


# Grafik evaluasi
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
word2vec_scores = [acc_w2v, prec_w2v, rec_w2v, f1_w2v]
bert_scores = [acc_bert, prec_bert, rec_bert, f1_bert]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, word2vec_scores, width, label='Word2Vec')
plt.bar(x + width/2, bert_scores, width, label='BERT')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Evaluasi Model: Word2Vec vs BERT')
plt.xticks(ticks=x, labels=metrics)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df["cosine_similarity_word2vec"], df["cosine_similarity_bert"], alpha=0.5)
plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold BERT")
plt.axvline(x=threshold, color="g", linestyle="--", label="Threshold Word2Vec")
plt.xlabel("Word2Vec Similarity")
plt.ylabel("BERT Similarity")
plt.title("Perbandingan Cosine Similarity Word2Vec vs BERT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Jumlah prediksi mirip dan tidak mirip oleh Word2Vec
jumlah_mirip_w2v = df["pred_w2v"].sum()
jumlah_tidak_mirip_w2v = len(df) - jumlah_mirip_w2v

# Jumlah prediksi mirip dan tidak mirip oleh BERT
jumlah_mirip_bert = df["pred_bert"].sum()
jumlah_tidak_mirip_bert = len(df) - jumlah_mirip_bert

# Jumlah prediksi mirip oleh keduanya
jumlah_mirip_keduanya = df["pred_both"].sum()
jumlah_tidak_mirip_keduanya = len(df) - jumlah_mirip_keduanya

# Tampilkan hasil
print("\n📊 Ringkasan Jumlah Prediksi:")
print("Word2Vec - Mirip        :", jumlah_mirip_w2v)
print("Word2Vec - Tidak Mirip :", jumlah_tidak_mirip_w2v)
print("BERT - Mirip            :", jumlah_mirip_bert)
print("BERT - Tidak Mirip      :", jumlah_tidak_mirip_bert)
print("Keduanya - Mirip        :", jumlah_mirip_keduanya)
print("Keduanya - Tidak Mirip :", jumlah_tidak_mirip_keduanya)


# ==== Paling Mirip ====
top_sim_w2v = df.loc[df["cosine_similarity_word2vec"].idxmax()]
top_sim_bert = df.loc[df["cosine_similarity_bert"].idxmax()]

# ==== Paling Tidak Mirip ====
low_sim_w2v = df.loc[df["cosine_similarity_word2vec"].idxmin()]
low_sim_bert = df.loc[df["cosine_similarity_bert"].idxmin()]

# ==== Tampilkan ke layar ====
print("\n🔝 Word2Vec - Paling Mirip:")
print("Pertanyaan 1:", top_sim_w2v["question1"])
print("Pertanyaan 2:", top_sim_w2v["question2"])
print("Cosine Similarity:", round(top_sim_w2v["cosine_similarity_word2vec"], 4))

print("\n🔝 BERT - Paling Mirip:")
print("Pertanyaan 1:", top_sim_bert["question1"])
print("Pertanyaan 2:", top_sim_bert["question2"])
print("Cosine Similarity:", round(top_sim_bert["cosine_similarity_bert"], 4))

print("\n🔻 Word2Vec - Paling Tidak Mirip:")
print("Pertanyaan 1:", low_sim_w2v["question1"])
print("Pertanyaan 2:", low_sim_w2v["question2"])
print("Cosine Similarity:", round(low_sim_w2v["cosine_similarity_word2vec"], 4))

print("\n🔻 BERT - Paling Tidak Mirip:")
print("Pertanyaan 1:", low_sim_bert["question1"])
print("Pertanyaan 2:", low_sim_bert["question2"])
print("Cosine Similarity:", round(low_sim_bert["cosine_similarity_bert"], 4))


jumlah_sim_1_w2v = (df["cosine_similarity_word2vec"] == 1.0).sum()
jumlah_sim_1_bert = (df["cosine_similarity_bert"] == 1.0).sum()

print("\n📌 Jumlah Pasangan dengan Cosine Similarity = 1.0")
print("Word2Vec :", jumlah_sim_1_w2v)
print("BERT     :", jumlah_sim_1_bert)


# =======================
# RATA-RATA SIMILARITY
# =======================

# Word2Vec
mean_sim_w2v_mirip = df[df["pred_w2v"]]["cosine_similarity_word2vec"].mean()
mean_sim_w2v_tidak = df[~df["pred_w2v"]]["cosine_similarity_word2vec"].mean()

# BERT
mean_sim_bert_mirip = df[df["pred_bert"]]["cosine_similarity_bert"].mean()
mean_sim_bert_tidak = df[~df["pred_bert"]]["cosine_similarity_bert"].mean()

# Kombinasi (keduanya setuju = True)
mean_sim_both_mirip_w2v = df[df["pred_both"]]["cosine_similarity_word2vec"].mean()
mean_sim_both_mirip_bert = df[df["pred_both"]]["cosine_similarity_bert"].mean()
mean_sim_both_tidak_w2v = df[~df["pred_both"]]["cosine_similarity_word2vec"].mean()
mean_sim_both_tidak_bert = df[~df["pred_both"]]["cosine_similarity_bert"].mean()

# =======================
# TAMPILKAN HASIL
# =======================
print("🔷 Word2Vec:")
print(f"Rata-rata similarity untuk pasangan MIRIP       : {mean_sim_w2v_mirip:.4f}")
print(f"Rata-rata similarity untuk pasangan TIDAK MIRIP : {mean_sim_w2v_tidak:.4f}")

print("\n🟠 BERT:")
print(f"Rata-rata similarity untuk pasangan MIRIP       : {mean_sim_bert_mirip:.4f}")
print(f"Rata-rata similarity untuk pasangan TIDAK MIRIP : {mean_sim_bert_tidak:.4f}")

print("\n🔗 Keduanya (kombinasi Word2Vec & BERT):")
print(f"Rata-rata Word2Vec (MIRIP): {mean_sim_both_mirip_w2v:.4f}")
print(f"Rata-rata BERT     (MIRIP): {mean_sim_both_mirip_bert:.4f}")
print(f"Rata-rata Word2Vec (TIDAK MIRIP): {mean_sim_both_tidak_w2v:.4f}")
print(f"Rata-rata BERT     (TIDAK MIRIP): {mean_sim_both_tidak_bert:.4f}")


from google.colab import files

# ==== SIMPAN FILE BERDASARKAN KATEGORI ====

# Word2Vec
df[df["pred_w2v"]].to_csv("mirip_word2vec.csv", index=False)
df[~df["pred_w2v"]].to_csv("tidak_mirip_word2vec.csv", index=False)

# BERT
df[df["pred_bert"]].to_csv("mirip_bert.csv", index=False)
df[~df["pred_bert"]].to_csv("tidak_mirip_bert.csv", index=False)

# Keduanya
df[df["pred_both"]].to_csv("mirip_keduanya.csv", index=False)
df[~df["pred_both"]].to_csv("tidak_mirip_keduanya.csv", index=False)

# ==== DOWNLOAD SEMUA FILE ====

files.download("mirip_word2vec.csv")
files.download("tidak_mirip_word2vec.csv")

files.download("mirip_bert.csv")
files.download("tidak_mirip_bert.csv")

files.download("mirip_keduanya.csv")
files.download("tidak_mirip_keduanya.csv")


from google.colab import files

# Unduh semua file hasil
files.download("hasil_perbandingan_bert_word2vec.csv")
files.download("hasil_mirip_word2vec.csv")
files.download("hasil_mirip_bert.csv")
files.download("hasil_mirip_keduanya.csv")


from google.colab import files

# Simpan DataFrame hasil lengkap
df.to_csv("hasil_evaluasi_bert_word2vec.csv", index=False)

# Unduh file hasil utama
files.download("hasil_evaluasi_bert_word2vec.csv")
