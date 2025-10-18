import instaloader
import pandas as pd
from transformers import pipeline
from tqdm import tqdm


# --- STEP 1: Ambil komentar Instagram ---


def ambil_komentar(username, shortcode):
    L = instaloader.Instaloader()
    L.load_session_from_file("your_username")

    post = instaloader.Post.from_shortcode(L.context, shortcode)
    comments = []
    for c in post.get_comments():
        comments.append({"user": c.owner.username, "text": c.text})
    return comments


# --- STEP 2: Analisis komentar dengan LLM ---
def analisis_komentar(df):
    sentiment_analyzer = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    sentiments = []
    for text in tqdm(df["text"], desc="Analisis sentimen"):
        result = sentiment_analyzer(text[:512])[0]
        sentiments.append(result["label"])
    df["sentiment"] = sentiments

    summary_input = " ".join(df["text"].tolist())[:4000]
    summary = summarizer(summary_input, max_length=130, min_length=30, do_sample=False)[
        0
    ]["summary_text"]

    return df, summary


# --- MAIN PROGRAM ---
if __name__ == "__main__":
    username = "instagram"  # ganti dengan akun target
    shortcode = "Cz4KPnKPIHt"  # ID unik postingan (bisa dari URL)

    print("📥 Mengambil komentar dari Instagram...")
    df = ambil_komentar(username, shortcode)
    print(f"✅ {len(df)} komentar berhasil diambil.")

    print("🤖 Menganalisis dengan model LLM...")
    hasil, ringkasan = analisis_komentar(df)

    hasil.to_csv("output/hasil.csv", index=False)
    print("\n📄 Hasil disimpan ke output/hasil.csv")
    print("\n🧠 Ringkasan:")
    print(ringkasan)
