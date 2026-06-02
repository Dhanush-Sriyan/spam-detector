import tkinter as tk
from tkinter import ttk
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model

model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_message():
    message = txt_message.get("1.0", tk.END).strip()


    if not message:
        result_label.config(text="Please enter a message")
        return

    cleaned = clean_text(message)
    vector = tfidf.transform([cleaned]).toarray()

    result = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0]

    score = round(max(confidence) * 100, 2)

    progress["value"] = score

    if result == 1:
        result_label.config(
            text=f"🚨 SPAM DETECTED\nConfidence: {score}%",
            fg="#ff4d4d"
        )
    else:
        result_label.config(
            text=f"✅ SAFE MESSAGE\nConfidence: {score}%",
            fg="#00cc66"
        )

def clear_text():
    txt_message.delete("1.0", tk.END)
    result_label.config(text="")
    progress["value"] = 0

root = tk.Tk()
root.title("Spam Detection System")
root.geometry("700x500")
root.configure(bg="#1e1e2f")

title = tk.Label(
root,
text="📩 AI Spam Detection System",
font=("Segoe UI", 20, "bold"),
bg="#1e1e2f",
fg="white"
)
title.pack(pady=15)

txt_message = tk.Text(
root,
width=70,
height=10,
font=("Segoe UI", 11),
bg="#2b2b3c",
fg="white",
insertbackground="white"
)
txt_message.pack(pady=10)

button_frame = tk.Frame(root, bg="#1e1e2f")
button_frame.pack()

check_btn = tk.Button(
button_frame,
text="Analyze Message",
command=predict_message,
font=("Segoe UI", 11, "bold"),
padx=15,
pady=5
)
check_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(
button_frame,
text="Clear",
command=clear_text,
font=("Segoe UI", 11),
padx=15,
pady=5
)
clear_btn.grid(row=0, column=1)

progress = ttk.Progressbar(
root,
orient="horizontal",
length=400,
mode="determinate"
)
progress.pack(pady=20)

result_label = tk.Label(
root,
text="",
font=("Segoe UI", 16, "bold"),
bg="#1e1e2f"
)
result_label.pack(pady=10)

footer = tk.Label(
root,
text="Machine Learning Project - Spam Classification",
font=("Segoe UI", 9),
bg="#1e1e2f",
fg="gray"
)
footer.pack(side="bottom", pady=10)

root.mainloop()
