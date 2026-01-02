# 🏦 Home Credit Default Risk Analizi | Uçtan Uca Makine Öğrenmesi Projesi

> 🇬🇧 **For English documentation [click here](README.md).**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Library](https://img.shields.io/badge/Library-Pandas_|_Seaborn-green?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-LightGBM-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Durum-Tamamland%C4%B1-success?style=for-the-badge)

## 📌 Proje Özeti
Bu projenin temel amacı, bankacılık sektöründeki kritik bir problemi çözmektir: **Kredi Batık Riski Tahmini (Credit Default Risk Prediction).**
Home Credit tarafından sağlanan tarihsel verileri kullanarak, müşteri davranışlarını analiz ettim ve bir borçlunun kredisini geri ödeyip ödemeyeceğini tahmin eden bir Makine Öğrenmesi modeli geliştirdim.

Bu proje sadece basit bir modelleme değil; derinlemesine **SQL Analizi**, **Özellik Mühendisliği (Feature Engineering)** ve gelişmiş Boosting algoritmaları kullanılarak **Dengesiz Veri (Imbalanced Data)** yönetimi süreçlerini içerir.

---

## 🛠️ Kullanılan Teknolojiler ve Akış
* **Veri Analizi:** SQL (SQLite) & Python (Pandas)
* **Görselleştirme:** Matplotlib & Seaborn
* **Makine Öğrenmesi:** Random Forest & **LightGBM** (Gradient Boosting)
* **Teknikler:** Dengesiz Veri için Sınıf Ağırlıklandırma (Class Weighting), Eşik Değeri Ayarlama (Threshold Tuning)

---

## 📊 1. Keşifçi Veri Analizi (EDA) ve Kritik Bulgular
Veriyi modele sokmadan önce, SQL ve Python kullanarak risk faktörlerini anlamak için detaylı bir analiz yaptım.

### 🚩 Bulgu 1: Yaş ve Risk İlişkisi
**Gözlem:** Genç müşteriler (<30 yaş), yaşlı müşterilere (>60 yaş) göre belirgin şekilde daha riskli.
* **20-30 Yaş Arası:** %11.46 Batık Oranı
* **60+ Yaş Üstü:** %4.92 Batık Oranı

![Yaş Risk Analizi](images/age_risk.png)

### 🚩 Bulgu 2: Aile Yapısının Etkisi
**Gözlem:** Çocuk sayısı ile risk arasında bir korelasyon var. 3 ve üzeri çocuğu olan aileler en yüksek batık riskini (%10.04) taşıyor.

![Çocuk Analizi](images/children_risk.png)

### 🚩 Bulgu 3: Kredi/Gelir Oranı ve "Survivor Bias"
**Gözlem:** Şaşırtıcı bir şekilde, gelirinin **6 katından fazla** kredi isteyenlerin riski, **3-6 kat** isteyenlerden daha düşük çıktı.
**Yorum:** Bu durum bankanın katı politikalarını gösteriyor; bu kadar büyük krediler sadece "süper nitelikli" müşterilere onaylandığı için risk yapay olarak düşük görünüyor (Survivor Bias).

![Finansal Analiz](images/credit_income.png)

---

## 🧠 2. Model Performansı (Çözüm)
Veri seti oldukça dengesizdi (sadece ~%8 batık oranı). Standart bir model, riskli müşterileri tespit etmekte başarısız olacaktı (Yüksek Doğruluk, Düşük Yakalama Oranı).
Bunu çözmek için `class_weight='balanced'` parametresi ile **LightGBM** kullandım.

### 🏆 Final Sonuçlar
| Metrik | Skor | Açıklama |
| :--- | :--- | :--- |
| **ROC-AUC** | **%75.26** | Sınıfları birbirinden ayırma yeteneği (Başarılı). |
| **True Positives** | **2,984** | Başarıyla tespit edilen potansiyel batık müşteriler. |
| **False Negatives** | **1,981** | Gözden kaçanlar (Baz modelden çok daha düşük). |

### 📉 Hata Matrisi (Confusion Matrix)
Model, aksi takdirde bankaya finansal zarar verecek olan **2,984** riskli müşteriyi başarıyla yakaladı.

![Confusion Matrix](images/confusion_matrix.png)

---

## 🔑 3. Özellik Önemi (Model Neye Göre Karar Verdi?)
İş değeri (Business Value) üretmek için modelin "Neden" karar verdiğini açıklamalıyız. LightGBM modeli şu özelliklere öncelik verdi:

1.  **EXT_SOURCE (1, 2, 3):** Dış kaynaklardan (KKB vb.) gelen kredi skorları en güçlü belirleyici.
2.  **DAYS_BIRTH:** Müşterinin yaşı kritik bir demografik faktör.
3.  **AMT_ANNUITY:** Aylık taksit yükü, ödeme kapasitesini doğrudan etkiliyor.

![Feature Importance](images/feature_importance.png)

---

## 🚀 Sonuç ve İşletmeye Etkisi
Basit bir Random Forest modelinden optimize edilmiş **LightGBM** modeline geçerek, riskli müşterileri tespit etme oranımızı **neredeyse sıfırdan ~%60'a** çıkardık.
Bu modelin uygulanması bankaya şunları sağlar:
* **Takipteki Kredilerin (NPL) Azaltılması.**
* **Risk Profillerine Göre Faiz** oranlarının optimize edilmesi.
* **Kredi Ön Değerlendirme** sürecinin otomatize edilmesi.

---
*Yazar: [Mehmetcan Mutlu]*