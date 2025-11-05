# RareCan-AI: Few-Shot Learning for Rare Cancer Diagnosis

Bu proje, nadir görülen kanser tiplerinin (örneğin pankreas nöroendokrin tümörleri, sarkom alt tipleri, tiroid medüller kanserleri) tanısında yapay zekâ desteği geliştirmeyi hedefler.

## 🎯 Proje Hedefleri

Nadir kanser tiplerinde veri yetersizliği nedeniyle klasik derin öğrenme modelleri iyi genelleşemez. Bu proje, **few-shot learning (az örnekle öğrenme)** yöntemlerini kullanarak daha az veriyle daha doğru tanı koymayı amaçlar.

## 📋 Özellikler

- **Prototypical Networks** ile few-shot öğrenme
- **Multi-modal fusion**: Histopatoloji görüntüleri + klinik veriler
- **Episode-based training**: N-way K-shot öğrenme stratejisi
- **Sentetik veri üretimi**: Test ve geliştirme için örnek veri seti
- **Açık kaynak**: Tüm kod ve dokümantasyon açık lisans altında

## 🏗️ Proje Yapısı

```
rarecan-ai/
├── data/
│   ├── synthetic_generator.py      # Sentetik veri üretici
│   └── sample/                     # Üretilen örnek görüntüler + metadata
├── models/
│   └── protonet.py                 # Prototypical Network modeli
├── src/
│   ├── dataset.py                  # Episode dataset (few-shot episode sampling)
│   ├── train.py                    # Eğitim döngüsü
│   ├── evaluate.py                 # Değerlendirme scripti
│   └── utils.py                    # Yardımcı fonksiyonlar
├── notebooks/
│   └── 01_experiments.ipynb        # Deney notebook'u
├── checkpoints/                     # Eğitilmiş modeller (git ignore)
├── results/                         # Değerlendirme sonuçları (git ignore)
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Kurulum

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Hızlı Test (Demo)

Projeyi hızlıca test etmek için:

```bash
python run_demo.py
```

### 3. Sentetik Veri Üretimi

Test ve geliştirme için örnek veri seti oluşturun:

```bash
python data/synthetic_generator.py
```

Bu komut, `data/sample/` dizininde 5 nadir kanser tipi için sentetik histopatoloji görüntüleri ve metadata oluşturur.

## 💻 Kullanım

### Eğitim

```bash
# Sadece görüntü verisi ile
python src/train.py --data_dir data/sample --metadata_path data/sample/metadata.json

# Görüntü + klinik veri ile
python src/train.py --data_dir data/sample --metadata_path data/sample/metadata.json --use_clinical
```

### Değerlendirme

```bash
python src/evaluate.py --checkpoint_path checkpoints/best_model.pt --data_dir data/sample --metadata_path data/sample/metadata.json
```

## ⚖️ Etik İlkeler

- Gerçek hasta verisi yalnızca kamuya açık anonim kaynaklardan alınır
- Hiçbir özel veya kurum verisi kullanılmaz
- Proje araştırma ve eğitim amaçlıdır, klinik karar verme aracı değildir
- Veri paylaşımı ve model çıktıları açık lisans altında yayınlanır

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
