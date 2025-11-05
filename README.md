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

### Eğitim Parametreleri

- `--n_way`: Episode başına sınıf sayısı (varsayılan: 5)
- `--k_shot`: Sınıf başına destek örnek sayısı (varsayılan: 5)
- `--n_query`: Sınıf başına sorgu örnek sayısı (varsayılan: 15)
- `--epochs`: Eğitim epoch sayısı (varsayılan: 50)
- `--lr`: Öğrenme oranı (varsayılan: 0.001)
- `--embedding_dim`: Embedding boyutu (varsayılan: 128)
- `--use_clinical`: Klinik veri kullanımını etkinleştir

## 🔬 Model Mimarisi

### Prototypical Network

- **Encoder**: Pre-trained ResNet-50 (ImageNet) backbone
- **Embedding Space**: L2-normalized 128-dimensional embeddings
- **Prototype Computation**: Sınıf başına destek örneklerinin ortalaması
- **Classification**: Euclidean distance tabanlı sınıflandırma

### Multi-Modal Fusion (Opsiyonel)

- **Clinical Encoder**: Yaş, cinsiyet, tümör evresi, genetik mutasyonlar için embedding
- **Fusion Layer**: Görüntü ve klinik embedding'lerin birleştirilmesi

## 📊 Veri Formatı

### Metadata JSON Yapısı

```json
{
  "cancer_type_name": [
    {
      "image_id": "unique_id",
      "cancer_type": "cancer_type_name",
      "image_path": "path/to/image.png",
      "age": 55,
      "gender": "M",
      "tumor_stage": "II",
      "genetic_mutation": "BRAF",
      "tumor_size_mm": 35.5
    }
  ]
}
```

## ⚖️ Etik İlkeler

- Gerçek hasta verisi yalnızca kamuya açık anonim kaynaklardan alınır
- Hiçbir özel veya kurum verisi kullanılmaz
- Proje araştırma ve eğitim amaçlıdır, klinik karar verme aracı değildir
- Veri paylaşımı ve model çıktıları açık lisans altında yayınlanır

## 📚 Referanslar

- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. NeurIPS.
- Vinyals, O., et al. (2016). Matching networks for one shot learning. NeurIPS.

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen pull request göndermeden önce kod standartlarına uyduğunuzdan emin olun.

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not**: Bu proje araştırma ve eğitim amaçlıdır. Gerçek klinik uygulamalarda kullanılmadan önce kapsamlı validasyon gereklidir.
