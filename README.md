# MuSiC: Multi-Task Multimodal Architecture for Memes’ Analysis

We propose **MuSiC**, a multitask multimodal architecture for analyzing sentiments and emotions present in memes, evaluated on the Memotion 2 and Memotion 3 datasets.

MuSiC jointly models the image and OCR-extracted text of each meme using:

- Image encoder: CLIP's image encoder 
- Text encoders: experimented with the following text encoders  
  - CLIP’s text encoder  
  - BERT  
  - XLM_RoBERTa  
  - MuRIL

On top of these encoders, MuSiC explores mainly three feature fusion strategies:

1. Simple concatenation (baseline)  
2. Cross-attention fusion  
3. Gated multimodal fusion

The fused representation is then passed through a shared MLP/ multi-task decision head for:

- Task A — sentiment classification: positive / neutral / negative  
- Task B — emotion classification: humorous / sarcastic / motivational / offensive

---

## 🔍 Datasets

We evaluate MuSiC on:

- **Memotion 2**  
  - 8,500 English memes  
  - Sentiment: Positive / Negative / Neutral  
  - Emotion: Humor, Sarcasm, Offense, Motivation (with subcategories)  
  - Strong class imbalance, especially in sentiment labels

- **Memotion 3**  
  - 8,500 Hinglish (Hinglish [Hindi+English] code-mixed) memes  
  - Same sentiment and emotion schema as Memotion 2  
  - More balanced sentiment distribution  
  - Provides a realistic and challenging benchmark for multilingual, code-mixed meme analysis

The Memotion 2 and Memotion 3 datasets used in this project were obtained directly from the dataset organizers and are not included in this repository.  
Please request access via the official Memotion challenge channels or the links provided in the original dataset papers, and place the downloaded files locally.  
After obtaining the data, update the dataset paths in each notebook to run the experiments and reproduce the results.

Both datasets provide **images** and **OCR text** for each meme, enabling genuine multimodal modeling.

---

## 🧹 Data Preprocessing

**Images**

- Convert BGR → RGB  
- Resize to a standard resolution  
- Data augmentation (e.g., horizontal flips, brightness, shift–scale operations)  
- Normalize using standard image statistics

**Text (OCR captions)**

- Lowercasing  
- Expanding contractions and continuous expressions (e.g., `"you’re"` → `"you are"`)  
- Removing emails, HTML tags, special characters, and accented characters  
- Tokenization with the corresponding encoder’s tokenizer (CLIP/BERT/RoBERTa/MuRIL)  
- Padding/truncation to a fixed maximum sequence length

This pipeline ensures cleaner, more consistent inputs for both modalities.

---

## 🧠 Model Overview

For each meme \(i\):

- Image: \(I_i\)  
- OCR text: \(T_i = \{ w_1, \dots, w_n \}\)

### Visual Feature Extraction

- CLIP ViT-based image encoder  
- Extract visual feature embedding:
  \[
  \text{VFE}_i = f_{\text{ViT}}(I_i)
  \]

### Textual Feature Extraction

- Transformer-based text encoder \(f_{\text{TE}}\) (CLIP text / BERT / RoBERTa / MuRIL)  
- Use the final \([\text{CLS}]\) token representation:
  \[
  \text{TFE}_i = f_{\text{TE}}(T_i)
  \]

### Feature Fusion Strategies

1. **Baseline Concatenation**  
   - L2-normalize \(\text{VFE}_i\) and \(\text{TFE}_i\)  
   - Concatenate:
     \[
     f_{\text{combined}} = \text{concat}(\bar{\text{VFE}}_i, \bar{\text{TFE}}_i)
     \]

2. **Cross-Attention Fusion**  
   - Project \(\text{VFE}_i\) and \(\text{TFE}_i\) into a shared space  
   - Apply bidirectional cross-attention (image↔text)  
   - L2-normalize attended features and concatenate to form \(f_{\text{combined}}\)

3. **Gated Multimodal Fusion**  
   - Concatenate \(\text{VFE}_i\) and \(\text{TFE}_i\) to form \(h_i\)  
   - Compute gate \(g_i = \sigma(W_g h_i + b_g)\)  
   - Combine:
     \[
     F_i^{\text{gated}} = g_i \odot \text{VFE}_i + (1 - g_i) \odot \text{TFE}_i
     \]
   - L2-normalize to obtain \(f_{\text{combined}}\)

### Multi-Task Decision Stage

- Shared 3-layer MLP with BatchNorm, GELU, Dropout  
- Task-specific final linear layers:
  - **Task A:** 3-way sentiment classification  
  - **Task B:** 4-way emotion classification  
- Optimization with cross-entropy; final predictions via Softmax.

---

## 📊 Evaluation

To handle class imbalance, we report:

- **F1-weighted**  
- **F1-macro**  
- **Accuracy** (for completeness)

where:

- F1 per class combines precision and recall  
- F1-weighted averages per-class F1 scores weighted by class support  
- F1-macro averages F1 uniformly across classes

---

## 📁 Repository Structure

The main experiment notebooks are:

- `CLIP_Memotion3_M1.ipynb`  
  CLIP image + CLIP text encoder with simple feature concatenation (baseline) for sentiment classification on Memotion 3.

- `CLIP_BERT_Memotion3.ipynb`  
  CLIP image + BERT text encoder with simple concatenation fusion for sentiment classification on Memotion 3.

- `CLIP_BERT_cross_attention_fusion.ipynb`  
  CLIP image + BERT text with cross-attention based feature fusion for sentiment classification on Memotion 3. 

- `CLIP+XLM_RoBERTa_sentiment_classification.ipynb`  
  CLIP image + XLM-RoBERTa text encoder with cross-attention based feature fusion for sentiment classification on Memotion 3.

- `CLIP_MuRIL_Memotion3_sentiment_classification.ipynb`
  CLIP image + MuRIL text encoder with gated multimodal fusion for sentiment classification on Memotion 3.  

- `CLIP_BERT_GatedMultimodalFusion.ipynb`  
  CLIP image + BERT text with gated multimodal fusion for sentiment classification on Memotion 3. 

- `Memotion_2_Sentiment_Classification.ipynb` 
  CLIP image + BERT text encoder with gated multimodal fusion for sentiment classification on Memotion 2.  

- `Memotion2(Sarcasm).ipynb`
  Best MuSiC variant for sarcasm classification on Memotion 2.

- `Memotion2_Humour.ipynb`
  Best MuSiC variant for humour classification on Memotion 2. 

- `Memotion2_Motivation.ipynb`
  Best MuSiC variant for motivation detection on Memotion 2.

- `Memotion_3_Motivation.ipynb`
  Best MuSiC variant for motivation detection on Memotion 3.

- `Memotion_3_emotion_offensive.ipynb`
  Best MuSiC variant for offense detection on Memotion 3. 

- `Memotion_3_sarcasm_detection.ipynb`
  Best MuSiC variant for sarcasm classification on Memotion 3.           

- `CLIP_Memotion3_M3.ipynb`  
  CLIP image + CLIP text with additional vision + text attention mechanisms for sentiment classification on Memotion 3.

- `CLIP_Memotion3_vision_attention.ipynb`  
  CLIP image encoder + CLIP text with enhanced vision attention for sentiment classification on Memotion 3.
    

---

## ⚙️ Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/Saihaj-coder/SI670_Final_Project.git
   cd SI670_Final_Project
   ```

2. **Install dependencies**

   Use `pip` or `conda` to install the packages listed above (PyTorch, transformers, pandas, numpy, scikit-learn, albumentations, opencv-python, Pillow, matplotlib, seaborn, text_hammer, etc.).

3. **Prepare the datasets**

   - Request access to Memotion 2 and Memotion 3 from the official organizers/authors.  
   - Download the images and label CSV files once access is granted.  
   - Arrange the data in a directory structure similar to:

     ```text
     data/
       Memotion2/
         trainImages/
         valImages/
         train.csv
         val.csv
       Memotion3/
         trainImages/
         valImages/
         train.csv
         val.csv
     ```

   - Update the dataset paths at the top of each notebook, for example:

     ```python
     train_image_dir = "/path/to/data/Memotion3/trainImages/"
     val_image_dir   = "/path/to/data/Memotion3/valImages/"
     train_csv       = "/path/to/data/Memotion3/train.csv"
     val_csv         = "/path/to/data/Memotion3/val.csv"
     ```

4. **Run experiments**

   - Open the desired notebook in Jupyter, Google Colab, or VS Code.  
   - Make sure the runtime has access to a GPU if possible (training is significantly faster).  
   - Run all cells to:
     - Load and preprocess the images and OCR text  
     - Initialize the chosen MuSiC variant (encoder + fusion strategy)  
     - Train on Memotion 2 or Memotion 3  
     - Evaluate using F1-weighted, F1-macro, and validation accuracy  
     - Generate t-SNE plots, training/validation curves, confusion matrices, precision-recall curves, etc.

5. **Adapting to other datasets or tasks**

   - Replace the CSV paths and image directories with your own dataset.  
   - Adjust the label loading and the final layer dimensions (number of classes) as needed.  
   - The overall MuSiC pipeline (encoders → fusion → multi-task heads) remains the same.

---

## 🔬 Experimental Results (High-Level Summary)

- On **Memotion 2**, MuSiC outperforms several strong baselines for both sentiment and emotion tasks, and remains competitive with more complex multi-branch architectures (e.g., those using EfficientNet, CLIP, and Sentence-BERT with multi-task transformers).  
- On **Memotion 3**, MuSiC achieves state-of-the-art performance on **Task A (sentiment classification)** and highly competitive results on **Task B (emotion classification)**, trailing only one prior method by a small margin.  
- Ablation studies comparing **CLIP–BERT concatenation** versus **CLIP–BERT cross-attention fusion** show that attention-based fusion produces more discriminative feature spaces (as seen in t-SNE visualizations) and improved F1 scores.

For detailed quantitative results and full tables, please refer to the project report.

---



