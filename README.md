# From Text to Data

### An Introduction to NLP for Empirical Research

Materials for the workshop held at the **DAISY International Summer School** — *Data Analytics for Innovation and Sustainability* — Taranto, September 2026.

---

## Open the notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martina-iori/DAISY_NLP_in_Python/blob/main/notebooks/DAISY_text_to_data.ipynb)

Click the badge above. Nothing to install: the notebook runs in the browser on Google Colab. You will need a Google account.

If you prefer to run it locally, see [Local setup](#local-setup) below.

---

## What the workshop covers

We classify patent documents as *green* or *non-green* technologies using four families of methods, and compare them on the same held-out data:

1. **Dictionary methods** — expert keyword lists, as used in much of the green-patent literature
2. **TF-IDF + logistic regression** — let the data pick the vocabulary
3. **Pretrained sentence embeddings + logistic regression** — capture meaning, no GPU budget required
4. **LLM zero-shot classification** — powerful, costly, and still in need of validation

---

## The data

`data/patstat_green_sample.csv` — a sample of patent applications extracted from **PATSTAT**.

| Column | Description |
|---|---|
| `appln_id` | PATSTAT application identifier |
| `title` | Patent title (English) |
| `abstract` | Patent abstract (English) |
| `earliest_filing_year` | Year of earliest application in the family |
| `green` | Binary label: 1 if the patent carries a green-technology CPC code (Y02), 0 otherwise |
| `cpc_subclasses` | List of 4-digit CPC codes |

**On the label.** The `green` variable is derived from the CPC classification scheme, not from expert human judgement of each document. It is a measurement instrument with its own inclusion criteria and its own error. Every performance figure in the notebook is accuracy *relative to that rule*. 

---

## Repository structure

```
├── README.md
├── notebooks/
|   ├── DAISY_text_to_data.ipynb     # the workshop notebook
│   └── precompute_llm_gemini.py     # LLM run
├── data/
|   ├── patstat_green_sample.csv     # dataset used in the workshop
|   ├── llm_val_predictions.csv      # LLM prediction
│   └── llm_run_metadata.json        # LLM metadata
└── requirements.txt                 # for local runs only
```

---

## Local setup

Not needed for the workshop, but useful afterwards.

```bash
git clone https://github.com/martina-iori/DAISY_NLP_in_Python.git
cd DAISY_NLP_in_Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/DAISY_text_to_data.ipynb
```

---

## Further reading

Ash, E. and S. Hansen (2023), "Text Algorithms in Economics", *Annual Review of Economics* 15: 659-688.
[Free PDF](https://sekhansen.github.io/pdf_files/are_2023.pdf) · [doi:10.1146/annurev-economics-082222-074352](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-082222-074352)

The sections on validation and on measurement error in downstream regressions are the most useful for empirical work.

The companion notebooks to that survey — a fuller short course covering preprocessing, LDA, word2vec, BERT feature extraction and fine-tuning — are at
[github.com/sekhansen/text_algorithms_econ](https://github.com/sekhansen/text_algorithms_econ).

---

## Contact

Martina Iori — Università Cattolica del Sacro Cuore, Milan

martina.iori@unicatt.it

Materials released for educational use.
