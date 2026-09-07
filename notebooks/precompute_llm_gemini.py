import os
import re
import time
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from google import genai
from google.genai import types

DATA_PATH    = 'patstat_green_sample.csv'
RANDOM_STATE = 42
OUT_PATH     = 'llm_val_predictions.csv'

MODEL = 'gemini-3.1-flash-lite'

SLEEP_BETWEEN_CALLS = 1.0

PROMPT = """You are classifying patent documents.

Decide whether the patent below describes a GREEN technology, meaning a technology
whose purpose is climate change mitigation or adaptation: renewable energy generation,
energy storage for decarbonisation, energy efficiency, carbon capture, waste recycling
and circular material recovery, clean transport, or emission abatement.

Answer with exactly one word: green or nongreen.

Title: {title}
Abstract: {abstract}"""

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

def build_config(disable_thinking=True):
    kw = dict(temperature=0, max_output_tokens=200)
    if disable_thinking:
        try:
            kw['thinking_config'] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
    return types.GenerateContentConfig(**kw)


CONFIG = build_config()

def parse(answer):
    """Map the model's text to 1 (green), 0 (nongreen), or None."""
    if not answer:
        return None
    a = answer.strip().lower()
    a = re.sub(r'[^a-z\- ]', '', a)
    if 'nongreen' in a or 'non-green' in a or 'non green' in a:
        return 0
    if 'green' in a:
        return 1
    return None


def classify(title, abstract, retries=4):
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT.format(title=title, abstract=abstract[:1500]),
                config=CONFIG,
            )
            out = parse(resp.text)
            if out is None:
                print(f'  unparseable: {(resp.text or "")[:60]!r}')
            return out
        except Exception as e:
            wait = 2 ** attempt * 5      # 5, 10, 20, 40s -- rate limits need patience
            print(f'  {e.__class__.__name__}, retrying in {wait}s')
            time.sleep(wait)
    return None

df = pd.read_csv(DATA_PATH)
df['title'] = df['title'].fillna('')
df['abstract'] = df['abstract'].fillna('')
df['text'] = (df['title'] + '. ' + df['abstract']).str.strip()
df = df[df['text'].str.split().apply(len) >= 5]
df = df.drop_duplicates(subset='text').reset_index(drop=True)

train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df['green'], random_state=RANDOM_STATE)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['green'], random_state=RANDOM_STATE)

print(f'Validation set: {len(val_df)} documents')

done = {}
if os.path.exists(OUT_PATH):
    prev = pd.read_csv(OUT_PATH)
    done = dict(zip(prev['appln_id'], prev['llm_pred']))
    print(f'Resuming: {len(done)} already scored\n')

rows = []
for i, (_, r) in enumerate(val_df.iterrows(), 1):
    aid = r['appln_id']
    if aid in done:
        rows.append({'appln_id': aid, 'llm_pred': done[aid]})
        continue

    rows.append({'appln_id': aid, 'llm_pred': classify(r['title'], r['abstract'])})
    time.sleep(SLEEP_BETWEEN_CALLS)

    if i % 25 == 0:
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f'{i}/{len(val_df)} scored')

out = pd.DataFrame(rows)
out.to_csv(OUT_PATH, index=False)

meta = {
    'provider': 'google-genai',
    'model': MODEL,
    'run_date': time.strftime('%Y-%m-%d'),
    'n_scored': int(out['llm_pred'].notna().sum()),
    'n_unparseable': int(out['llm_pred'].isna().sum()),
    'temperature': 0,
    'random_state': RANDOM_STATE,
    'split': 'validation only',
    'prompt': PROMPT,
}
with open('llm_run_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nWritten {OUT_PATH} ({len(out)} rows)')
print(f'Unparseable answers: {meta["n_unparseable"]}')
print('Written llm_run_metadata.json')
