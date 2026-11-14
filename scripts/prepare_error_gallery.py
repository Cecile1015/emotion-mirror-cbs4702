"""Build a balanced teaching gallery from saved predictions; never relabel test data."""
from pathlib import Path
import hashlib,json,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.config import LABELS
from src.keywords import KeywordLexicon
from src.error_analysis import code_error

def main():
    annotations=json.loads((ROOT/'data/handcrafted/website_error_examples.json').read_text())
    data=json.loads((ROOT/'results/app/error_analysis.json').read_text())
    train=pd.read_parquet(ROOT/'data/train.parquet')
    lex=KeywordLexicon().fit(train.text,[LABELS[int(v)]for v in train.label])
    names={'bert':'BERT','bilstm':'BiLSTM','tfidf_lr':'TF-IDF+LR','improved_lr':'Improved LR','bow_nb':'BoW+NB'}
    saved={m:pd.read_csv(ROOT/f'results/predictions/{m}__test.csv').set_index('idx')for m in names}
    categories={c['code']:c['category']for c in data['categories']};examples=[]
    for annotation in annotations['examples']:
        idx=annotation['idx'];focus=annotation['focus_model'];row=saved[focus].loc[idx]
        assert not row['correct'],f'Not a saved error: {idx}/{focus}'
        code=code_error(row.text,row.true_label,row.pred_label,lex)['code']
        assert code==annotation['category_code'],f'Category mismatch: {idx}'
        predictions={names[m]:f"{frame.loc[idx].pred_label} ({frame.loc[idx].confidence:.2f})"for m,frame in saved.items()}
        examples.append({**annotation,'text':row.text,'gold':row.true_label,**predictions,'focus_model_name':names[focus],
                         'focus_prediction':row.pred_label,'auto category':code+' '+categories[code],
                         'provenance':f'results/predictions/{focus}__test.csv; test idx {idx}'})
    counts=Counter(e['category_code']for e in examples)
    assert counts=={c:3 for c in categories} and len({e['idx']for e in examples})==21
    data.update(examples=examples,gallery_counts=dict(counts),gallery_note=annotations['note'])
    (ROOT/'web/data/error_analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    files=['data/handcrafted/website_error_examples.json']+[f'results/predictions/{m}__test.csv'for m in names]
    report={'counts':dict(counts),'unique_cases':21,'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in files},'selection_note':annotations['note']}
    (ROOT/'docs/reproducibility/error_gallery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['counts'])
if __name__=='__main__':main()
