import sys
from pathlib import Path
from traceback import print_exc

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.drug.crawler import crawl_pubchem_activity_dataset
except Exception:
    print('IMPORT ERROR')
    print_exc()
    raise

print('START')
try:
    df = crawl_pubchem_activity_dataset(start_cid=1, n=5, max_workers=2, rate_limit=2.0, include_outcome_breakdown=True)
    out = Path('data') / 'pubchem_test_1_5.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_csv(index=False))
    print('SAVED', out)
except Exception:
    print('RUN ERROR')
    print_exc()
    raise
