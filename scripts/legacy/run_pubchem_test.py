from pathlib import Path
import sys
from pathlib import Path as _P
# Ensure project package `src` is importable when running this script
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from src.drug.crawler import crawl_pubchem_activity_dataset


if __name__ == '__main__':
    df = crawl_pubchem_activity_dataset(
        start_cid=1,
        n=20,
        max_workers=4,
        rate_limit=2.0,
        include_outcome_breakdown=True,
    )
    out = Path('data') / 'pubchem_test_1_20.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.head(20).to_csv(index=False))
    print('SAVED:', str(out))
