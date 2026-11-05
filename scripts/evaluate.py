from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from movielens.analysis import load_data, evaluate
ratings, _ = load_data(ROOT / 'data')
result = evaluate(ratings)
(ROOT / 'reports').mkdir(exist_ok=True)
(ROOT / 'reports/metrics.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
