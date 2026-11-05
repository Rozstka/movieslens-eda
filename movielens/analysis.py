from pathlib import Path
import numpy as np
import pandas as pd

def load_data(directory):
    directory = Path(directory)
    if not (directory / 'u.data').exists():
        raise FileNotFoundError('Run python scripts/download_data.py from the repository root first')
    ratings = pd.read_csv(directory / 'u.data', sep='\t', names=['userId','movieId','rating','timestamp'])
    movies = pd.read_csv(directory / 'u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movieId','title'])
    if not ratings.rating.between(1,5).all() or ratings.duplicated(['userId','movieId']).any():
        raise ValueError('Invalid ratings or duplicate user/movie pairs')
    return ratings, movies

def similar_movies(ratings, movies, movie_id, min_common=20, limit=10):
    if min_common < 2 or limit < 1:
        raise ValueError('Require at least two shared viewers and positive limit')
    matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    if movie_id not in matrix:
        raise ValueError('Unknown movie ID')
    target = matrix[movie_id]
    common = matrix.notna().mul(target.notna(), axis=0).sum()
    # Pairwise correlation requires enough shared users, not total film popularity.
    eligible = common[common >= min_common].index.difference([movie_id])
    rows = []
    for candidate in eligible:
        both = pd.concat([target, matrix[candidate]], axis=1).dropna()
        if both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
            continue
        corr = both.iloc[:, 0].corr(both.iloc[:, 1])
        if np.isfinite(corr):
            rows.append({'movieId': int(candidate), 'correlation': float(corr), 'common_viewers': int(common[candidate])})
    result = pd.DataFrame(rows, columns=['movieId','correlation','common_viewers'])
    return result.merge(movies[['movieId','title']], on='movieId').sort_values(
        ['correlation','common_viewers','movieId'], ascending=[False,False,True]).head(limit)

def chronological_split(ratings, fraction=0.8):
    if not 0 < fraction < 1:
        raise ValueError('fraction must be between 0 and 1')
    ordered = ratings.sort_values(['timestamp','userId','movieId'], kind='stable')
    cut = int(len(ordered) * fraction)
    if cut == 0 or cut == len(ordered):
        raise ValueError('Not enough ratings')
    # Keep equal timestamps together to avoid ambiguous time leakage.
    threshold = ordered.iloc[cut]['timestamp']
    train, test = ordered[ordered.timestamp < threshold], ordered[ordered.timestamp >= threshold]
    if train.empty or test.empty:
        raise ValueError('Not enough distinct timestamps')
    return train, test

def evaluate(ratings):
    train, test = chronological_split(ratings)
    global_mean = float(train.rating.mean())
    stats = train.groupby('movieId').rating.agg(['sum','count'])
    regularized = (stats['sum'] + 10 * global_mean) / (stats['count'] + 10)
    pred = test.movieId.map(regularized).fillna(global_mean)
    return {'train_rows': len(train), 'test_rows': len(test),
        'train_max_timestamp': int(train.timestamp.max()), 'test_min_timestamp': int(test.timestamp.min()),
        'global_mean_rmse': float(np.sqrt(((test.rating - global_mean) ** 2).mean())),
        'regularized_item_mean_rmse': float(np.sqrt(((test.rating - pred) ** 2).mean())),
        'cold_item_fraction': float((~test.movieId.isin(train.movieId)).mean())}
