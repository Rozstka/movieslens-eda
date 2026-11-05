import pandas as pd
import pytest
from movielens.analysis import similar_movies, chronological_split, evaluate

def test_counts_shared_viewers_not_total_popularity():
    rows=[(1,1,1),(2,1,3),(3,1,5),(1,2,1),(2,2,3),(4,2,5),(5,2,2),(1,3,1),(2,3,3),(3,3,5)]
    ratings=pd.DataFrame(rows,columns=['userId','movieId','rating'])
    movies=pd.DataFrame({'movieId':[1,2,3],'title':['Target','Popular but sparse','Supported']})
    result=similar_movies(ratings,movies,1,min_common=3)
    assert result.movieId.tolist()==[3]
    assert result.common_viewers.tolist()==[3]

def test_empty_result_and_unknown_movie():
    r=pd.DataFrame({'userId':[1,2],'movieId':[1,1],'rating':[3,4]})
    m=pd.DataFrame({'movieId':[1],'title':['Target']})
    assert similar_movies(r,m,1,2).empty
    with pytest.raises(ValueError): similar_movies(r,m,99)

def test_temporal_holdout_and_cold_start():
    r=pd.DataFrame({'userId':range(10),'movieId':[1]*8+[2]*2,'rating':[3]*8+[4,4], 'timestamp':range(10)})
    train,test=chronological_split(r)
    assert train.timestamp.max() < test.timestamp.min()
    metrics=evaluate(r)
    assert metrics['cold_item_fraction']==1
    assert metrics['regularized_item_mean_rmse']==1

def test_timestamp_ties_stay_together():
    r=pd.DataFrame({'userId':range(10),'movieId':range(10),'rating':[3]*10,'timestamp':[1]*5+[2]*5})
    train,test=chronological_split(r)
    assert len(train)==5 and len(test)==5
