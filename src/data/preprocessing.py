import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset.

    The unnamed first column is the row counter of the source file each row came
    from; it is kept as `source_row` because it restarts at every file.

    Exact duplicate rows are kept. They all sit in runs with constant throughput:
    a short trajectory logged three times, positions revisited within a run, and
    rows that also occur in another run with the same settings. Dropping them
    would cut steps out of the middle of runs and shorten the run that comes later.
    """
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'source_row'})
    return df

def assign_run_ids(df: pd.DataFrame, run_cols: list) -> pd.DataFrame:
    """Label each row with the simulation run it belongs to.

    The dataset is a concatenation of separate simulation runs. A new run starts
    wherever the radio settings in `run_cols` change, or where `source_row`
    restarts (a new source file, which separates repeated runs with identical
    settings).
    """
    df = df.reset_index(drop=True)
    run_cols = [c for c in run_cols if c in df.columns]
    new_run = pd.Series(False, index=df.index)
    if run_cols:
        new_run |= (df[run_cols] != df[run_cols].shift()).any(axis=1)
    if 'source_row' in df.columns:
        new_run |= df['source_row'].diff() < 0
    new_run.iloc[0] = True
    df['run_id'] = new_run.cumsum() - 1
    return df

def drop_end_markers(df: pd.DataFrame, max_step: float) -> pd.DataFrame:
    """Drop a run's last row when its position is more than `max_step` from the row before it.

    Many runs end with a row at a fixed far-away position with zero throughput
    and delay. It marks the end of the log and is not a time step.
    """
    if not max_step or not {'X', 'Y'} <= set(df.columns):
        return df
    step = np.hypot(df['X'].diff(), df['Y'].diff())
    same_run_as_previous = df['run_id'] == df['run_id'].shift()
    last_of_run = df['run_id'] != df['run_id'].shift(-1)
    return df[~(last_of_run & same_run_as_previous & (step > max_step))].reset_index(drop=True)

def drop_zero_runs(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop runs in which every value of `cols` is zero (failed simulations)."""
    dead = (df[cols] == 0).all(axis=1).groupby(df['run_id']).transform('all')
    return df[~dead].reset_index(drop=True)

def label_splits(df: pd.DataFrame, val_ratio: float, test_ratio: float) -> pd.DataFrame:
    """Label every row train / val / test by its position inside its run.

    The last `test_ratio` of a run's rows are test rows, the `val_ratio` before
    them are validation rows and the rest are training rows.
    """
    position = df.groupby('run_id').cumcount()
    size = df.groupby('run_id')['run_id'].transform('size')
    val_end = size - (size * test_ratio).astype(int)
    train_end = val_end - (size * val_ratio).astype(int)
    df['split'] = np.where(position < train_end, 'train', np.where(position < val_end, 'val', 'test'))
    return df

def impute_missing(df: pd.DataFrame, fit_rows: pd.Series = None) -> pd.DataFrame:
    """Impute missing values using linear regression.

    The regressions are fitted on the rows selected by the boolean mask
    `fit_rows` (the training rows) and applied to every row, so nothing is
    learned from validation or test rows. With no mask all rows are used.
    """
    if fit_rows is None:
        fit_rows = pd.Series(True, index=df.index)

    # Impute jitter from delay, and distance from pdr
    for target, source in (('jitter', 'delay'), ('distance', 'pdr')):
        if target not in df.columns or not df[target].isnull().any():
            continue
        clean = df.loc[fit_rows, [target, source]].dropna()
        model = LinearRegression()
        model.fit(clean[[source]], clean[target])
        missing = df[target].isnull()
        df.loc[missing, target] = model.predict(df.loc[missing, [source]])

    return df

def smooth_features(df: pd.DataFrame, cols: list, filter_len: int) -> pd.DataFrame:
    """Apply rolling mean smoothing to specified columns.

    If the dataframe has a `run_id` column the rolling mean restarts at every
    run, so values from one simulation never bleed into the next.
    """
    for col in cols:
        if col in df.columns:
            series = df.groupby('run_id')[col] if 'run_id' in df.columns else df[col]
            rolled = series.rolling(window=filter_len, min_periods=1).mean()
            df[col] = rolled.reset_index(level=0, drop=True) if 'run_id' in df.columns else rolled

    df = df.bfill().ffill()
    return df

def select_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Subset the dataframe to the specified feature columns."""
    return df[feature_cols]
