import numpy as np
from sklearn.model_selection import GroupKFold


def make_group_splits(df, n_splits=5, group_col="case_id"):
    """
    Create cross-validation splits grouped by case_id.
    This prevents leakage between timesteps of the same case.
    """

    groups = df[group_col].values

    gkf = GroupKFold(n_splits=n_splits)

    splits = []

    dummy_X = np.zeros(len(df))
    dummy_y = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(dummy_X, dummy_y, groups=groups),
        start=1,
    ):
        splits.append(
            {
                "fold": fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_cases": sorted(
                    df.iloc[train_idx][group_col].unique().tolist()
                ),
                "test_cases": sorted(
                    df.iloc[test_idx][group_col].unique().tolist()
                ),
            }
        )

    return splits