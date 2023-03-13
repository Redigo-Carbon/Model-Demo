from sklearn.datasets import make_regression
import pandas as pd

if __name__ == "__main__":
    X, y, coefficients = make_regression(
        n_samples=10000,
        n_features=5,
        n_informative=5,
        n_targets=1,
        bias=140.0,
        # effective_rank=1
        # tail_strength=0.5,
        noise=0.01,
        shuffle=True,
        coef=True,
        random_state=7,
    )

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X -= X.min()
    X /= X.max()
    y -= y.min()
    y /= y.max()

    df = pd.DataFrame()
    df["num_employees"] = (X[1]*19_999+1).astype(int)
    df['income'] = (X[3]*1_000_000_000+12_500).astype(int).round(-3)
    df['energy_usage'] = (X[2]*100000+5000).astype(int).round(-3)
    df['co2'] = (y*10_000*7+50_000).astype(int).round(-2)

    df.to_csv('data/processed.csv')
