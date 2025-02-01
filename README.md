# covImpute
### Matrix Imputation Using External Covariance Structure

`covImpute` is a matrix imputation method that incorporates an external covariance structure to improve completion accuracy. It employs a Fast Gradient Method with a covariance-regularized objective function, biasing the imputed values towards the given covariance structure.

### Dependencies

- `numpy` 
- `scikit-learn` 

### Usage

```python
import numpy as np
from covImpute import covImpute, cv_mu

# Example matrix with missing values (NaNs)
X = np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]])

# Covariance matrix
E = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])

# Cross-validation to select the best mu
mu_values = [0.01, 0.1, 1, 10]
best_mu, _ = cv_mu(X, E, mu_values)
print(f"Best mu: {best_mu}")

# Perform imputation
_, final_result = covImpute(X, E, mu=0.1)
print("Imputed Matrix:\n", final_result)
```

### License

MIT License

### Author

Hanqing Wu
