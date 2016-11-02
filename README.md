This small library provides a `solve_triangular` function that is like
`scipy.linalg.solve_triangular` except that it broadcasts along leading
dimensions like `np.linalg.solve`. (Also it doesn't work with arrays of complex
numbers.)

This library also includes an autograd gradient definition of
`solve_triangular`, and an alternative gradient definition for `cholesky` that
uses this `solve_triangular` for a more efficient broadcasted implementation.
