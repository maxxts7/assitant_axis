# Principal Component Analysis: A Deep Dive

## Opening

Karl Pearson (1857–1936) was a British mathematician and biostatistician at University College London. In the late 1890s he was measuring organisms — skull dimensions, bone lengths, body proportions — trying to understand biological variation. Each specimen produced many numbers, and most of them were correlated. A longer femur usually came with a longer tibia. A wider skull usually came with a heavier jaw. The measurements were partly redundant: they overlapped in what they told you.

Pearson asked: is there a systematic way to find the "true" underlying dimensions — fewer than the raw measurements — that capture most of the variation between specimens? In 1901, he published "On Lines and Planes of Closest Fit to Systems of Points in Space," where he framed this as a geometric problem: find the line (or plane) through a cloud of data points that minimizes the sum of squared perpendicular distances to the points. He did not use the term "principal component."

Harold Hotelling (1895–1973), an American mathematical statistician, took this further. Working with psychological test scores — IQ subtests, personality inventories — he formalized the method in 1933 under the name we use today. His question was Pearson's question, stated with algebraic precision:

> Given a set of correlated measurements, how do we find new, uncorrelated combinations of those measurements that capture the maximum possible amount of variation in the data?

This is the question PCA answers. The rest of this document shows exactly how — from first principles, at full granularity, with nothing skipped.

---

## Key Terminology

**Variable**: A single quantity measured on each observation. If you measure the height and weight of 100 people, "height" is one variable and "weight" is another. Synonyms: feature, dimension.

**Observation**: A single entity on which variables are measured. One person in the height-weight study. Synonyms: data point, sample.

**Mean**: The average value of a variable across all observations. Sum all values, divide by the count. Written μ (Greek letter mu). Five heights of 160, 165, 170, 175, 180 cm give a mean of 170.

**Variance**: A number measuring how spread out a variable's values are around its mean. For each observation, take the distance from the mean, square it, and average those squared distances. Written Var(x) or σ². Larger variance means more spread. A variable where every observation is identical has variance zero.

**Standard deviation**: The square root of variance, written σ. Same units as the original variable, making it more interpretable than variance.

**Covariance**: A number measuring how two variables move together. When high values of x tend to appear alongside high values of y (and low with low), covariance is positive. When they move oppositely, it is negative. When there is no linear pattern, it is near zero. Written Cov(x, y). The covariance of a variable with itself equals its variance.

**Covariance matrix**: A square table (matrix) containing the covariance between every pair of variables. For p variables, it is p × p. Diagonal entries are variances; off-diagonal entries are covariances. It is always symmetric: Cov(x, y) = Cov(y, x).

**Matrix**: A rectangular grid of numbers arranged in rows and columns. A 3 × 2 matrix has 3 rows and 2 columns.

**Vector**: An ordered list of numbers. A vector with p entries can represent a point in p-dimensional space or a direction (an arrow) in that space.

**Unit vector**: A vector whose length is exactly 1. The length of vector w = (w₁, w₂, ..., wₚ) is √(w₁² + w₂² + ... + wₚ²). A unit vector specifies a pure direction, with no magnitude.

**Dot product**: A way to combine two vectors of the same length into a single number. For u = (u₁, u₂, ..., uₚ) and v = (v₁, v₂, ..., vₚ), the dot product is u₁v₁ + u₂v₂ + ... + uₚvₚ. Written u · v. Geometrically: u · v = |u| × |v| × cos(θ), where θ is the angle between them. If they are perpendicular, the dot product is zero.

**Projection**: The shadow of a data point onto a line. If you shine a flashlight perpendicular to a line, each point casts a shadow on that line — that shadow is the projection. Mathematically, the projection of point x onto unit vector w is the number x · w (their dot product).

**Eigenvector**: A special direction associated with a matrix. When the matrix acts on this vector (as a transformation), the result points in the same direction — only the length changes. The factor by which the length changes is the eigenvalue. Precisely defined in Section 9.

**Eigenvalue**: The scaling factor paired with an eigenvector. If matrix A applied to vector v yields λv (same direction, scaled by λ), then λ is the eigenvalue. Always non-negative for covariance matrices. Precisely defined in Section 9.

**Principal component (PC)**: A new variable created by PCA. It is a specific weighted combination of the original variables, chosen to capture maximum variance. The first PC captures the most variance possible in any single linear combination; the second captures the most of what remains while being uncorrelated with the first; and so on.

**Loading**: The weight an original variable receives in a principal component. If PC1 = 0.7 × height + 0.7 × weight, then 0.7 is the loading of both height and weight on PC1. The loadings are the entries of the eigenvector that defines that component.

---

## 1. The Problem: Why We Need PCA

Suppose you measure five traits on each of 200 animals: body length, tail length, leg length, skull width, and body mass. You now have a 200 × 5 table — 200 observations, 5 variables.

These variables are correlated. Bigger animals tend to be longer, heavier, and have wider skulls. Much of the information in five numbers is redundant — they are partly telling you the same thing: "how big is this animal?"

Three problems arise:

**Redundancy.** Five numbers per animal, but perhaps only two independent patterns exist (e.g., "overall size" and "body proportions"). The raw variables obscure this.

**Visualization.** You cannot plot five dimensions. You can plot two. Any pair of original variables you choose discards information from the other three. There is no obvious "best pair."

**Noise.** With many correlated variables, small measurement errors create misleading patterns. Combining variables into fewer, stronger signals can improve signal-to-noise.

PCA addresses all three. It finds new variables — linear combinations of the originals — that are uncorrelated and ordered by how much variation they capture. The first few PCs often summarize most of the data, letting you reduce five variables to two or three without arbitrarily picking favorites among the originals.

---

## 2. Variance: Why Spread Equals Information

Before finding "important directions," we need a definition of "important."

PCA uses **variance** as its measure of importance. Why? Because a variable with high variance differs a lot across observations — it distinguishes them. A variable with zero variance is the same for everyone — it carries no information about differences.

Three parallel examples, holding everything constant except the spread:

- Variable A: all 4 observations are 5. Values: 5, 5, 5, 5. Variance = 0. You learn nothing about which observation is which.
- Variable B: values 4, 5, 5, 6. Variance = 0.5. Some distinguishing power — you can tell the first and last observations apart.
- Variable C: values 1, 4, 6, 9. Variance = 8.5. Strong distinguishing power — every observation is clearly different.

**Definition.** Given n observations of a variable x with values x₁, x₂, ..., xₙ and mean μ = (x₁ + x₂ + ... + xₙ) / n, the variance is:

    Var(x) = (1/n) × [(x₁ − μ)² + (x₂ − μ)² + ... + (xₙ − μ)²]

Each term (xᵢ − μ)² measures how far observation i deviates from the average. Squaring serves two purposes: it prevents positive and negative deviations from canceling each other, and it penalizes large deviations more than small ones. The average of these squared deviations is the variance.

**Worked example.** Four heights: 160, 170, 180, 190 cm. Mean μ = 175.
Deviations from mean: −15, −5, 5, 15.
Squared deviations: 225, 25, 25, 225.
Variance = (225 + 25 + 25 + 225) / 4 = 125.

If all four were 175 cm, every deviation is zero, variance is zero — no information about differences.

**Subtlety: variance as importance is a choice, not a law.** PCA declares "important = high variance." This works when variance reflects genuine signal. It can mislead when one variable has high variance purely because of its measurement units (e.g., millimeters vs. kilometers). This is addressed in Section 16.

**Subtlety: 1/n vs. 1/(n−1).** Some formulations divide by (n−1) instead of n. Dividing by (n−1) gives an unbiased estimate of the population variance when working with a sample. Dividing by n gives the exact variance of the data you have. For PCA's core logic, the choice makes no difference: it scales all eigenvalues by the same factor, leaving the eigenvectors (and thus the principal component directions) unchanged. This document uses 1/n for simplicity.

---

## 3. Covariance: How Variables Move Together

Variance measures the spread of one variable. **Covariance** measures how two variables relate to each other.

**Motive.** If we only cared about individual variables, we could just rank them by variance and keep the highest-variance ones. But PCA's power comes from recognizing that variables share information through correlation. We need a quantity that captures this sharing.

**Definition.** Given n observations of variables x and y, with means μₓ and μᵧ:

    Cov(x, y) = (1/n) × [(x₁ − μₓ)(y₁ − μᵧ) + (x₂ − μₓ)(y₂ − μᵧ) + ... + (xₙ − μₓ)(yₙ − μᵧ)]

Each term (xᵢ − μₓ)(yᵢ − μᵧ) is the product of the deviations of x and y from their respective means for observation i. This product is:

- **Positive** when both x and y deviate in the same direction (both above mean, or both below).
- **Negative** when they deviate in opposite directions (one above, one below).
- **Zero** when at least one of them is exactly at its mean.

Averaging these products gives the covariance.

Three parallel examples of what different covariance values look like:

- Cov > 0: Height and weight. Taller people tend to weigh more. Both variables deviate in the same direction for most observations, making most product terms positive.
- Cov < 0: Hours of exercise per week and resting heart rate. More exercise tends to lower heart rate. The deviations go in opposite directions, making most products negative.
- Cov ≈ 0: Height and number of siblings. No systematic relationship. Positive and negative product terms roughly cancel.

**Key identity:** Cov(x, x) = Var(x). The covariance of a variable with itself is its variance. (Substituting y = x in the formula gives the variance formula exactly.)

**Example.** Four students' centered scores (mean already subtracted):
Math deviations: 3, −3, 1, −1. Science deviations: 1, −1, 3, −3.

    Cov(math, science) = (3×1 + (−3)×(−1) + 1×3 + (−1)×(−3)) / 4
                       = (3 + 3 + 3 + 3) / 4
                       = 3

Positive covariance: students above average on math tend to be above average on science.

---

## 4. The Covariance Matrix: All Relationships at Once

With p variables, there are p variances and p(p−1)/2 unique covariances. The **covariance matrix** C packs all of these into a single p × p table.

**Motive.** PCA must reason about all variables simultaneously — not one pair at a time. The covariance matrix is the compact mathematical object that encodes the complete picture of how every variable relates to every other.

**Structure for p = 2 variables (x and y):**

    C = | Var(x)    Cov(x,y) |
        | Cov(y,x)  Var(y)   |

The diagonal holds variances. The off-diagonal holds covariances. Since Cov(x, y) = Cov(y, x), the matrix is **symmetric**: the entry in row i, column j equals the entry in row j, column i.

**From the example in Section 3:**

    C = | 5  3 |
        | 3  5 |

Var(math) = 5, Var(science) = 5, Cov(math, science) = 3.

**For p = 3 variables** (x, y, z), the matrix is 3 × 3:

    C = | Var(x)    Cov(x,y)  Cov(x,z) |
        | Cov(y,x)  Var(y)    Cov(y,z) |
        | Cov(z,x)  Cov(z,y)  Var(z)   |

The pattern extends to any number of variables.

**A critical property: positive semi-definiteness.** The covariance matrix is always **positive semi-definite**: for any vector w, the quantity w^T C w ≥ 0.

Here w^T C w means: multiply the matrix C by the vector w, then take the dot product of the result with w. This quantity will appear repeatedly — it turns out to equal the variance of data projected onto the direction w.

**Why is it positive semi-definite?** If X is the n × p matrix of centered data (rows are observations, columns are variables), then C = (1/n) × X^T X. For any vector w:

    w^T C w = (1/n) × w^T X^T X w = (1/n) × (Xw)^T(Xw) = (1/n) × ||Xw||²

The quantity ||Xw||² is a sum of squared numbers — it cannot be negative. This guarantees that all eigenvalues of C are non-negative, which will matter when we interpret eigenvalues as variances.

---

## 5. The Geometric Picture

This section gives the visual intuition that makes PCA click.

**Two variables.** Plot each observation as a point in a 2D plane, with variable x on the horizontal axis and variable y on the vertical axis. The observations form a **cloud of points**.

If the variables are correlated (positive covariance), the cloud is elongated along a diagonal — it looks like a tilted ellipse. The stronger the correlation, the more elongated. If the variables are uncorrelated, the cloud is roughly circular.

Our four students, in centered coordinates: (3, 1), (−3, −1), (1, 3), (−1, −3). Two points cluster along the upper-right/lower-left diagonal (the "both high" and "both low" students). Two cluster along the other diagonal (one subject high, the other low). The overall shape is an ellipse tilted at 45°.

**The axes of the ellipse.** Every ellipse has a long axis and a short axis, perpendicular to each other. The long axis is the direction of greatest spread. The short axis is the direction of least spread.

PCA finds these axes. That is the entire geometric content of PCA.

**Higher dimensions.** With 3 variables, the data cloud lives in 3D and forms an ellipsoid (a 3D ellipse — like a squashed football). It has three axes: long, medium, short. With p variables, the cloud lives in p-dimensional space and forms a p-dimensional ellipsoid with p perpendicular axes. You cannot visualize it, but the algebra is identical to the 2D case.

**The key geometric insight:** PCA rotates the coordinate system so the new axes align with the natural axes of the data's elliptical shape. The first axis points along the longest direction. The second axis points along the next longest, perpendicular to the first. This rotation preserves all the data — nothing is lost. Dimensionality reduction happens only when you then decide to ignore the shorter axes.

---

## 6. Projection: Collapsing Data onto a Line

To find the "best direction," we need to formalize what it means to view data from a single direction. This is **projection**.

**Motive.** We want to reduce p dimensions to fewer dimensions. The simplest case: reduce to 1 dimension. This means collapsing every data point onto a single line. Different lines will give different 1D summaries. We need the math to compare them.

**Setup.** A data point x = (x₁, x₂) in 2D and a direction defined by a unit vector w = (w₁, w₂), where w₁² + w₂² = 1.

**The projection** of x onto w is the scalar (single number):

    z = x · w = x₁w₁ + x₂w₂

This number z tells you how far x lies along the direction w. It is the "shadow" of x on the line defined by w.

**Example.** Point x = (3, 1). Direction w = (1/√2, 1/√2) — the 45° diagonal.

    z = 3 × (1/√2) + 1 × (1/√2) = 4/√2 ≈ 2.83

The point (3, 1) is approximately 2.83 units along the diagonal direction.

**Why unit vector?** If w were twice as long, z would be twice as large — the projection would be artificially inflated by the length of w rather than reflecting the true position of x. Requiring ||w|| = 1 ensures z measures only the position of x along the direction, not the arbitrary scale of w.

**From 2D to 1D.** Projecting all n observations onto direction w collapses each 2D point to a single number. You now have n numbers instead of n pairs. The question becomes: which direction w loses the least information?

**The geometric reconstruction.** The projected point in the original 2D space is (x · w) × w — the shadow's position on the line. The difference between the original point x and its projection is the **reconstruction error**: the information lost by collapsing to 1D. The reconstruction error vector is always perpendicular to w (it is the component of x that is "invisible" from direction w).

---

## 7. The Core Optimization: Maximize Projected Variance

PCA's answer to "which direction is best?" is: the one that makes the projected data have the **largest possible variance**.

**Motive.** Consider two directions to project your 2D data onto:
- Direction along the long axis of the ellipse: the projected points are spread far apart. High variance. You can distinguish observations.
- Direction along the short axis: the projected points are bunched together. Low variance. Distinct observations look the same.

Maximizing projected variance preserves the most distinction between observations. This is what "best" means for PCA.

**The formula.** Let x₁, x₂, ..., xₙ be centered data points (mean subtracted). The projection of xᵢ onto unit vector w is zᵢ = xᵢ · w. The variance of the projections is:

    Var(z) = (1/n) × (z₁² + z₂² + ... + zₙ²)

The z values have mean zero (because the data was centered), so the variance simplifies to the average of the squared projections.

**Connecting to the covariance matrix.** This variance can be rewritten in a compact form using the covariance matrix C:

    Var(z) = w^T C w

**Derivation.** Starting from Var(z) = (1/n) Σᵢ zᵢ², and substituting zᵢ = xᵢ · w = w^T xᵢ:

    Var(z) = (1/n) Σᵢ (w^T xᵢ)(xᵢ^T w)
           = w^T [(1/n) Σᵢ xᵢ xᵢ^T] w
           = w^T C w

The bracketed expression (1/n) Σᵢ xᵢ xᵢ^T is exactly the covariance matrix C for centered data.

**The optimization problem:**

> Find the unit vector w that maximizes w^T C w, subject to the constraint w^T w = 1.

This is PCA's foundational problem. Everything that follows is about solving it.

**Why the constraint matters.** Without w^T w = 1, you could scale w to infinity and make w^T C w arbitrarily large. The constraint forces a choice of direction — it decouples "which way to look" from "how far to stretch."

---

## 8. Constrained Optimization with Lagrange Multipliers

We must maximize a function (w^T C w) subject to a constraint (w^T w = 1). The standard tool for this is **Lagrange multipliers**, developed by Joseph-Louis Lagrange in the 1780s.

**Motive.** Ordinary calculus says: at a maximum, the derivative is zero. But we're not maximizing over all possible w — only over w on the unit sphere (the surface where w^T w = 1). We need a method that respects this constraint.

**The intuition.** Imagine walking along the constraint surface — in our case, the unit circle in 2D (the set of all vectors with length 1). At each point on this circle, the objective w^T C w has some value. As you walk, this value changes. At the constrained maximum, it stops changing: no small step along the circle increases it.

When can no step along the circle increase the objective? When the gradient of the objective (the direction of steepest increase) points **perpendicular** to the circle. If the gradient had any component along the circle, you could walk in that direction and increase the objective — contradicting the assumption that you're at the maximum.

The gradient of the constraint function g(w) = w^T w − 1 always points perpendicular to the constraint surface (outward from the circle). So the condition for a constrained maximum is that the gradient of the objective is parallel to the gradient of the constraint:

    ∇(w^T C w) = λ × ∇(w^T w − 1)

where λ is a scalar called the **Lagrange multiplier**.

**Computing the gradients.** For those unfamiliar with matrix calculus, here is what each gradient means:

- ∇(w^T C w) is the vector of partial derivatives of w^T C w with respect to each entry of w. For a symmetric matrix C, this equals 2Cw. (Each entry of this vector tells you how fast the objective changes when you nudge the corresponding entry of w.)

- ∇(w^T w) is the vector of partial derivatives of w^T w with respect to each entry of w. This equals 2w.

**Setting them proportional:**

    2Cw = λ × 2w

Dividing both sides by 2:

    Cw = λw

This is the **eigenvalue equation**. The direction w that maximizes projected variance must satisfy this equation — it must be an eigenvector of the covariance matrix C with eigenvalue λ.

---

## 9. Eigenvectors and Eigenvalues: What They Are

The equation Cw = λw from the previous section is one of the most important equations in linear algebra. Let us understand it fully.

**Motive.** We need to solve Cw = λw to find the optimal directions for PCA. But first we must understand what this equation means and what its solutions look like.

**Definition.** Given a square matrix A (same number of rows as columns), a nonzero vector v is an **eigenvector** of A if there exists a scalar λ such that:

    Av = λv

The scalar λ is the **eigenvalue** associated with eigenvector v. The pair (λ, v) is called an eigenpair. The word "eigen" is German for "own" or "characteristic."

**What this equation says, symbol by symbol:**
- A is a square matrix (for PCA, it is the covariance matrix C).
- v is a nonzero vector with p entries (for PCA, it is a candidate direction in p-dimensional space).
- Av means "apply the matrix A as a linear transformation to the vector v" — this is standard matrix-vector multiplication, producing a new vector with p entries.
- λv means "scale the vector v by the number λ" — multiply every entry by λ.
- The equation says: applying A to v produces the exact same direction as v, only scaled by λ.

**Geometric meaning.** Normally, multiplying a matrix by a vector changes both its direction and its length. An eigenvector is special: the matrix changes only its length (by factor λ), not its direction.

Three cases:
- λ > 1: the eigenvector is stretched.
- 0 < λ < 1: the eigenvector is compressed.
- λ = 0: the eigenvector is collapsed to the zero vector (the transformation destroys information in this direction entirely).

For covariance matrices, λ < 0 never occurs (proven below).

**How to find eigenvalues.** Rearrange Av = λv to (A − λI)v = 0, where I is the identity matrix (1s on the diagonal, 0s elsewhere). This equation has a nonzero solution v only when the matrix (A − λI) is "singular" — meaning its determinant is zero:

    det(A − λI) = 0

This is called the **characteristic equation**. It is a polynomial of degree p in λ, so it has (counting multiplicity) exactly p roots. Each root is an eigenvalue. For each eigenvalue, you solve (A − λI)v = 0 to find the corresponding eigenvector.

**For a symmetric p × p matrix (like a covariance matrix), three guarantees hold:**

1. **All p eigenvalues are real numbers** (not complex). This follows from the symmetry.
2. **Eigenvectors corresponding to distinct eigenvalues are orthogonal** (perpendicular). Proved below.
3. **There are always p mutually orthogonal eigenvectors** (even if some eigenvalues are repeated). This is the **spectral theorem**.

These guarantees are why PCA works so cleanly. They are not obvious and require proof.

**Proof: eigenvectors of a symmetric matrix with distinct eigenvalues are orthogonal.**

Suppose Cv₁ = λ₁v₁ and Cv₂ = λ₂v₂ with λ₁ ≠ λ₂. Consider the quantity v₁^T C v₂. Compute it two ways:

Way 1: v₁^T(Cv₂) = v₁^T(λ₂v₂) = λ₂(v₁^T v₂) = λ₂(v₁ · v₂)

Way 2: Because C is symmetric (C^T = C), we have v₁^T C = (Cv₁)^T = (λ₁v₁)^T = λ₁v₁^T. So v₁^T C v₂ = λ₁(v₁^T v₂) = λ₁(v₁ · v₂).

Setting equal: λ₁(v₁ · v₂) = λ₂(v₁ · v₂).

Since λ₁ ≠ λ₂, the only way this holds is if v₁ · v₂ = 0. A dot product of zero means perpendicular. The orthogonality of eigenvectors is forced by the symmetry of C — it is not a design choice.

**Proof: eigenvalues of a covariance matrix are non-negative.**

For eigenvector v with eigenvalue λ: v^T C v = v^T(λv) = λ(v^T v) = λ||v||². But Section 4 showed w^T C w ≥ 0 for any vector w. So λ||v||² ≥ 0. Since v is nonzero, ||v||² > 0, giving λ ≥ 0.

This makes physical sense: eigenvalues will represent variances, and variance cannot be negative.

---

## 10. Why Eigenvectors Solve PCA

Now we connect everything from the preceding sections.

**The punchline.** The direction w that maximizes projected variance w^T C w (subject to ||w|| = 1) must satisfy Cw = λw — it must be an eigenvector of C. And the projected variance along that direction equals the eigenvalue:

    w^T C w = w^T(λw) = λ(w^T w) = λ × 1 = λ

Step by step:
- w^T C w: the projected variance we want to maximize.
- Substitute Cw = λw: w^T(λw).
- Factor out the scalar λ: λ(w^T w).
- Use the unit-length constraint w^T w = 1: λ × 1 = λ.

The variance captured by direction w **equals** its eigenvalue λ. To maximize this, choose the eigenvector with the **largest** eigenvalue.

> **The first principal component is the eigenvector of the covariance matrix with the largest eigenvalue. Its eigenvalue is the variance of the data projected onto that direction.**

This is the central result of PCA. Everything else — second components, reconstruction, interpretation — follows from it.

---

## 11. The Second Component and Beyond

After the first direction, we want the next best direction. A new constraint enters: the second direction must be **orthogonal** (perpendicular) to the first.

**Motive.** Orthogonality ensures the second component captures new information, not information already captured by the first. If the second direction had any component along the first, it would partly duplicate what the first component already measures. Requiring perpendicularity forces it to capture genuinely different variation. Formally, two perpendicular projections have zero covariance — they are uncorrelated.

**The problem:** Find unit vector w₂ that maximizes w₂^T C w₂, subject to:
1. w₂^T w₂ = 1 (unit length)
2. w₂^T w₁ = 0 (orthogonal to the first component)

**The answer:** The eigenvector with the **second-largest eigenvalue**.

**Proof sketch.** Apply Lagrange multipliers with two constraints. The stationarity condition gives Cw₂ = λ₂w₂ + μw₁ for some scalars λ₂ and μ. Take the dot product of both sides with w₁:

    w₁^T C w₂ = λ₂(w₁^T w₂) + μ(w₁^T w₁)

The left side: w₁^T C w₂ = (Cw₁)^T w₂ = λ₁(w₁^T w₂) = 0 (using the orthogonality constraint).
The right side: λ₂ × 0 + μ × 1 = μ.
So μ = 0, giving Cw₂ = λ₂w₂ — the eigenvalue equation again. The maximum is the largest remaining eigenvalue.

**The pattern continues.** The k-th principal component is the eigenvector with the k-th largest eigenvalue. Each one is perpendicular to all previous components. For a p × p covariance matrix, there are exactly p principal components, forming a complete new coordinate system.

**Total variance is conserved.** The sum of all eigenvalues equals the sum of all original variances:

    λ₁ + λ₂ + ... + λₚ = Var(x₁) + Var(x₂) + ... + Var(xₚ)

This follows from a property of matrices called the **trace**: the sum of the diagonal entries of C equals the sum of the eigenvalues of C. The diagonal entries of C are the variances of the original variables.

PCA does not create or destroy variance. It redistributes it — concentrating it into the first few components. The total is unchanged. It is a rotation of the coordinate system, not a distortion.

---

## 12. The Full PCA Procedure

The complete algorithm, step by step:

**Step 1: Center the data.** For each variable, subtract its mean from every observation. Each variable now has mean zero. This is essential — PCA finds directions of maximum variance about the origin, and without centering, the position of the mean would distort the directions.

**Step 2: Compute the covariance matrix C.** Using the centered data, compute the p × p matrix where entry (i, j) is Cov(variable i, variable j).

**Step 3: Find the eigenvectors and eigenvalues of C.** This yields p eigenvectors (the principal component directions) and p eigenvalues (the variance along each direction).

**Step 4: Sort by eigenvalue.** Rank the eigenvectors from largest to smallest eigenvalue. The eigenvector with the largest eigenvalue defines the first principal component direction. The one with the second largest defines the second. And so on.

**Step 5: Project.** To reduce from p dimensions to k dimensions (where k < p), project each observation onto the first k eigenvectors. Each observation collapses from a p-dimensional vector to a k-dimensional vector of **principal component scores**.

**What is lost in Step 5.** The variance along the discarded components. The fraction of variance retained is:

    (λ₁ + λ₂ + ... + λₖ) / (λ₁ + λ₂ + ... + λₚ)

If the first 2 eigenvalues account for 95% of the total, keeping 2 components loses only 5% of the variance. The lost 5% is the variation in the directions that were discarded — the short axes of the ellipsoid.

---

## 13. Worked Example

**Setup.** Four students take two exams: Math and Science.

| Student | Math | Science |
|---------|------|---------|
| A       | 73   | 66      |
| B       | 67   | 64      |
| C       | 71   | 68      |
| D       | 69   | 62      |

**Step 1: Center.** Mean Math = (73 + 67 + 71 + 69) / 4 = 70. Mean Science = (66 + 64 + 68 + 62) / 4 = 65. Subtract the means:

| Student | Math (centered) | Science (centered) |
|---------|:---------------:|:------------------:|
| A       | 3               | 1                  |
| B       | −3              | −1                 |
| C       | 1               | 3                  |
| D       | −1              | −3                 |

**Step 2: Covariance matrix.**

Var(math) = (3² + (−3)² + 1² + (−1)²) / 4 = (9 + 9 + 1 + 1) / 4 = **5**

Var(science) = (1² + (−1)² + 3² + (−3)²) / 4 = (1 + 1 + 9 + 9) / 4 = **5**

Cov(math, science) = (3×1 + (−3)×(−1) + 1×3 + (−1)×(−3)) / 4 = (3 + 3 + 3 + 3) / 4 = **3**

    C = | 5  3 |
        | 3  5 |

**Step 3a: Eigenvalues.** Solve det(C − λI) = 0:

    det | 5−λ  3   | = 0
        | 3    5−λ |

    (5 − λ)(5 − λ) − 3 × 3 = 0
    λ² − 10λ + 25 − 9 = 0
    λ² − 10λ + 16 = 0
    (λ − 8)(λ − 2) = 0

    λ₁ = 8,  λ₂ = 2

**Step 3b: Eigenvectors.**

For λ₁ = 8, solve (C − 8I)v = 0:

    (5 − 8)v₁ + 3v₂ = 0
    −3v₁ + 3v₂ = 0
    v₁ = v₂

Any vector where both entries are equal works. Normalized to unit length: **v₁ = (1/√2, 1/√2) ≈ (0.707, 0.707)**.

For λ₂ = 2, solve (C − 2I)v = 0:

    (5 − 2)v₁ + 3v₂ = 0
    3v₁ + 3v₂ = 0
    v₁ = −v₂

Normalized: **v₂ = (1/√2, −1/√2) ≈ (0.707, −0.707)**.

**Verify orthogonality:** v₁ · v₂ = (1/√2)(1/√2) + (1/√2)(−1/√2) = 1/2 − 1/2 = 0. ✓

**Step 4: Sort.** Already sorted: λ₁ = 8 > λ₂ = 2.

**Step 5: Project.** Compute each student's principal component scores.

PC1 score = (math centered) × (1/√2) + (science centered) × (1/√2)
PC2 score = (math centered) × (1/√2) + (science centered) × (−1/√2)

| Student | Math | Sci  | PC1 = (m + s)/√2 | PC2 = (m − s)/√2 |
|---------|:----:|:----:|:-----------------:|:-----------------:|
| A       | 3    | 1    | 4/√2 ≈ 2.83      | 2/√2 ≈ 1.41      |
| B       | −3   | −1   | −4/√2 ≈ −2.83    | −2/√2 ≈ −1.41    |
| C       | 1    | 3    | 4/√2 ≈ 2.83      | −2/√2 ≈ −1.41    |
| D       | −1   | −3   | −4/√2 ≈ −2.83    | 2/√2 ≈ 1.41      |

**Verify eigenvalues match projected variances:**

PC1 scores: 2.83, −2.83, 2.83, −2.83. Squared: 8, 8, 8, 8. Variance = 32/4 = **8**. ✓ Matches λ₁.

PC2 scores: 1.41, −1.41, −1.41, 1.41. Squared: 2, 2, 2, 2. Variance = 8/4 = **2**. ✓ Matches λ₂.

**Verify total variance is conserved:** 8 + 2 = 10 = 5 + 5 = Var(math) + Var(science). ✓

**Verify PC1 and PC2 are uncorrelated:**
Cov(PC1, PC2) = (2.83×1.41 + (−2.83)×(−1.41) + 2.83×(−1.41) + (−2.83)×1.41) / 4
= (4 + 4 − 4 − 4) / 4 = 0. ✓

**Interpretation.**

PC1 direction (1/√2, 1/√2): equal positive weight on both subjects. This component measures **overall academic performance** — the sum of both scores. Students A and C have high PC1 (above average on both exams); B and D have low PC1 (below average on both).

PC2 direction (1/√2, −1/√2): positive weight on math, negative weight on science. This component measures **math-vs-science relative strength** — the difference between scores. Students A and D have positive PC2 (relatively stronger in math); B and C have negative PC2 (relatively stronger in science).

**Variance explained:**
- PC1 captures 8/10 = **80%** of total variance.
- PC2 captures 2/10 = **20%** of total variance.

If you kept only PC1, you would represent each student with a single number instead of two, retaining 80% of the variation. You would know each student's overall ability but lose the math-vs-science distinction.

---

## 14. What PCA Finds and Why It Finds It

**What PCA finds:**

1. **The directions of maximum variance in the data** — the eigenvectors of the covariance matrix. These are the natural axes of the data's elliptical shape.

2. **The amount of variance along each direction** — the eigenvalues. These quantify how "important" each direction is (in the variance sense).

3. **New coordinates for each observation** — the principal component scores. These are the data's positions in the rotated coordinate system.

**Why it finds these particular things:**

The answer is not a matter of convention or design. PCA solves a specific optimization problem — maximize w^T C w subject to ||w|| = 1. The mathematics (Lagrange multipliers applied to a quadratic form constrained to the unit sphere) forces the solution to be eigenvectors of C, with eigenvalues as the captured variances. There is no freedom in this derivation. Given the data, the answer is fully determined.

The deeper reason is that the covariance matrix C encodes all pairwise linear relationships in the data. Its eigenvectors are the "natural coordinates" in which these relationships decouple — each eigenvector captures an independent axis of variation. The eigenvalue decomposition is, in a precise sense, the most informative way to decompose a symmetric matrix into independent components.

**What it reveals about the data:**

- **Dominant patterns of co-variation.** If many variables rise and fall together, PCA captures their shared movement as a single component. The loadings tell you which variables participate and how strongly. In the exam example, PC1 captured the shared tendency for both scores to be simultaneously high or low.

- **Effective dimensionality.** If the first k eigenvalues account for (say) 95% of the total variance, the data is effectively k-dimensional. The remaining dimensions contain only 5% of the variation — often noise or negligible detail. A dataset with 100 variables but only 3 large eigenvalues "really" has about 3 degrees of freedom.

- **Independence structure.** The PCs are uncorrelated by construction. Complex correlation patterns among original variables are disentangled into independent axes of variation.

**What it does NOT find:**

- **Nonlinear relationships.** PCA finds only linear combinations. If data lies on a curve, a spiral, or any nonlinear manifold, PCA may give misleading results. The first PC of data arranged in a circle does not capture the circular structure — it finds a diameter, which misses the point.

- **Causation.** PC1 might look like "overall size" or "general intelligence." PCA has no notion of what causes what. It finds correlational structure. The semantic interpretation is yours to make — and yours to get wrong.

- **Small but important signals.** If a biologically crucial variable has low variance compared to others, PCA may bury it in a low-ranked component. High variance does not guarantee importance in every domain.

---

## 15. The Other View: Minimum Reconstruction Error

Pearson's 1901 formulation was different from Hotelling's 1933 formulation. Pearson asked: what line (or plane) is closest to the data points, measuring closeness by the sum of squared perpendicular distances?

This turns out to be the **exact same answer**. Here is why.

**Setup.** Take a centered data point x and project it onto unit vector w. The projected point in the original space is (x · w) × w. The reconstruction error is the vector from the projected point back to the original:

    error = x − (x · w)w

The squared reconstruction error for this point is:

    ||x − (x · w)w||²

**The Pythagorean connection.** The original point, its projection, and the origin form a right triangle. The Pythagorean theorem gives:

    ||x||² = (x · w)² + ||x − (x · w)w||²

This says: the squared length of x equals the squared projection plus the squared error. Rearranging:

    ||x − (x · w)w||² = ||x||² − (x · w)²

Summing over all observations and dividing by n:

    Average reconstruction error = (1/n) Σᵢ ||xᵢ||² − (1/n) Σᵢ (xᵢ · w)²

The first term, (1/n) Σᵢ ||xᵢ||², is a constant — it depends only on the data, not on w. The second term is the variance of the projections.

Therefore:

> **Minimizing average reconstruction error = Maximizing projected variance.**

They are two sides of the same coin, separated by a constant. This is why Pearson (minimize distance to line) and Hotelling (maximize projected variance) arrived at the same answer from completely different starting points.

This duality also gives a second intuition for what PCA finds: the first principal component is the line that the data clusters around most tightly. The second is the next-tightest line, perpendicular to the first. PCA finds the "skeleton" of the data cloud.

---

## 16. Subtleties and Practical Considerations

**Centering is non-negotiable.** PCA operates on the covariance matrix, which is defined relative to the mean. Without centering, PCA partly captures the location of the mean rather than the spread of the data. The first component could end up pointing toward the mean instead of along the direction of maximum variation. Always center.

**Scaling matters.** If Math scores range from 0 to 100 and Science scores range from 0 to 1, Math will dominate the covariance matrix — not because Math is more informative, but because its numerical scale is larger. PCA sees raw numbers, not meaning. Two approaches:

- **Covariance PCA (no scaling):** Use when all variables share the same units and the absolute magnitude of variation is meaningful. Example: measurements of the same physical quantity (all in centimeters).
- **Correlation PCA (standardize first):** Divide each centered variable by its standard deviation, making each variable have variance 1. Then apply PCA. This is equivalent to computing eigenvectors of the correlation matrix instead of the covariance matrix. Use when variables have different units or wildly different scales, or when you want each variable to contribute equally regardless of magnitude.

**Sign ambiguity.** If v is an eigenvector, so is −v. Both satisfy Av = λv (check: A(−v) = −Av = −λv = λ(−v)). PCA defines a line (two directions), not an arrow (one direction). The sign of loadings and scores can be flipped without changing the analysis. Do not over-interpret the sign — "positive loading on math" and "negative loading on math" are the same component viewed from opposite ends of the line.

**Eigenvalue degeneracy.** If two eigenvalues are equal (say λ₂ = λ₃), any direction in the plane spanned by their eigenvectors is equally valid as a principal component. The plane is well-defined; the particular axes within it are not. Different software may return different eigenvectors in this case, depending on numerical details. This is not a bug — the choice genuinely does not matter, because all directions in that plane capture the same variance.

**PCA vs. factor analysis.** These are often confused because both reduce dimensionality. The difference: PCA decomposes **total variance** (the entire covariance matrix). Factor analysis decomposes only **shared variance** (the off-diagonal covariance), attributing some variance to unique, variable-specific noise. PCA asks "what linear combinations capture the most total spread?" Factor analysis asks "what latent variables best explain the correlations between observed variables?" They often give similar results, but they are different methods with different assumptions and different targets.

---

## 17. Stress-Testing the Claims

**Claim: "PCA finds the most important directions."**

If this were literally true, PCA would always surface the meaningful structure in the data. Counter-example: a badly calibrated sensor adds large random noise to one variable. That variable now has the highest variance — but the variance is noise, not signal. PCA's first component will be dominated by this sensor's noise. "Important" in PCA means "highest variance." Whether high variance corresponds to genuine signal depends on the data. In well-behaved data (where signal variance exceeds noise variance), it usually does. In noisy data, PCA may need preprocessing (like removing known noise sources) to work well.

**Claim: "PCA removes correlation."**

The principal components are uncorrelated: Cov(PCi, PCj) = 0 for i ≠ j. But uncorrelated does not mean independent. Two variables can have zero covariance yet be statistically dependent. Classic example: let X be uniformly distributed on [−1, 1], and let Y = X². Then Cov(X, Y) = 0 (the positive and negative X values cancel), but Y is completely determined by X. PCA removes linear correlation. Nonlinear dependencies can survive untouched.

**Claim: "PCA reduces dimensionality."**

Full PCA — keeping all p components — is a rotation. No information is lost. Dimensionality reduction occurs only when you discard some components (keep k < p). This reduction is **lossy**: you lose the variance along the discarded directions. PCA makes this loss as small as possible by discarding the lowest-variance directions first. But "as small as possible" is not zero unless those eigenvalues are exactly zero (meaning the data was already lower-dimensional to begin with).

**Apparent tension: "PCA is a rotation (no information lost)" vs. "PCA reduces dimensionality (information lost)."**

Resolution: these describe different stages of the same procedure. The **rotation step** (computing all p components and their scores) is lossless — it is an orthogonal change of basis. The **truncation step** (discarding the last p−k components) is lossy. "PCA" colloquially refers to both steps together, which creates the apparent contradiction. More precisely: the eigenvector decomposition is lossless; the decision to keep only k components is where information is sacrificed. PCA's contribution is arranging the rotation so that the subsequent truncation loses the minimum possible variance.

**Claim: "The first PC captures the most variance of any single direction."**

True within the class of linear projections. But a nonlinear function of the variables could capture more "structure." If data lies on a circle, no linear direction captures the angular position well — yet angular position is what distinguishes the observations. PCA is provably optimal only among linear projections. For nonlinear structure, extensions like kernel PCA or autoencoders may do better.

---

## 18. Recap

The full chain of reasoning, compressed:

1. **Problem.** Many correlated variables contain redundant information. We want fewer, uncorrelated variables that capture the essential variation.

2. **Measure.** Variance — how much observations differ from each other — serves as the definition of "information" for this purpose.

3. **Covariance matrix.** The symmetric matrix C encodes all pairwise variance and covariance relationships among the variables.

4. **Goal.** Find the unit vector w that maximizes the variance of data projected onto it: maximize w^T C w subject to ||w|| = 1.

5. **Lagrange multipliers.** The constrained maximum requires the gradient of the objective to be parallel to the gradient of the constraint: 2Cw = 2λw, giving Cw = λw.

6. **Eigenvalue equation.** The optimal direction must be an eigenvector of C. The projected variance along that direction equals the eigenvalue λ.

7. **First principal component.** The eigenvector with the largest eigenvalue. It captures the single direction of maximum variance.

8. **Subsequent components.** The eigenvector with the k-th largest eigenvalue, perpendicular to all previous components. Each captures the maximum remaining variance.

9. **Total variance conserved.** The sum of eigenvalues equals the sum of original variances. PCA redistributes variance; it does not create or destroy it.

10. **Equivalence.** Maximizing projected variance is mathematically identical to minimizing reconstruction error (via the Pythagorean theorem).

11. **Interpretation.** Eigenvalues quantify importance (variance captured). Eigenvectors (loadings) reveal which original variables contribute to each component. Scores place each observation in the new coordinate system.

---

## 19. Why This Matters

PCA is foundational well beyond statistics textbooks.

**Data compression.** Image compression, signal processing, and genomics use PCA to reduce vast data to manageable dimensions while preserving the most information. A million-pixel image can be represented by a few hundred principal components with minimal visible loss.

**Noise reduction.** Low-variance components often correspond to measurement noise. Discarding them and reconstructing the data from high-variance components alone can clean the signal. This is used in spectroscopy, astronomy, and neuroimaging.

**Visualization.** Datasets in biology, finance, and machine learning routinely have hundreds or thousands of variables. PCA often reveals that 2–3 components explain most variation, enabling human-interpretable plots of high-dimensional data. The famous "PCA plot" in genomics, where thousands of genetic variants collapse to a 2D scatter showing population structure, is a direct application.

**Feature extraction.** In machine learning, PCA-derived components serve as inputs to predictive models, reducing overfitting and computation while preserving most of the predictive signal.

**Conceptual impact.** PCA introduced a profound idea: high-dimensional data often lives on a lower-dimensional subspace. The "true" dimensionality of a dataset can be far less than the number of measurements. This insight underpins modern manifold learning, compressed sensing, and the success of low-rank approximations throughout machine learning.

---

## 20. Historical Note

- **1901.** Karl Pearson publishes "On Lines and Planes of Closest Fit to Systems of Points in Space" in the *Philosophical Magazine*. He frames PCA as a geometric problem (best-fitting line/plane) without the term "principal component."

- **1933.** Harold Hotelling publishes "Analysis of a Complex of Statistical Variables into Principal Components" in the *Journal of Educational Psychology*. He formalizes PCA as eigenvector decomposition of the covariance matrix, introduces the term "principal component," and connects it to psychological testing.

- **1936.** Carl Eckart and Gale Young publish their theorem on low-rank matrix approximation, connecting PCA to the singular value decomposition (SVD). This provides an alternative computation path — decomposing the data matrix directly rather than forming the covariance matrix — that is more numerically stable and is what modern software actually uses.

- **The convergence.** That Pearson's geometric approach (minimum perpendicular distance) and Hotelling's algebraic approach (maximum projected variance) yield the same answer was recognized early. The Pythagorean equivalence (Section 15) makes this inevitable. Two researchers, three decades apart, solving apparently different problems, arrived at the same mathematical object — because the mathematics left no room for a different answer.
