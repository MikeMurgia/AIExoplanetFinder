# Exoplanet Transit Detection Using Machine Learning

A machine learning project to detect exoplanets by analyzing light curves from NASA's Kepler and TESS missions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Science: Understanding Exoplanets](#the-science-understanding-exoplanets)
3. [The Transit Method](#the-transit-method)
4. [Data Sources](#data-sources)
5. [Mathematical Foundations](#mathematical-foundations)
   - [Linear Algebra](#linear-algebra)
   - [Calculus and Derivatives](#calculus-and-derivatives)
   - [Gradient Descent](#gradient-descent)
   - [Probability](#probability)
6. [Neural Network Fundamentals](#neural-network-fundamentals)
7. [Project Roadmap](#project-roadmap)
8. [Setup and Installation](#setup-and-installation)
9. [Resources](#resources)

---

## Project Overview

### Goal

Build a machine learning system that:

1. Takes a **light curve** (brightness measurements over time) as input
2. Outputs a **prediction**: "this is probably a planet" or "this is probably not a planet"
3. Provides a **confidence score** for how certain the model is

### Why Machine Learning?

- **Tiny signals**: Earth-sized planets cause only 0.01% brightness drops
- **Stellar noise**: Stars have their own variability that can mask or mimic transits
- **False positives**: Eclipsing binary stars can look like planet transits
- **Scale**: TESS monitors millions of stars — humans can't examine every light curve manually

---

## The Science: Understanding Exoplanets

### What Is an Exoplanet?

An **exoplanet** is a planet that orbits a star other than our Sun. The prefix "exo" means "outside" — so exoplanet means a planet outside our solar system.

- Our solar system has 8 planets orbiting the Sun
- Other stars also have planets orbiting them
- Over 5,000 exoplanets have been confirmed to date
- Thousands more candidates await verification

### Why Can't We See Them Directly?

Stars are incredibly bright; planets are incredibly dim.

**Analogy**: Imagine standing in New York while someone in Los Angeles turns on a massive spotlight (the star). A tiny firefly orbits that spotlight (the planet). Could you see the firefly from New York? The spotlight's glare completely overwhelms it.

The star is millions to billions of times brighter than the planet. The planet's faint reflected light gets lost in the star's glare.

**Solution**: Instead of looking for planets directly, we look for the *effect* planets have on their stars.

---

## The Transit Method

### Core Concept

When a planet passes between its star and Earth, it blocks a tiny amount of the star's light. By measuring a star's brightness very carefully over time, we can detect this small dip.

### Step-by-Step Explanation

**Step 1: Constant Brightness**

A star without a transiting planet has relatively constant brightness. Measuring it over time yields steady readings:

```
Brightness: 100, 100, 100, 100, 100...
```

**Step 2: The Dip**

When a planet passes in front, it blocks some light, causing a small decrease:

```
Brightness: 100, 100, 99.9, 99.9, 100, 100...
```

**Step 3: Periodic Dips**

Since the planet orbits repeatedly, the dips occur at regular intervals — once per orbit:

```
Brightness
    |    ~~~~~~\    /~~~~~~~~\    /~~~~~~~~
    |           \  /          \  /
    |            \/            \/
    |______________________________________ Time
              Transit 1    Transit 2
```

### What the Light Curve Tells Us

A **light curve** is a graph of brightness (flux) versus time.

| Feature | What It Reveals |
|---------|-----------------|
| **Depth of dip** | Size of the planet (bigger planets block more light) |
| **Duration of dip** | Orbital distance and speed |
| **Time between dips** | Orbital period (how long one orbit takes) |
| **Shape of edges** | Orbital geometry and limb darkening |

### Transit Depth and Planet Size

The transit depth is the fraction of starlight blocked:

```
Transit Depth = (Planet Radius / Star Radius)²
```

**Examples**:
- Jupiter-sized planet around Sun-sized star: ~1% dip
- Earth-sized planet around Sun-sized star: ~0.01% dip

This is why detecting Earth-sized planets is challenging — the signal is 100× smaller.

### Geometric Constraints

We can only detect transits when the planet's orbit is aligned edge-on to our line of sight. If the orbit is tilted, the planet passes "above" or "below" the star from our perspective.

For an Earth-like planet around a Sun-like star, there's roughly a **0.5% chance** the geometry allows detection. For every transiting planet we find, approximately 200 more exist that we cannot detect this way.

---

## Data Sources

### Space Missions

#### Kepler Mission (2009–2018)

- Stared at one patch of sky (~150,000 stars) for 4 years
- Found over 2,700 confirmed planets
- Data is extensively labeled (confirmed planets vs. false positives)
- **Best for training**: Decade of analysis provides gold-standard labels

#### TESS Mission (2018–present)

- Surveys almost the entire sky
- Observes each region for ~27 days
- Better at finding short-period planets
- Over 400 confirmed planets, thousands of candidates
- **Best for discovery**: Less thoroughly analyzed, opportunities for contribution

### Data Archives

| Archive | URL | Contents |
|---------|-----|----------|
| MAST | https://mast.stsci.edu | Primary archive for Kepler/TESS light curves |
| NASA Exoplanet Archive | https://exoplanetarchive.ipac.caltech.edu | Confirmed planets and properties |
| ExoFOP | https://exofop.ipac.caltech.edu | Candidate tracking and dispositions |

### Light Curve File Contents

Downloaded light curves (FITS format) contain:

| Field | Description |
|-------|-------------|
| **Time** | Timestamps in BJD (Barycentric Julian Date) |
| **Flux** | Brightness measurements (normalized to ~1.0) |
| **Flux Error** | Uncertainty in each measurement |
| **Quality Flags** | Indicators of potential data issues |
| **Metadata** | Star position, brightness, catalog IDs |

### Label Categories

| Category | Code | Description |
|----------|------|-------------|
| Confirmed Planet | CP | Verified by follow-up observations |
| False Positive | FP | Ruled out (eclipsing binary, variability, artifact) |
| Planet Candidate | PC | Promising but unconfirmed |

---

## Mathematical Foundations

### Linear Algebra

Linear algebra is the language of data in machine learning. All data is represented as numbers organized in vectors and matrices.

#### Scalars, Vectors, and Matrices

**Scalar**: A single number

```
Examples: 5, -3.2, 0.001
Use: A single brightness measurement
```

**Vector**: An ordered list of numbers

```
Examples: [1, 2, 3] or [0.98, 1.01, 0.99, 1.00]
Use: A light curve's flux measurements (1000 measurements = 1000-element vector)
Notation: Bold (x) or arrow (→x)
```

**Matrix**: A grid of numbers with rows and columns

```
Example:
[1  2  3]
[4  5  6]

This is a 2×3 matrix (2 rows, 3 columns)
Use: Dataset of 1000 light curves with 500 measurements each = 1000×500 matrix
```

#### Vector Operations

**Addition**: Add corresponding elements

```
[1, 2, 3] + [4, 5, 6] = [5, 7, 9]
```

**Scalar Multiplication**: Multiply every element by the scalar

```
2 × [1, 2, 3] = [2, 4, 6]
```

**Dot Product**: Multiply corresponding elements, then sum

```
[1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

**Why the Dot Product Matters**

The dot product is the fundamental building block of neural networks. When a neuron processes inputs, it computes a dot product of the input vector with a weight vector.

Intuition: The dot product measures "how similar" two vectors are:
- Same direction → large positive value
- Opposite directions → large negative value  
- Perpendicular (unrelated) → zero

#### Matrix Multiplication

Matrix multiplication extends the dot product. Each row of the matrix performs a dot product with the vector:

```
[1  2]     [5]     [(1×5 + 2×6)]     [17]
[3  4]  ×  [6]  =  [(3×5 + 4×6)]  =  [27]
```

**Why It Matters**: A neural network layer is essentially matrix multiplication followed by a simple function. The matrix contains learned weights.

---

### Calculus and Derivatives

#### What Is a Derivative?

A derivative measures **how fast something is changing**.

**Analogy**: You're driving a car. Your position changes over time. The derivative of position with respect to time is your **speed** — how fast your position changes.

- High speed = large derivative
- Stopped = zero derivative

**Mathematical Notation**: If y depends on x, the derivative is written as dy/dx

It answers: "If I increase x by a tiny amount, how much does y change?"

#### Why Derivatives Matter for Machine Learning

We want our model to make good predictions. "Good" means low error. We define a **loss function** that measures error, and we want to **minimize** it.

The derivative tells us:
- **If derivative > 0**: Increasing the parameter increases error → decrease it
- **If derivative < 0**: Increasing the parameter decreases error → increase it
- **If derivative = 0**: We might be at a minimum

The derivative points us toward the minimum.

#### Partial Derivatives

When a function depends on multiple variables, we use **partial derivatives**.

If z depends on both x and y:
- ∂z/∂x = how z changes as x changes (y held constant)
- ∂z/∂y = how z changes as y changes (x held constant)

Neural networks have thousands or millions of parameters. The **gradient** is the collection of all partial derivatives:

```
gradient = [∂L/∂w₁, ∂L/∂w₂, ∂L/∂w₃, ...]
```

#### The Chain Rule

The most important calculus concept for deep learning.

Neural networks are **composed functions** — output of one layer feeds into the next.

If f depends on g, and g depends on x:

```
df/dx = (df/dg) × (dg/dx)
```

**Example**:
- g(x) = 2x
- f(g) = g²
- What is df/dx?

Solution:
- dg/dx = 2
- df/dg = 2g
- df/dx = df/dg × dg/dx = 2g × 2 = 4g = 4(2x) = 8x

**Backpropagation** (the algorithm that trains neural networks) is just the chain rule applied systematically through all layers.

---

### Gradient Descent

#### The Core Algorithm

Gradient descent finds the parameter values that minimize the loss function.

**Procedure**:

1. **Initialize**: Start with random parameter values
2. **Forward Pass**: Compute prediction and loss
3. **Compute Gradient**: Calculate ∂Loss/∂parameter for each parameter
4. **Update**: Adjust parameters in the opposite direction of the gradient
5. **Repeat**: Until loss stops decreasing

**Update Rule**:

```
parameter_new = parameter_old - learning_rate × gradient
```

The **learning rate** controls step size (typically 0.001 to 0.1).

The minus sign ensures we move toward lower loss:
- Positive gradient → decrease parameter
- Negative gradient → increase parameter

#### Worked Example: Single Parameter

**Setup**:
- Data point: x = 2, y = 10
- Model: ŷ = wx (predict y by multiplying x by learned weight w)
- Loss: L = (ŷ - y)² = (wx - y)²
- Starting w = 0
- Learning rate = 0.1

**Derivative Calculation**:

```
L = (wx - y)²

Using chain rule:
- Let u = wx - y, so L = u²
- ∂L/∂u = 2u
- ∂u/∂w = x
- ∂L/∂w = 2u × x = 2(wx - y) × x
```

**Iteration 1**:

| Step | Value |
|------|-------|
| Current w | 0 |
| Prediction ŷ | 0 × 2 = 0 |
| Loss | (0 - 10)² = 100 |
| Gradient | 2(0 - 10) × 2 = -40 |
| Update | 0 - 0.1 × (-40) = **4** |

**Iteration 2**:

| Step | Value |
|------|-------|
| Current w | 4 |
| Prediction ŷ | 4 × 2 = 8 |
| Loss | (8 - 10)² = 4 |
| Gradient | 2(8 - 10) × 2 = -8 |
| Update | 4 - 0.1 × (-8) = **4.8** |

**Iteration 3**:

| Step | Value |
|------|-------|
| Current w | 4.8 |
| Prediction ŷ | 4.8 × 2 = 9.6 |
| Loss | (9.6 - 10)² = 0.16 |
| Gradient | 2(9.6 - 10) × 2 = -1.6 |
| Update | 4.8 - 0.1 × (-1.6) = **4.96** |

**Convergence Summary**:

| Iteration | w | Prediction | Loss | Gradient |
|-----------|------|------------|--------|----------|
| 1 | 0 | 0 | 100 | -40 |
| 2 | 4 | 8 | 4 | -8 |
| 3 | 4.8 | 9.6 | 0.16 | -1.6 |
| 4 | 4.96 | 9.92 | 0.0064 | -0.32 |
| 5 | 4.992 | 9.984 | 0.000256 | -0.064 |

We're converging to w = 5, which gives perfect prediction: 5 × 2 = 10.

#### Worked Example: Two Parameters

**Setup**:
- Data point: x = 2, y = 10
- Model: ŷ = wx + b
- Loss: L = (wx + b - y)²
- Starting: w = 0, b = 0
- Learning rate: 0.1

**Partial Derivatives**:

```
∂L/∂w = 2(wx + b - y) × x
∂L/∂b = 2(wx + b - y) × 1
```

**Iteration 1**:

| Step | Value |
|------|-------|
| Current w, b | 0, 0 |
| Prediction ŷ | 0 × 2 + 0 = 0 |
| Loss | (0 - 10)² = 100 |
| ∂L/∂w | 2(0 - 10) × 2 = -40 |
| ∂L/∂b | 2(0 - 10) × 1 = -20 |
| New w | 0 - 0.1 × (-40) = **4** |
| New b | 0 - 0.1 × (-20) = **2** |

**Iteration 2**:

| Step | Value |
|------|-------|
| Current w, b | 4, 2 |
| Prediction ŷ | 4 × 2 + 2 = 10 |
| Loss | (10 - 10)² = **0** |

Perfect prediction achieved! The model learned w = 4, b = 2, giving 4 × 2 + 2 = 10.

#### Common Pitfalls

**Learning Rate Too High**:

```
Loss
  |  *        *
  |   \      /
  |    \    /
  |     \  /
  |      *
  +-------------- iterations

Loss oscillates or explodes instead of decreasing smoothly.
```

**Learning Rate Too Low**:

```
Loss
  |****
  |    ****
  |        ****
  |            ****
  +----------------------- iterations

Training takes thousands of iterations to converge.
```

**Local vs Global Minima**:

```
Loss
  |    *
  |   * *    *
  |  *   *  * *
  | *     **   *
  |*            *
  +--------------- w
     ↑       ↑
   local   global
   min     min
```

Gradient descent finds *a* minimum, not necessarily the *best* minimum.

---

### Probability

#### Why Probability?

Real data is noisy. When we see a dip in a light curve, we can't be 100% certain it's a planet. Probability lets us express confidence levels.

Machine learning models often output probabilities:
- Instead of: "This is a planet"
- We get: "There's an 87% chance this is a planet"

#### Basic Concepts

| Concept | Definition | Example |
|---------|------------|---------|
| **Probability** | Number between 0 (impossible) and 1 (certain) | P(rain) = 0.3 |
| **Event** | Something that might happen | "Light curve contains a transit" |
| **P(A)** | Probability of event A | P(planet) = 0.01 |
| **P(A\|B)** | Probability of A given B occurred | P(planet \| deep transit) |

#### Bayes' Theorem

Tells us how to update beliefs based on evidence:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

**For Exoplanets**:
- A = "This star has a transiting planet"
- B = "We observe a 1% brightness dip"

To find P(A|B) — probability of a planet given we see a dip:

- **P(B|A)**: If there IS a planet, how likely is this dip?
- **P(A)**: Prior probability any star has a transiting planet (~1%)
- **P(B)**: Overall probability of seeing this dip (from any cause)

#### Probability Distributions

**Normal (Gaussian) Distribution**: The bell curve. Noise in measurements often follows this pattern — most values cluster near the mean, fewer at extremes.

**Bernoulli Distribution**: Binary outcomes (success/failure, planet/not-planet).

---

## Neural Network Fundamentals

### How Math Becomes Machine Learning

1. **Data as vectors**: Each light curve is a vector of flux values
2. **Model as matrix operations**: Layer = matrix multiplication + bias + activation function
3. **Loss function**: Measures prediction error (often involves probability)
4. **Optimization**: Gradient descent adjusts weights to minimize loss
5. **Learning**: Repeat forward pass → loss → backward pass → update

### A Neural Network Layer

**Components**:
- Input vector: x
- Weight matrix: W
- Bias vector: b
- Activation function: f (e.g., ReLU)

**Forward Pass**:

```
z = W × x + b       (linear transformation)
a = f(z)            (activation function)
```

**Example**:
- Input: x = [1, 2]
- Weights: W = [[0.5, -0.3], [0.2, 0.8]]
- Bias: b = [0.1, -0.1]
- Activation: ReLU

Step 1 — Linear transformation:
```
z₁ = 0.5×1 + (-0.3)×2 + 0.1 = 0.5 - 0.6 + 0.1 = 0
z₂ = 0.2×1 + 0.8×2 + (-0.1) = 0.2 + 1.6 - 0.1 = 1.7
z = [0, 1.7]
```

Step 2 — ReLU activation:
```
ReLU(x) = max(0, x)
ReLU(0) = 0
ReLU(1.7) = 1.7
a = [0, 1.7]
```

### Activation Functions

Without activation functions, stacking layers would collapse into a single linear transformation. Activations introduce **nonlinearity**, allowing networks to learn complex patterns.

**ReLU (Rectified Linear Unit)**:
```
f(x) = max(0, x)

If x > 0: output x
If x ≤ 0: output 0
```

Popular because:
- Simple and fast to compute
- Derivative is simple (1 if positive, 0 if negative)
- Works well in practice

### Backpropagation

The algorithm for computing gradients in neural networks.

**Process**:
1. Forward pass: Input → Layer 1 → Layer 2 → ... → Output → Loss
2. Backward pass: Compute ∂Loss/∂weights by applying chain rule layer by layer
3. Update: Adjust all weights using gradient descent

**Chain Rule Through Layers**:

```
∂Loss/∂w₁ = ∂Loss/∂output × ∂output/∂layer₃ × ∂layer₃/∂layer₂ × ∂layer₂/∂layer₁ × ∂layer₁/∂w₁
```

Errors propagate **backward** through the network, hence "backpropagation."

### The Complete Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                           │
│                                                             │
│  1. FORWARD PASS                                            │
│     Input → Layer 1 → Layer 2 → ... → Output → Loss        │
│                                                             │
│  2. BACKWARD PASS (Backpropagation)                        │
│     Apply chain rule to compute gradient for each weight    │
│                                                             │
│  3. UPDATE WEIGHTS                                          │
│     w = w - learning_rate × ∂Loss/∂w                       │
│                                                             │
│  4. REPEAT                                                  │
│     Continue until loss converges                           │
└─────────────────────────────────────────────────────────────┘
```

---
#   A I E x o p l a n e t F i n d e r  
 