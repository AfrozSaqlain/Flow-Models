# Flow Matching for Generative Modeling

This project implements a **Continuous Normalizing Flow (CNF)** using the **Flow Matching (FM)** framework to learn and sample from a 2D checkerboard distribution.

## 🚀 Overview

The goal is to transform simple Gaussian noise ($p_0$) into a complex structured pattern ($p_1$) by learning a time-dependent vector field $v_\theta(x, t)$. Once trained, we can generate new samples by solving an Ordinary Differential Equation (ODE).

---

## 🧩 Core Components

### 1. Data Sampling (`sample.py`)
We use **Rejection Sampling** to generate the target checkerboard dataset. 
- **The Mask:** A $4 \times 4$ grid where cells are active if `(row + col) % 2 == 1`.
- **The Process:** We sample uniform points $(x, y) \in [-4, 4]^2$. If a point falls within an active cell of the mask, we keep it; otherwise, we discard it.
- **Result:** A set of discrete 2D points representing the "ground truth" distribution $p_1$.

### 2. Flow Matching Theory (`train.py`)
Instead of complex diffusion math, we use a simple **Linear Probability Path**:
$$x_t = (1 - t)x_0 + t x_1$$
where $x_0$ is noise and $x_1$ is data.

Taking the derivative with respect to time $t$ gives us the **Optimal Transport Velocity**:
$$\frac{dx_t}{dt} = x_1 - x_0$$
The neural network $v_\theta(x_t, t)$ is trained to regress directly onto this constant velocity using a Mean Squared Error (MSE) loss. This is significantly more efficient than traditional score-matching or GAN training.

### 3. Model Architecture (`fm.py`)
The model is a Multi-Layer Perceptron (MLP) with two critical features:
- **Sinusoidal Time Embeddings:** Maps the scalar time $t$ into a high-dimensional vector using sine/cosine phases. This allows the network to "understand" its position along the flow path with high precision.
- **Residual Connections ($y = x + f(x)$):** These act as "gradient highways," preventing the vanishing gradient problem. More importantly, they bias the network toward learning incremental updates, which is the discrete equivalent of the continuous ODE steps we solve during inference.

---

## 🛠️ How It Works (The Inference Pipeline)

1. **Start at $t=0$:** Sample 1000 points from a Standard Gaussian $\mathcal{N}(0, I)$.
2. **Iterate:** For 200 steps, move the points along the predicted velocity field:
   $$x_{t+dt} = x_t + v_\theta(x_t, t) \cdot dt$$
3. **End at $t=1$:** The points have "flowed" from random noise into the checkerboard shape.

## 📈 Key Insights from the Technical Report

- **Stability:** Without residual connections, the model often predicts extreme velocities, causing points to fly out of the $[-4, 4]$ bounds. Residuals ensure the flow remains smooth and contained.
- **Gaussian vs. Uniform:** Training with Gaussian noise ($x_0 \sim \mathcal{N}$) instead of Uniform noise ensures the model generalizes better during inference, as the source distribution matches the starting noise.
- **Euler Integration:** A simple first-order ODE solver (Euler) is sufficient for this linear path, though the model supports higher-order solvers like RK4 if needed.

---

## 🏃 Running the Project

1. **Install Dependencies:** `pip install torch matplotlib tqdm numpy`
2. **Train & Visualize:** `python train.py`
   - This will train the model for 100,000 steps and then display an animation of the noise transforming into the checkerboard.
