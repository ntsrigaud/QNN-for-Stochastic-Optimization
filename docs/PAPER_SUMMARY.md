# **A Quantile Neural Network Framework for Two-stage Stochastic Optimization**

**Authors:** Antonio Alcántara, Carlos Ruiz, Calvin Tsay

---

### **Abstract**
Two-stage stochastic programming is a popular framework for optimization under uncertainty, where decision variables are split between first-stage decisions, and second-stage (or recourse) decisions, with the latter being adjusted after uncertainty is realized. These problems are often formulated using Sample Average Approximation (SAA), where uncertainty is modeled as a finite set of scenarios, resulting in a large “monolithic” problem where the model is repeated for each scenario. The resulting models can be challenging to solve, leading to several problem-specific decomposition approaches. An alternative approach is to approximate the expected second-stage objective value using a surrogate model, which can then be embedded in the first-stage problem to produce good heuristic solutions. 

In this work, we propose to instead model the distribution of the second-stage objective, specifically using a quantile neural network. Embedding this distributional approximation enables capturing uncertainty and is not limited to expected-value optimization; for example, the proposed approach enables optimization of the Conditional Value at Risk (CVaR). We discuss optimization formulations for embedding the quantile neural network and demonstrate the effectiveness of the proposed framework using several computational case studies including a set of mixed-integer optimization problems.

**Keywords:** Optimization under Uncertainty, Stochastic Programming, Neural Networks, Mixed-Integer Programming (MIP)

---

### **1. Introduction**
Mathematical optimization provides a powerful framework for solving decision-making problems, but conventional deterministic formulations rely on having exact estimates of involved model inputs and parameters. When uncertainty exists, stochastic programming (SP) approaches are preferred, as they deal with known distributions for uncertain inputs. SP has been applied in supply chain optimization, production scheduling, unit commitment, and process systems engineering.

Within SP, two-stage stochastic programming divides variables into first-stage (“here-and-now”) variables decided before the realization of uncertainty, and second-stage (“wait-and-see”) variables adjusted after the realization. The primary challenge is the computational expense, as uncertain parameters are represented using a set of scenarios where first-stage decisions are fixed across them, known as the Sample Average Approximation (SAA). SAA results in a deterministic, monolithic problem that repeats elements over many scenarios, which can quickly become practically intractable if the underlying model is large or the scenario set is high-dimensional. 

Existing work, such as the Neur2SP framework, uses neural networks to learn the expected second-stage objective value as a function of recourse variables. However, these are often limited to risk-neutral decisions and can suffer from scalability issues. In this work, we propose using **quantile neural networks (QNNs)** as the second-stage surrogate model. QNNs are multi-output networks that allow for learning the distribution of the second-stage objective value rather than just the expected value. This framework is computationally efficient, allows for fast data generation, and enables heuristic general solutions to be obtained quickly ($<1s$ in many cases) regardless of the scenario set size.

---

### **2. Background**

#### **2.1. Two-stage Stochastic Optimization**
A general representation is defined as:
$$\min_{X \in \mathcal{X}} E_{\xi}[F(X, \xi)] = \min_{X \in \mathcal{X}} c^TX + E_{\xi}[V(X, \xi)]$$

where $V(X, \xi)$ represents the second-stage value function:
$$V(X, \xi) = \min_{Y \in \mathcal{Y}(X, \xi)} f(Y, X, \xi)$$

The SAA approach samples the distribution $P$ into a finite set of scenarios $\xi_{\omega}$. However, this involves $\omega$ duplicates of the second-stage objective and feasibility region, leading to a tradeoff between tractability and accuracy. For risk management, mean-risk formulations are used:
$$\min_{X \in \mathcal{X}} E_{\xi}[F(X, \xi)] + \lambda R_{\xi}[F(X, \xi)]$$

A popular measure is **Conditional Value-at-Risk (CVaR)**, defined for a random variable $Z$ and confidence level $\alpha$ as:
$$CVaR_{\alpha} = E[Z | Z \geq VaR_{\alpha}(Z)]$$

#### **2.2. Quantile Neural Networks**
Quantile regression estimates the $\tau$-th quantile of the conditional distribution, $Q_{\tau}(y|X)$. The model is fitted by minimizing the **quantile (pinball) loss**:
$$\theta = \text{argmin} \frac{1}{N} \sum_{i=1}^{N} [\tau \epsilon_i I_{\epsilon_i \geq 0} + (1-\tau) \epsilon_i I_{\epsilon_i < 0}]$$

QNNs extend this into the neural network paradigm, simultaneously estimating multiple quantiles by minimizing the mean quantile loss across the dataset. 

#### **2.3. Mixed-Integer Formulations for ReLU Neural Networks**
When activation functions $g[l]$ are piecewise linear (like ReLU), the neural network can be embedded as constraints of a mixed-integer linear program (MILP). A popular method is the **big-M method**, which produces a mixed-integer formulation for each ReLU activation node using auxiliary binary variables and big-M constants derived from interval arithmetic. 

---

### **3. Methodology**

#### **3.1. Quantile Neural Network Methodology**
We propose using a QNN as a surrogate model for the second stage, training it to learn the mapping $X \rightarrow Q_{\tau}(V(X, \xi))$. This replaces scenarios in the optimization problem with a piecewise-linear approximation using ReLU activation functions. 

One issue is the **"quantile crossing"** phenomenon, where the non-decreasing property of conditional quantile estimations is violated. To solve this, we introduce the **Incremental Quantile Neural Network (IQNN)**, which predicts increments in the quantile function using ReLU activations in the output layer to ensure monotonicity by design.

#### **3.2. Data Generation**
Algorithm 1 describes the fast data generation procedure:
1. Generate $N$ random feasible inputs $X_i$ and scenario realizations $\xi_i$.
2. Solve the single-scenario second-stage optimization problem for each to find value $v_i$.
3. Save the $(X_i, v_i)$ pairs.

#### **3.3. Problem Formulation**
The surrogate problem is formulated by embedding the trained (I)QNN. The objective function can be set for:
*   **Risk-Neutral:** Mean value of the predicted conditional quantiles.
*   **Risk-Averse:** Mean of the predicted distribution's right-hand tail (approximating CVaR).

For the standard QNN, a tolerance parameter $\Delta$ is included to manage quantile crossing, selected through a prescriptive procedure (Algorithm 2).

---

### **4. Case Studies**

#### **4.1. Experimental setup**
Problems studied include the Capacitated Facility Location Problem (**CFLP-n-m**) and an Investment Problem (**IP-I-H**). Both (I)QNN architectures were trained using a grid-search across 100 configurations.

#### **4.2. Dataset size impact**
Experiments show that the framework performs well with at least **10,000 samples**, showing diminishing returns in validation accuracy beyond that point.

#### **4.3. Results for risk-neutral optimization**
(I)QNN solvers obtain solutions in **less than one second** for most problems. Compared to SAA, the QNN structure showed improvements in the true objective of up to **14.5%** for large scenario sets, as SAA often fails to reach optimality within time limits.

#### **4.4. Results for risk-averse optimization**
The same trained (I)QNN model can be used for various risk-aversion levels by adjusting $\lambda$ and $\alpha$. While SAA often reached its 2-hour time limit, the IQNN-based framework required no more than **0.12 seconds** to produce a solution.

| Model | Solving Time (s) | Best Objective Improvement over SAA |
| :--- | :--- | :--- |
| **QNN** | 0.01 - 5.26 | 14.32% |
| **IQNN** | < 0.12 | 11.49% |
| **SAA** | 7,200 (limit) | Baseline |

*(Summarized from Tables 4 and 5)*

---

### **5. Conclusions**
This paper introduces an innovative Quantile Neural Network-based framework as an alternative to SAA for two-stage stochastic programs. The framework is fast for data generation and solving the surrogate problem, while its distributional nature allows for both risk-neutral and risk-averse (CVaR) optimization. The framework stands out for large-scale problems where SAA becomes intractable. Future research could include improving cost tail modeling and online adjustments of the surrogate model.

---

### **Appendix A: Model Selection**
*   **Hyper-parameters:** Hidden layers were fixed to one; batch size, learning rate, and dropout were tuned via random search.
*   **Quantile levels:** 50 equally-spaced quantiles ($\tau$ from 0.01 to 0.99).
*   **Tolerance Selection:** Evaluated $\Delta$ values of 0, 10, 50, 100, and 500.

