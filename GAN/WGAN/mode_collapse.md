# 🎯 Understanding Mode Collapse in GANs and How WGAN Solves It

This README explains the concept of **mode collapse**, a key issue in training GANs, and how **Wasserstein GAN (WGAN)** provides a better approach by using **Wasserstein distance** instead of Jensen-Shannon divergence.

---

## ❓ What is Mode Collapse?

**Mode collapse** is a common failure in Generative Adversarial Networks (GANs) where the **generator produces limited types of outputs**, despite the diversity in the real data distribution.

### 🧠 Why Does It Happen?

- The generator finds a few outputs that **fool the discriminator well**.
- It keeps producing those few outputs without learning other variations.
- The discriminator fails to provide meaningful gradients to encourage diversity.
- As a result, the **diversity of generated data collapses**.

### 📷 Example:

Consider a dataset of digits (0–9). The generator might only produce the digits `3` and `7`. Since these examples successfully fool the discriminator, the generator gets no incentive to generate other digits, like `0`, `1`, or `9`.

---

## 🧮 Aren’t Both GAN and WGAN Matching PDFs?

Yes — **both GAN and WGAN try to align the generator’s distribution \( P_g \) with the real data distribution \( P_r \)**.

However, **how they measure the “distance” between these distributions is fundamentally different**, which affects gradient quality, stability, and mode diversity.

---

## ⚖️ GAN vs WGAN: How They Measure Distribution Distance

### 🔹 GAN: Jensen-Shannon Divergence

The original GAN objective is:

\[
\min_G \max_D \mathbb{E}_{x \sim P_r}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]
\]

This objective **implicitly minimizes the Jensen-Shannon (JS) divergence** between \( P_r \) and \( P_g \).

#### 🚫 Limitations:

- When \( P_r \) and \( P_g \) are **non-overlapping or far apart**, the **JS divergence becomes constant or undefined**, resulting in **vanishing gradients**.
- This leads to:
  - **Unstable training**
  - **Mode collapse**

---

### 🔸 WGAN: Wasserstein-1 (Earth Mover's) Distance

WGAN replaces JS divergence with the **Wasserstein-1 distance**:

\[
W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]
\]

Where:
- \( \Pi(P_r, P_g) \) is the set of all joint distributions whose marginals are \( P_r \) and \( P_g \)
- This measures how much "mass" must be moved and how far to transform one distribution into another

#### ✅ Benefits:

- **Provides non-zero gradients** even when distributions are disjoint
- Leads to **stable training**
- Helps **reduce mode collapse**
- The loss value **correlates with image quality**

---

## 🔍 Summary: JS vs Wasserstein

| Feature                          | JS Divergence (GAN)               | Wasserstein Distance (WGAN)             |
|----------------------------------|-----------------------------------|-----------------------------------------|
| When distributions don’t overlap | Gradient vanishes                 | Meaningful gradient                     |
| Sensitivity to support           | High                              | Low                                     |
| Gradient stability               | Poor                              | Stable                                  |
| Signal quality                   | Low if distributions far apart    | Informative and smooth                  |

---

## 💡 In Short

- **GANs and WGANs both aim to match the generated data distribution to the real one.**
- **GANs** use **JS divergence**, which can lead to vanishing gradients and **mode collapse**.
- **WGAN** uses **Wasserstein distance**, which:
  - Is better behaved mathematically
  - Always provides useful gradients
  - Encourages diversity in generated outputs
  - Makes training more stable

---

## 📚 References

- [Original GAN paper (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)
- [Improved WGAN with Gradient Penalty (Gulrajani et al., 2017)](https://arxiv.org/abs/1704.00028)

---

Let us know if you'd like a code example or visual illustration of mode collapse in action.
