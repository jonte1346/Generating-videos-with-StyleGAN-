# StyleGAN Latent-Space Playground

This small playground demonstrates how to:

1. **Generate deterministic random faces** with StyleGAN‑3 (FFHQ, 256×256).
2. **Interpolate** smoothly between two latent codes and assemble the result into an MP4.
3. **Encode a real photograph** into the W⁺ latent space with a pre‑trained pSp encoder and render it with StyleGAN‑2 (FFHQ, 1024×1024).

---

## Quick look

The two images above are the very first faces produced from seeds **0** and **1**.  The latent walk between them is available as `media/interpolation_seed0‑seed1.mp4`.

---

## How it works

> All the heavy lifting is done by [StyleGAN‑3](https://github.com/NVlabs/stylegan3), [StyleGAN‑2](https://github.com/NVlabs/stylegan2) and [pixel2style2pixel (pSp)](https://github.com/eladrich/pixel2style2pixel).  This repo just wires them together in a reproducible way.

### 1. Set‑up the generator

```python
# load StyleGAN‑3 FFHQ 256×256 (t configuration)
G = legacy.load_network('models/stylegan3-t-ffhq-256x256.pkl')["G_ema"].eval()
for p in G.parameters():
    p.requires_grad_(False)  # inference‑only
```

- Switching to `.eval()` mode and freezing the parameters (`requires_grad_=False`) ensures **deterministic** output.
- Latent codes are sampled from **𝒩(0, 1)** → **z**.
- z → mapping network → **w** → replicate to 16 layers → **w⁺** → synthesis network → **image**.
- Output in **[0, 1]** is linearly remapped to **[0, 255]**, channel order flipped to **BGR** so that OpenCV can display / save it.

### 2. Latent interpolation

```python
alpha_values = np.linspace(0.0, 1.0, num_frames)  # 0 → 1
for alpha in alpha_values:
    w_interp = (1 - alpha) * w0 + alpha * w1
    frame = G.synthesis(w_interp)[0]  # torch.Tensor, [C, H, W]
    save_frame(frame)
```

1. Keep the same random seeds as above so `w0` and `w1` are unchanged.
2. Linearly blend in **W** (or **W⁺**) space.
3. Render each frame, scale/convert, and write out with **ffmpeg**.

### 3. Encoding a real face with pSp

```python
psp = load_psp_encoder('models/psp_ffhq_encode.pt').eval()
latents_w_plus = psp(preprocess(image))  # shape [1, 18, 512]
```

- pSp is trained against **StyleGAN‑2**, whose synthesis network expects **18** style layers, so we switch the generator to `stylegan2‑ffhq‑1024x1024.pkl` for this part.
- The resulting latent walk (generated without any added noise) is saved as `media/psp_interpolation.mp4`.

---


## Running the demo

Checkout the freshly saved MP4s or open the notebooks for a step‑by‑step walkthrough.


