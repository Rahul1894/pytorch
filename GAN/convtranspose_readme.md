# ConvTranspose2d in PyTorch

This README provides an overview of the ConvTranspose2d layer in PyTorch, including the underlying theory, the output shape formula, and a small code example to illustrate its usage.

---

## 1. Theory

**ConvTranspose2d**, often referred to as "deconvolution" or "fractionally-strided convolution", is used to perform learnable upsampling in convolutional neural networks. While a standard `Conv2d` layer reduces spatial dimensions by sliding a kernel over the input feature map with a stride, `ConvTranspose2d` inversely projects smaller feature maps into larger ones, enabling the network to generate higher-resolution outputs.

Key points:

* **Learned Upsampling**: Unlike fixed interpolation (e.g., nearest or bilinear), ConvTranspose2d learns the optimal filters for expanding feature maps.
* **End-to-End Differentiable**: The upsampling process is integrated into training, so gradient-based optimization tunes these filters.
* **Parameter Efficiency**: One layer both upsamples spatial dimensions and learns feature transformations via convolution.

---

## 2. Output Shape Formula

For a single spatial dimension, given:

* $H_{in}$: input height (or width)
* $K$: kernel size
* $S$: stride
* $P$: padding
* $O$: output\_padding

The output height (or width) $H_{out}$ is computed as:

$$
H_{out} = (H_{in} - 1) \times S - 2P + K + O
$$

Where:

* $(H_{in} - 1) \times S$ spreads out input "pixels" by the stride.
* $-2P$ undoes the effect of padding that would have been applied in a forward convolution with the same parameters.
* $+K$ accounts for the size of the kernel’s receptive field.
* $+O$ allows fine-grained control if the exact desired output size is not met by the other terms.

Example manual check for a layer that upsamples from 4×4 to 8×8:

```
ConvTranspose2d(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1,
    output_padding=0
)
```

Plugging in: $H_{in}=4, K=4, S=2, P=1, O=0$:

$$
H_{out} = (4 - 1) \times 2 - 2\times1 + 4 + 0 = 6 - 2 + 4 = 8
$$

Hence, output size is 8×8.

---

## 3. Small Code Example

Below is a minimal PyTorch example demonstrating how to use `ConvTranspose2d` in a simple generator block (as found in GANs):

```python
import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, feature_map_size, out_channels):
        super(SimpleGenerator, self).__init__()
        self.net = nn.Sequential(
            # Input: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_map_size * 4, kernel_size=4, stride=1, padding=0),  # -> (N, feature_map_size*4, 4, 4)
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1),  # -> (N, feature_map_size*2, 8, 8)
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1),        # -> (N, feature_map_size, 16, 16)
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size, out_channels, kernel_size=4, stride=2, padding=1),                # -> (N, out_channels, 32, 32)
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Example usage:
if __name__ == '__main__':
    batch_size = 8
    latent_dim = 100
    feature_map_size = 64
    out_channels = 3

    noise = torch.randn(batch_size, latent_dim, 1, 1)
    gen = SimpleGenerator(latent_dim, feature_map_size, out_channels)
    fake_images = gen(noise)
    print("Generated images shape:", fake_images.shape)  # Should print (8, 3, 32, 32)
```

---

## 4. References

* **PyTorch Documentation**: [`torch.nn.ConvTranspose2d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
* **DCGAN Paper**: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)
