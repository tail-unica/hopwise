Training in FP16 and BF16
=========================

Hopwise supports mixed-precision training using FP16 (float16) and BF16 (bfloat16) to reduce
memory usage and speed up training on compatible hardware.

Overview
--------

Mixed-precision training uses lower precision (16-bit) floating point numbers for most operations
while maintaining full precision (32-bit) for critical computations. This provides:

- **Reduced memory usage**: Models use approximately half the GPU memory
- **Faster training**: Modern GPUs have dedicated tensor cores for 16-bit operations
- **Maintained accuracy**: Critical operations still use full precision

Configuration
-------------

Enable half-precision training in your configuration:

.. code:: yaml

    # FP16 training (recommended for NVIDIA GPUs)
    train_precision: fp16

    # BF16 training (recommended for newer GPUs like A100, RTX 40 series)
    train_precision: bf16

Or via command line:

.. code:: bash

    hopwise train --model BPR --dataset ml-100k --train_precision=fp16

Or in Python:

.. code:: python

    from hopwise.quick_start import run_hopwise

    config_dict = {
        'train_precision': 'fp16',
    }

    run_hopwise(model='BPR', dataset='ml-100k', config_dict=config_dict)

Choosing Between FP16 and BF16
------------------------------

+---------------+----------------------------------+----------------------------------+
| Precision     | Advantages                       | Best For                         |
+===============+==================================+==================================+
| FP16          | Widest GPU support               | V100, RTX 20/30 series           |
+---------------+----------------------------------+----------------------------------+
| BF16          | Better numerical stability,      | A100, H100, RTX 40 series        |
|               | no loss scaling needed           |                                  |
+---------------+----------------------------------+----------------------------------+

.. note::
    BF16 requires NVIDIA Ampere architecture (compute capability 8.0+) or newer.

Gradient Scaling (FP16)
-----------------------

When using FP16, hopwise automatically applies gradient scaling to prevent underflow:

.. code:: python

    # This is handled automatically, but you can customize:
    config_dict = {
        'train_precision': 'fp16',
        'grad_scaler_init_scale': 65536.0,  # Initial scale factor
    }

Best Practices
--------------

1. **Start with FP32**: Verify your model works in full precision first
2. **Monitor loss**: Watch for NaN or inf values during training
3. **Adjust learning rate**: You may need slightly different learning rates
4. **Use BF16 if available**: It's more stable and doesn't require gradient scaling

Troubleshooting
---------------

**NaN losses with FP16:**

- Try reducing the learning rate
- Use BF16 instead if your GPU supports it
- Increase the gradient scaler's initial scale

**Slow training:**

- Ensure your GPU has tensor cores (V100, A100, RTX series)
- Check that CUDA is properly configured

**Out of memory:**

- Half precision should reduce memory by ~50%
- Combine with gradient checkpointing for larger models