# Troubleshooting

## Loss inf/NaN

1. Using AMP:

   As it is stated here: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#advanced-topics:

   Disable autocast or GradScaler individually (by passing enabled=False to their constructor) and see if infs/NaNs persist.

2. Not using AMP:

   If you are not using AMP, try these following solutions:

   - Reduce the learning rate
   - Change the weight initialization
   - Set the gradient clipping (by using this: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html before the `optimizer.step()`)

## RuntimeError: no valid convolution algorithms available in CuDNN

it could occur because the VRAM memory limit was hit (which is rather non-intuitive from the error message).

For my case with PyTorch model training, decreasing batch size helped. You could try this or maybe decrease your model size to consume less VRAM.

## RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

Normally, you should not get this error because the dataloader is launched with  `multiprocessing_context` set to `spawn`. 
But if it is not enough, you just need to set `num_workers=0`, as it was stated [here](https://discuss.pytorch.org/t/not-using-multiprocessing-but-getting-cuda-error-re-forked-subprocess/54610/10)
