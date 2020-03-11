# Patch for NVIDIA's Automatic Mixed Precision (AMP) Package

Just comment out the following ```if: raise Exception``` statement starting at line 75 in the ```setup.py``` :

```
if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):
    raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does " +
                       "not match the version used to compile Pytorch binaries.  " +
                       "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda) +
                       "In some cases, a minor-version mismatch will not cause later errors:  " +
                       "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
                       "You can try commenting out this check (at your own risk).")oes " +
``` 