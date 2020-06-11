import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#noise input
sz=256
img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255
print(img)
# variation
img_var = V(img[None], requires_grad=True) 
#
model = vgg16(pre=True).eval()
set_trainable(model, False)

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()