import numpy as np

a = np.random.randn(4,5).astype(np.float32)
print(a)
np.save("./test_data",a)
np.savez("./test_data",data=a)
