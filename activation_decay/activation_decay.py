import torch
from torchtune.modules import RotaryPositionalEmbeddings
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

dim = 1024         
num_heads = 1          
head_dim = dim // num_heads
seq_len = 50000    
base = 500_000         
batch_size = 1        

# scale down qk initialization
alpha = 1

rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=seq_len, base=base)

def constant_queries_and_keys():
    Q =  torch.ones(batch_size, seq_len, num_heads, head_dim) * alpha
    K =  torch.ones(batch_size, seq_len, num_heads, head_dim)* alpha
    
    Q_rotated = rope(Q)  
    K_rotated = rope(K) 
    
    Q_rotated = Q_rotated.reshape(seq_len, dim)
    K_rotated = K_rotated.reshape(seq_len, dim)
    
    query_0 = Q_rotated[0:1]  
    activations = torch.matmul(query_0, K_rotated.T).squeeze()  
    activations = activations / (dim ** 0.5) 
    return activations

def gaussian_queries_and_keys():
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)* alpha
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)* alpha

    Q_rotated = rope(Q)
    K_rotated = rope(K)
    
    Q_rotated = Q_rotated.reshape(seq_len, dim)
    K_rotated = K_rotated.reshape(seq_len, dim)
    
    query_0 = Q_rotated[0:1]
    activations = torch.matmul(query_0, K_rotated.T).squeeze()
    activations = activations / (dim ** 0.5)  
    return activations


activations_constant = constant_queries_and_keys()
activations_gaussian = gaussian_queries_and_keys()
relative_distances = torch.arange(seq_len)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(relative_distances, activations_constant, color='blue')
plt.title("Constant queries and keys")
plt.xlabel("Relative Distance")
plt.ylabel("Activation")
plt.grid(True) 

plt.subplot(1, 2, 2)
plt.plot(relative_distances, activations_gaussian, color='blue')
plt.title("Gaussian queries and keys")
plt.xlabel("Relative Distance")
plt.ylabel("Activation")
plt.grid(True)

plt.tight_layout()
plt.show()