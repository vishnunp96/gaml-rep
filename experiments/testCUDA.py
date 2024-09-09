import torch

print(torch.cuda.is_available())

t_cpu = torch.zeros(1)

print(t_cpu)

t_gpu = torch.zeros(1).cuda()

print(t_gpu)
