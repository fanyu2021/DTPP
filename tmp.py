import torch
# A = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# B = torch.tensor([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
# cat_result = torch.cat((A, B), dim = 0)
# print("原始张量A形状:", A.shape)
# print("原始张量B形状:", B.shape)
# print("拼接后张量形状:", cat_result.shape)
# print(cat_result)

# stack_result = torch.stack((A, B), dim = 0)
# print("原始张量A形状:", A.shape)
# print("原始张量B形状:", B.shape)
# print("堆叠后张量形状:", stack_result.shape)
# print(stack_result)

# Alist = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]]), torch.tensor([[9, 10, 11, 12]])]
# a = torch.tensor([[11,22,33,44]])
# Blist = [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
# stack_list_result = torch.stack(Alist + [a], dim = 0)
# print("原始列表A形状:", Alist)
# print("原始列表 Alist + a 形状:", Alist + [a])
# print("堆叠后张量形状:", stack_list_result.shape)
# print(stack_list_result)

# x = 0
# y = 20
# res = torch.linspace(x, y, steps=11)
# print(res)

