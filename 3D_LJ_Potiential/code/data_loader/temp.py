import torch

a = torch.load("/home/liuw/GitHub/Data/LLUF/3d/Validation/n64lt0.1stpstraj180_l_dpt20000.pt", map_location="cpu", weights_only=False)
print(type(a), a.keys())
for k in a:
    print(k, type(a[k]), a[k] if not hasattr(a[k], 'shape') else a[k].shape)

print(a['tau_short'], a['tau_long'])

train_data = a['qpl_trajectory'][:80, :, :16].clone()
val_data = a['qpl_trajectory'][80:100, :, :16].clone()
test_data = a['qpl_trajectory'][100:120, :, :16].clone()
print(train_data.shape, val_data.shape, test_data.shape)
torch.save({'qpl_trajectory':train_data, 'tau_short':float(a['tau_short']), 'tau_long':float(a['tau_long'])}, 
           "/home/liuw/GitHub/Data/LLUF/3d/Validation/thumbnail_train.pt")
torch.save({'qpl_trajectory':val_data, 'tau_short':float(a['tau_short']), 'tau_long':float(a['tau_long'])}, 
           "/home/liuw/GitHub/Data/LLUF/3d/Validation/thumbnail_val.pt")
torch.save({'qpl_trajectory':test_data, 'tau_short':float(a['tau_short']), 'tau_long':float(a['tau_long'])}, 
           "/home/liuw/GitHub/Data/LLUF/3d/Validation/thumbnail_test.pt")