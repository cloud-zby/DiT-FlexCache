import utmosv2
import torch
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)
device_eval = torch.device("cuda:1")
# model = utmosv2.create_model(pretrained=True,checkpoint_path="/gemini/space/ScoreTTS/checkpoints/utmos_v2/fold3_s42_best_model.pth",device="cuda:1")
model = utmosv2.create_model(pretrained=True,checkpoint_path="/gemini/space/ScoreTTS/checkpoints/utmos_v2/fold3_s42_best_model.pth",device=device_eval)

mos_list = []
path = "/gemini/space/ScoreTTS/Eval_Models/UTMOSv2-main/instruct_0.wav"
# afaik, increasing num_repetitions should improve the result
mos = model.predict(input_path=path, num_repetitions=5,remove_silent_section=True,device=device_eval)
print(mos)
mos_list.append(mos)

print(f"Mean MOS: {np.mean(mos_list):.3f} ± {np.std(mos_list):.3f}")
print(f"Min: {min(mos_list):.3f}, Max: {max(mos_list):.3f}")
print(f"Median: {np.median(mos_list):.3f}")
print(f"Num repetitions: {5}")
print(f"Mos list: {mos_list}")
print("=======================")