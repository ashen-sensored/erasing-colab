import torch
import argparse
import safetensors.torch


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, help="output path")
parser.add_argument("input_models", type=str, nargs="+", help="input models")

args = parser.parse_args()

result_dict = {}

for input_model in args.input_models:
    if input_model.endswith(".ckpt"):
        input_model = torch.load(input_model, map_location="cpu")
        if "model" in input_model:
            input_model = input_model["model"]
    elif input_model.endswith(".safetensors"):
        input_model = safetensors.torch.load_file(input_model, device="cpu")
    for key, value in input_model.items():
        if key not in result_dict:
            result_dict[key] = value
        else:
            print(f"warning: key {key} already in result_dict")
            result_dict[key] = value




# output_dict = {}
# output_dict['model'] = gilgen_delta_state_dict
if args.output_path.endswith(".ckpt"):
    state_dict_save = {'state_dict': result_dict}
    torch.save(result_dict, args.output_path)
elif args.output_path.endswith(".safetensors"):
    safetensors.torch.save_file(result_dict, args.output_path)


# for key, value in input_model_unet_state_dict.items():
#     if key not in gilgen_unet_state_dict:
#         print(f"model key {key} not in gilgen_sample")
#     gilgen_unet_state_dict[key] = value.to(dtype=torch.float32)
#
# input_model_clip_state_dict = {k[len('cond_stage_model.'):]: v for k, v in input_model.items() if k.startswith('cond_stage_model.')}
# for key, value in input_model_clip_state_dict.items():
#     if key not in gilgen_sample['text_encoder']:
#         print(f"text_encoder key {key} not in gilgen_sample")
#     gilgen_sample['text_encoder'][key] = value.to(dtype=torch.float32)
#
#
# torch.save(gilgen_sample, args.output_path)
#
#
#
# print('output finished')
# # torch.save(save, args.input_path))

