import torch
from lmms import config_utils
from data import load_viddiff_dataset as lvd
from lmms import lmm_utils as lu
import eval_viddiff
from apis import qwen_api
import json

config= "lmms/configs/config.yaml"
name = "qwen2vl_7b_open_easy"
split = "easy"
eval_mode = "open"
model= "Qwen/Qwen2-VL-7B-Instruct"
subset_mode="0"

args = config_utils.load_config(
        config,
        name=name,
        split=split,
        eval_mode=eval_mode,
        model=model,
        subset_mode=subset_mode,
    )
dataset = lvd.load_viddiff_dataset(
    [args.data.split], args.data.subset_mode, cache_dir=None
)
videos = lvd.load_all_videos(
    dataset, do_tqdm=True, cache=True, cache_dir="cache/cache_data"
)
n_differences = dataset["n_differences_open_prediction"]  # for open eval only

# make prompts and call the lmm
batch_prompts_text, batch_prompts_video = lu.make_text_prompts(
    dataset, videos, n_differences, args.eval_mode, args.lmm
)
print("~~~~~~~~~~~ PREDICTION STAGE START ~~~~~~~~~~~")
predictions = lu.run_lmm(
    batch_prompts_text,
    batch_prompts_video,
    args.lmm,
    args.eval_mode,
    n_differences,
    overwrite_cache=False,
    # debug=debug,
    verbose=True,
)
# predictions = json.load(open("lmms/results/qwen2vl_7b_open_easy/seed_0/input_predictions.json", "r"))
print(type(predictions))
print("~~~~~~~~~~~ PREDICTION STAGE COMPLETE ~~~~~~~~~~~")
print(args.seed)
# do eval
metrics = eval_viddiff.eval_viddiff(
    dataset=dataset,
    predictions_unmatched=predictions,
    eval_mode=args.eval_mode,
    n_differences=None,
    seed=1,
    results_dir=args.logging.results_dir,
)
print(metrics)


# batch_prompts_text = pickle.load(open("/data2/skabi9001/batch_prompts_text.pkl", "rb"))
# batch_prompts_video = pickle.load(open("/data2/skabi9001/batch_prompts_video.pkl", "rb"))

# batch_prompts_text=batch_prompts_text[:10]
# batch_prompts_video=batch_prompts_video[:10]

# print(batch_prompts_text[0])
# print(batch_prompts_video)

# seeds = [0] * len(batch_prompts_text)

# msgs, responses = qwen_api.call_qwen_batch(
#             batch_prompts_text,
#             batch_prompts_video,
#             seeds=seeds,
#             model="Qwen/Qwen2-VL-7B-Instruct",
#             debug=None,
#             json_mode=False,
#         )
# print(responses)

# model_dict: dict = qwen_api.get_qwen_model(
#         model_name="Qwen/Qwen2-VL-7B-Instruct",
#         torch_dtype=torch.bfloat16,
#         device="auto",
#     )

# print(model_dict)


#  from HF
'''

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
'''