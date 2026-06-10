import json, os, torch, time, logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
from metrics import IndicGenBenchMetrics
from prompt import get_prompt_for_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/workspace/IndicGenBench/run_accelerate.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

MODEL = "/fsxnew/post-training-data/vijay/converted_17b/checkpoint-3000"
DATA_DIR = "./data"
OUTPUT_DIR = "./results_accelerate"
LANGUAGES = ["hi", "bn", "ta", "te", "mr"]
TASKS = ["crosssum_in", "flores_in", "xquad_in", "xorqa_in"]
SAMPLE_SIZE = 30
BATCH_SIZE = 32
LANGUAGE_NAMES = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi"}

accelerator = Accelerator()
os.makedirs(OUTPUT_DIR, exist_ok=True)

if accelerator.is_main_process:
    log.info(f"Using {accelerator.num_processes} GPUs")
    log.info("Loading model...")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = accelerator.prepare(model)
model = accelerator.unwrap_model(model)
model.eval()

if accelerator.is_main_process:
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

def batch_generate(prompts, batch_size=BATCH_SIZE, desc="generating"):
    all_preds = []
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    for batch in tqdm(batches, desc=desc, unit="batch", disable=not accelerator.is_main_process):
        t = time.time()
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=1024, return_token_type_ids=False)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for o in out:
            decoded = tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            all_preds.append(decoded.strip())
        if accelerator.is_main_process:
            log.info(f"  batch done | {len(batch)} examples | {time.time()-t:.2f}s")
    return all_preds

all_results = []
total_start = time.time()

for task in TASKS:
    for lang in LANGUAGES:
        if task == "crosssum_in":
            fpath = f"{DATA_DIR}/{task}/crosssum_english-{lang}_dev.json"
        elif task == "flores_in":
            fpath = f"{DATA_DIR}/{task}/flores_en_{lang}_dev.json"
        elif task == "xquad_in":
            fpath = f"{DATA_DIR}/{task}/xquad_{lang}_dev.json"
        elif task == "xorqa_in":
            fpath = f"{DATA_DIR}/{task}/xorqa_{lang}_dev.json"

        if not os.path.exists(fpath):
            if accelerator.is_main_process:
                log.warning(f"MISSING: {fpath}")
            continue

        data = json.load(open(fpath))["examples"][:SAMPLE_SIZE]
        lang_name = LANGUAGE_NAMES[lang]

        prompts, refs = [], []
        for ex in data:
            p = get_prompt_for_task(task, ex, lang_name)
            if not p:
                continue
            if task == "crosssum_in":
                ref = ex.get("summary", "")
            elif task == "flores_in":
                ref = ex.get("target", "")
            elif task == "xquad_in":
                ref = ex["answers"][0]["text"] if ex.get("answers") else ""
            elif task == "xorqa_in":
                ref = ex["translated_answers"][0]["text"] if ex.get("translated_answers") else ""
            if ref:
                prompts.append(p)
                refs.append(ref)

        if not prompts:
            continue

        if accelerator.is_main_process:
            log.info(f"Starting {task}-{lang} | {len(prompts)} examples")
        task_start = time.time()
        preds = batch_generate(prompts, desc=f"{task}-{lang}")
        elapsed = time.time() - task_start

        if accelerator.is_main_process:
            metrics = IndicGenBenchMetrics.evaluate(task, preds, refs)
            log.info(f"Done {task}-{lang} | {elapsed:.1f}s | {elapsed/len(prompts):.2f}s/example | metrics: {metrics}")
            all_results.append({"task": task, "language": lang, "metrics": metrics,
                                 "num_examples": len(preds), "elapsed_s": round(elapsed, 2)})

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    total_elapsed = time.time() - total_start
    log.info(f"All done in {total_elapsed:.1f}s")
    with open(f"{OUTPUT_DIR}/results_raw.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {OUTPUT_DIR}/results_raw.json")
