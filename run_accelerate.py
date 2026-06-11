import json, os, torch, time, logging, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime
from metrics import IndicGenBenchMetrics
from prompt import get_prompt_for_task
from runner import IndicGenBenchEvaluator

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
TASK_PRIMARY_METRIC = {"crosssum_in": "rougeL", "flores_in": "bleu", "xquad_in": "f1", "xorqa_in": "f1"}

accelerator = Accelerator()
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/samples", exist_ok=True)

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


def get_ref(task, ex):
    if task == "crosssum_in":
        return ex.get("summary", "")
    elif task == "flores_in":
        return ex.get("target", "")
    elif task == "xquad_in":
        return ex["answers"][0]["text"] if ex.get("answers") else ""
    elif task == "xorqa_in":
        return ex["translated_answers"][0]["text"] if ex.get("translated_answers") else ""
    return ""


def save_lm_eval_samples(task, lang, split, examples, prompts, preds, refs, metrics):
    samples = []
    for i, (ex, prompt, pred, ref) in enumerate(zip(examples, prompts, preds, refs)):
        samples.append({
            "doc_id": i,
            "doc": ex,
            "target": ref,
            "arguments": [prompt],
            "resps": [[pred]],
            "filtered_resps": [pred],
            "metadata": {"task": task, "language": lang, "split": split, "model": MODEL}
        })
    out = {
        "model": MODEL,
        "model_args": "trust_remote_code=True,dtype=bfloat16",
        "config": {
            "task": task, "language": lang, "split": split,
            "num_fewshot": 0, "batch_size": BATCH_SIZE,
            "device": str(accelerator.device), "limit": SAMPLE_SIZE,
            "gen_kwargs": "max_new_tokens=256,do_sample=False"
        },
        "results": {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()},
        "samples": samples,
        "n_samples": len(samples),
        "higher_is_better": {k: True for k in metrics},
        "timestamp": datetime.now().isoformat()
    }
    fname = f"{OUTPUT_DIR}/samples/{task}_{lang}_{split}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"  saved samples → {fname}")


# ── Main eval loop ──────────────────────────────────────────────────────────
all_results = []
total_start = time.time()

for task in TASKS:
    for lang in LANGUAGES:
        split = "dev"
        if task == "crosssum_in":
            fpath = f"{DATA_DIR}/{task}/crosssum_english-{lang}_{split}.json"
        elif task == "flores_in":
            fpath = f"{DATA_DIR}/{task}/flores_en_{lang}_{split}.json"
        elif task == "xquad_in":
            fpath = f"{DATA_DIR}/{task}/xquad_{lang}_{split}.json"
        elif task == "xorqa_in":
            fpath = f"{DATA_DIR}/{task}/xorqa_{lang}_{split}.json"

        if not os.path.exists(fpath):
            if accelerator.is_main_process:
                log.warning(f"MISSING: {fpath}")
            continue

        random.seed(42)
        all_examples = json.load(open(fpath))["examples"]
        examples = random.sample(all_examples, min(SAMPLE_SIZE, len(all_examples)))
        lang_name = LANGUAGE_NAMES[lang]

        prompts, refs, valid_examples = [], [], []
        for ex in examples:
            p = get_prompt_for_task(task, ex, lang_name)
            ref = get_ref(task, ex)
            if p and ref:
                prompts.append(p)
                refs.append(ref)
                valid_examples.append(ex)

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

            save_lm_eval_samples(task, lang, split, valid_examples, prompts, preds, refs, metrics)

            all_results.append({
                "model": MODEL, "task": task, "language": lang,
                "language_name": lang_name, "split": split,
                "metrics": metrics, "num_examples": len(preds),
                "elapsed_s": round(elapsed, 2)
            })

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    total_elapsed = time.time() - total_start
    log.info(f"All done in {total_elapsed:.1f}s")

    # Save raw results
    processed = {
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
        "models": [MODEL],
        "tasks": TASKS,
        "languages": LANGUAGES,
        "splits": ["dev"]
    }
    with open(f"{OUTPUT_DIR}/results_raw.json", "w") as f:
        json.dump(processed, f, indent=2, default=str)
    log.info("Saved results_raw.json")

    # Use runner.py's existing visualization + report methods
    evaluator = IndicGenBenchEvaluator.__new__(IndicGenBenchEvaluator)
    evaluator.output_dir = OUTPUT_DIR
    evaluator.models = []
    evaluator._generate_reports(processed)
    log.info(f"Reports and visualizations saved to {OUTPUT_DIR}/")
