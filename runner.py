#!/usr/bin/env python3
"""
IndicGenBench Benchmark Runner

This script orchestrates the full benchmark evaluation against multiple LLMs.
It handles data loading, generation, evaluation, and reporting.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Custom modules
from prompt import get_prompt_for_task, TASK_DESCRIPTIONS
from metrics import IndicGenBenchMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indicgenbench_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Language name mappings
LANGUAGE_NAMES = {
    'as': 'Assamese',
    'awa': 'Awadhi',
    'bgc': 'Haryanvi',
    'bho': 'Bhojpuri',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'brx': 'Bodo',
    'en': 'English',
    'gbm': 'Garhwali',
    'gom': 'Konkani',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'hne': 'Chhattisgarhi',
    'hoj': 'Rajasthani',
    'kn': 'Kannada',
    'mai': 'Maithili',
    'ml': 'Malayalam',
    'mni': 'Manipuri',
    'mr': 'Marathi',
    'mup': 'Malvi',
    'mwr': 'Marwari',
    'ne': 'Nepali',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ps': 'Pashto',
    'sa': 'Sanskrit',
    'sat': 'Santali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu'
}

class LLMInterface:
    """Base class for LLM interfaces"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text based on the prompt"""
        raise NotImplementedError("Subclasses must implement this method")
    
    @property
    def name(self) -> str:
        return self.model_name

class HuggingFaceLLM(LLMInterface):
    """Interface for HuggingFace models with Unsloth optimization"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name)
        try:
            # Try to import Unsloth for faster inference
            from unsloth import FastLanguageModel
            
            logger.info(f"Loading model with Unsloth optimization: {model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,  # Adjust based on your needs
                load_in_4bit=True,    # Use 4-bit quantization for memory efficiency
            )
            # Enable faster inference
            FastLanguageModel.for_inference(self.model)
            self.using_unsloth = True
        except ImportError:
            # Fall back to standard HuggingFace if Unsloth is not available
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Unsloth not available, falling back to standard loading: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto")
            self.using_unsloth = False
            
        self.device = device
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Unsloth-optimized or standard HuggingFace model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.using_unsloth:
                # Unsloth generation
                if self.device != "cpu" and hasattr(inputs, "to"):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True
                )
            else:
                # Standard HuggingFace generation
                inputs = inputs.to(self.device)
                output = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            # Return an empty string in case of error
            return ""

class OpenAILLM(LLMInterface):
    """Interface for OpenAI API models"""
    
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name)
        import openai
        
        # Set API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key not provided and not found in environment")
                
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

class AnthropicLLM(LLMInterface):
    """Interface for Anthropic API models"""
    
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name)
        import anthropic
        
        # Set API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("Anthropic API key not provided and not found in environment")
                
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Anthropic API"""
        response = self.client.completions.create(
            model=self.model_name,
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            max_tokens_to_sample=max_tokens,
            temperature=0.7
        )
        return response.completion.strip()

def create_llm_interface(model_config: Dict[str, Any]) -> LLMInterface:
    """Create an LLM interface based on the model config"""
    model_type = model_config.get("type", "").lower()
    model_name = model_config.get("name", "")
    
    if model_type == "huggingface":
        return HuggingFaceLLM(
            model_name=model_name,
            device=model_config.get("device", "cuda")
        )
    elif model_type == "openai":
        return OpenAILLM(
            model_name=model_name,
            api_key=model_config.get("api_key")
        )
    elif model_type == "anthropic":
        return AnthropicLLM(
            model_name=model_name,
            api_key=model_config.get("api_key")
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class IndicGenBenchEvaluator:
    """Main evaluator class for IndicGenBench"""
    
    def __init__(
        self, 
        data_dir: str, 
        model_configs: List[Dict[str, Any]],
        languages: List[str] = None,
        tasks: List[str] = None,
        splits: List[str] = None,
        sample_size: int = None,
        output_dir: str = "./results"
    ):
        self.data_dir = data_dir
        self.model_configs = model_configs
        self.languages = languages or self._discover_languages()
        self.tasks = tasks or ["crosssum_in", "flores_in", "xquad_in", "xorqa_in"]
        self.splits = splits or ["dev", "test"]
        self.sample_size = sample_size
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models
        self.models = [create_llm_interface(config) for config in model_configs]
        
        logger.info(f"Initialized evaluator with {len(self.models)} models")
        logger.info(f"Tasks: {self.tasks}")
        logger.info(f"Languages: {self.languages}")
        logger.info(f"Splits: {self.splits}")
    
    def _discover_languages(self) -> List[str]:
        """Discover all available languages in the dataset"""
        languages = set()
        for task in ["crosssum_in", "flores_in", "xquad_in", "xorqa_in"]:
            task_dir = os.path.join(self.data_dir, task)
            if not os.path.exists(task_dir):
                continue
                
            for filename in os.listdir(task_dir):
                if not filename.endswith(".json"):
                    continue
                    
                parts = filename.replace(".json", "").split("_")
                if task == "flores_in":
                    # Extract languages from flores format
                    if len(parts) >= 4:
                        languages.add(parts[1])
                        languages.add(parts[2])
                else:
                    # Extract language from other task formats
                    if len(parts) >= 3:
                        languages.add(parts[1])
        
        # Remove English if it's in the set since we're focusing on Indic languages
        if 'en' in languages:
            languages.remove('en')
            
        return sorted(list(languages))
    
    def _load_data(self, task: str, language: str, split: str) -> List[Dict[str, Any]]:
        """Load data for a specific task, language and split"""
        filepath = None
        
        if task == "crosssum_in":
            filepath = os.path.join(self.data_dir, task, f"crosssum_english-{language}_{split}.json")
        elif task == "flores_in":
            # Try both en-to-lang and lang-to-en
            filepath_en_to_lang = os.path.join(self.data_dir, task, f"flores_en_{language}_{split}.json")
            filepath_lang_to_en = os.path.join(self.data_dir, task, f"flores_{language}_en_{split}.json")
            
            if os.path.exists(filepath_en_to_lang):
                filepath = filepath_en_to_lang
            elif os.path.exists(filepath_lang_to_en):
                filepath = filepath_lang_to_en
        elif task in ["xquad_in", "xorqa_in"]:
            filepath = os.path.join(self.data_dir, task, f"{task[:-3]}_{language}_{split}.json")
        
        if filepath is None or not os.path.exists(filepath):
            logger.warning(f"File not found for {task}-{language}-{split}")
            return []
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            examples = data.get("examples", [])
            
            # Apply sample size if specified
            if self.sample_size is not None and len(examples) > self.sample_size:
                import random
                random.seed(42)  # For reproducibility
                examples = random.sample(examples, self.sample_size)
                
            # Add language code if not already present
            for example in examples:
                if "lang" not in example:
                    example["lang"] = language
                    
                # Add source/target language fields for flores
                if task == "flores_in":
                    if filepath == filepath_en_to_lang:
                        example["src_lang"] = "en"
                        example["tgt_lang"] = language
                    elif filepath == filepath_lang_to_en:
                        example["src_lang"] = language
                        example["tgt_lang"] = "en"
                    
            return examples
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []
    
    def _prepare_evaluation_task(
        self, 
        model: LLMInterface,
        task: str,
        language: str,
        split: str
    ) -> Dict[str, Any]:
        """Prepare an evaluation task for a specific model, task, language and split"""
        data = self._load_data(task, language, split)
        if not data:
            return None
            
        return {
            "model": model,
            "task": task,
            "language": language,
            "split": split,
            "data": data,
            "language_name": LANGUAGE_NAMES.get(language, language)
        }
    
    def _evaluate_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single task configuration"""
        model = task_config["model"]
        task = task_config["task"]
        language = task_config["language"]
        split = task_config["split"]
        data = task_config["data"]
        language_name = task_config["language_name"]
        
        logger.info(f"Evaluating {model.name} on {task}-{language}-{split} ({len(data)} examples)")
        
        predictions = []
        references = []
        
        for example in tqdm(data, desc=f"{model.name} / {task} / {language} / {split}"):
            prompt = get_prompt_for_task(task, example, language_name)
            
            if not prompt:
                logger.warning(f"Failed to generate prompt for example in {task}-{language}-{split}")
                continue
                
            # Get reference output(s)
            reference = self._get_reference(task, example)
            
            if not reference:
                logger.warning(f"Missing reference in {task}-{language}-{split}")
                continue
                
            # Generate prediction
            try:
                prediction = model.generate(prompt)
                predictions.append(prediction)
                references.append(reference)
            except Exception as e:
                logger.error(f"Error generating prediction: {e}")
        
        # Calculate metrics
        if not predictions:
            logger.warning(f"No predictions generated for {task}-{language}-{split}")
            return None
            
        metrics = IndicGenBenchMetrics.evaluate(task, predictions, references)
        
        return {
            "model": model.name,
            "task": task,
            "language": language,
            "language_name": language_name,
            "split": split,
            "metrics": metrics,
            "num_examples": len(predictions)
        }
    
    def _get_reference(self, task: str, example: Dict[str, Any]) -> Any:
        """Get the reference (gold) output for a specific task and example"""
        if task == "crosssum_in":
            return example.get('summary', '')
        
        elif task == "flores_in":
            return example.get('target', '')
        
        elif task == "xquad_in":
            if "answers" in example and isinstance(example["answers"], list):
                # Handle the standard xquad format with answers as a list
                answers = [ans["text"] for ans in example["answers"] if "text" in ans]
                return answers[0] if answers else ""
            else:
                # Fallback to generic answer field
                return example.get('answer', '')
            
        elif task == "xorqa_in":
            if "translated_answers" in example and isinstance(example["translated_answers"], list):
                # Handle translated answers for xorqa
                answers = [ans["text"] for ans in example["translated_answers"] if "text" in ans]
                return answers[0] if answers else ""
            elif "answers" in example:
                if isinstance(example["answers"], list):
                    # Handle answers as a list of objects
                    answers = [ans["text"] for ans in example["answers"] if "text" in ans]
                    return answers[0] if answers else ""
                elif isinstance(example["answers"], str):
                    # Handle answers as a string
                    return example["answers"]
            
            # Fallback to generic answer field
            return example.get('answer', '')
            
        return ""
    
    def evaluate(self, num_workers: int = 1) -> Dict[str, Any]:
        """Run the full evaluation"""
        all_tasks = []
        
        # Prepare all evaluation tasks
        for model in self.models:
            for task in self.tasks:
                for language in self.languages:
                    for split in self.splits:
                        task_config = self._prepare_evaluation_task(model, task, language, split)
                        if task_config:
                            all_tasks.append(task_config)
        
        logger.info(f"Prepared {len(all_tasks)} evaluation tasks")
        
        # Run evaluation tasks
        results = []
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._evaluate_task, task) for task in all_tasks]
                for future in tqdm(as_completed(futures), desc="Evaluating", total=len(futures)):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for task in tqdm(all_tasks, desc="Evaluating"):
                result = self._evaluate_task(task)
                if result:
                    results.append(result)
        
        # Process results
        processed_results = self._process_results(results)
        
        # Generate reports
        self._generate_reports(processed_results)
        
        return processed_results
    
    def _process_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw evaluation results into a structured format"""
        processed = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "models": [model.name for model in self.models],
            "tasks": self.tasks,
            "languages": self.languages,
            "splits": self.splits
        }
        
        # Save raw results
        with open(os.path.join(self.output_dir, "results_raw.json"), 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2)
            
        return processed
    
    def _generate_reports(self, processed_results: Dict[str, Any]):
        """Generate evaluation reports"""
        self._generate_summary_report(processed_results)
        self._generate_detailed_reports(processed_results)
        self._generate_visualizations(processed_results)
    
    def _generate_summary_report(self, processed_results: Dict[str, Any]):
        """Generate a summary report of the evaluation"""
        results = processed_results["results"]
        models = processed_results["models"]
        tasks = processed_results["tasks"]
        
        report = []
        report.append("# IndicGenBench Evaluation Summary")
        report.append("")
        report.append(f"Evaluation timestamp: {processed_results['timestamp']}")
        report.append("")
        report.append("## Models Evaluated")
        report.append("")
        for i, model in enumerate(models, 1):
            report.append(f"{i}. {model}")
        report.append("")
        
        # Task-wise summary
        for task in tasks:
            report.append(f"## {task}")
            report.append("")
            
            # Get primary metric for this task
            primary_metric = None
            if task == "crosssum_in":
                primary_metric = "rougeL"
            elif task == "flores_in":
                primary_metric = "bleu"
            elif task in ["xquad_in", "xorqa_in"]:
                primary_metric = "f1"
                
            report.append(f"Primary metric: {primary_metric}")
            report.append("")
            
            # Create table header
            table_header = "| Model | Language | Split | Score | Other Metrics |"
            table_divider = "| --- | --- | --- | --- | --- |"
            report.append(table_header)
            report.append(table_divider)
            
            # Filter results for this task
            task_results = [r for r in results if r["task"] == task]
            
            # Sort by model, language, split
            task_results.sort(key=lambda r: (r["model"], r["language"], r["split"]))
            
            for result in task_results:
                model = result["model"]
                language = result["language"]
                language_name = result["language_name"]
                split = result["split"]
                metrics = result["metrics"]
                
                if primary_metric in metrics:
                    primary_score = f"{metrics[primary_metric]:.2f}"
                    
                    # Create a string of other metrics
                    other_metrics = ", ".join([
                        f"{k}: {v:.2f}" 
                        for k, v in metrics.items() 
                        if k != primary_metric
                    ])
                    
                    row = f"| {model} | {language_name} ({language}) | {split} | {primary_score} | {other_metrics} |"
                    report.append(row)
            
            report.append("")
        
        # Write the report
        with open(os.path.join(self.output_dir, "summary_report.md"), 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
    
    def _generate_detailed_reports(self, processed_results: Dict[str, Any]):
        """Generate detailed reports for each model"""
        results = processed_results["results"]
        
        # Group results by model
        model_results = {}
        for result in results:
            model = result["model"]
            if model not in model_results:
                model_results[model] = []
                
            model_results[model].append(result)
            
        # Generate report for each model
        for model, model_data in model_results.items():
            report = []
            report.append(f"# IndicGenBench Detailed Report: {model}")
            report.append("")
            report.append(f"Evaluation timestamp: {processed_results['timestamp']}")
            report.append("")
            
            # Group by task
            task_results = {}
            for result in model_data:
                task = result["task"]
                if task not in task_results:
                    task_results[task] = []
                    
                task_results[task].append(result)
                
            # Generate report for each task
            for task, task_data in task_results.items():
                report.append(f"## {task}")
                report.append("")
                
                # Format the task description appropriately based on the task
                if task == "flores_in":
                    task_description = TASK_DESCRIPTIONS[task].format(
                        source_language='[source language]',
                        target_language='[target language]'
                    )
                else:
                    task_description = TASK_DESCRIPTIONS[task].format(language='[language]')
                
                report.append(f"Task description: {task_description}")
                report.append("")
                
                # Get primary metric for this task
                primary_metrics = []
                if task == "crosssum_in":
                    primary_metrics = ["rougeL", "rouge1", "rouge2"]
                elif task == "flores_in":
                    primary_metrics = ["bleu", "chrf", "meteor"]
                elif task in ["xquad_in", "xorqa_in"]:
                    primary_metrics = ["f1", "exact_match"]
                    
                # Create table header
                header = "| Language | Split | Samples |"
                divider = "| --- | --- | --- |"
                
                for metric in primary_metrics:
                    header += f" {metric} |"
                    divider += " --- |"
                    
                report.append(header)
                report.append(divider)
                
                # Sort by language, split
                task_data.sort(key=lambda r: (r["language"], r["split"]))
                
                for result in task_data:
                    language = result["language"]
                    language_name = result["language_name"]
                    split = result["split"]
                    metrics = result["metrics"]
                    num_examples = result["num_examples"]
                    
                    row = f"| {language_name} ({language}) | {split} | {num_examples} |"
                    
                    for metric in primary_metrics:
                        if metric in metrics:
                            row += f" {metrics[metric]:.2f} |"
                        else:
                            row += " N/A |"
                            
                    report.append(row)
                
                report.append("")
            
            # Write the report
            filename = f"{model.replace('/', '_')}_detailed_report.md"
            with open(os.path.join(self.output_dir, filename), 'w', encoding='utf-8') as f:
                f.write("\n".join(report))
    
    def _generate_visualizations(self, processed_results: Dict[str, Any]):
        """Generate visualizations of the evaluation results"""
        results = processed_results["results"]
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert results to DataFrame for easier manipulation
        data = []
        for result in results:
            model = result["model"]
            task = result["task"]
            language = result["language"]
            language_name = result["language_name"]
            split = result["split"]
            metrics = result["metrics"]
            
            row = {
                "Model": model,
                "Task": task,
                "Language": language,
                "Language_Name": language_name,
                "Split": split
            }
            
            # Add metrics
            for metric, value in metrics.items():
                row[metric] = value
                
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Save the DataFrame
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        # Generate task-specific visualizations
        for task in processed_results["tasks"]:
            task_df = df[df["Task"] == task]
            if task_df.empty:
                continue
                
            # Get primary metric for this task
            primary_metric = None
            if task == "crosssum_in":
                primary_metric = "rougeL"
            elif task == "flores_in":
                primary_metric = "bleu"
            elif task in ["xquad_in", "xorqa_in"]:
                primary_metric = "f1"
                
            if primary_metric not in task_df.columns:
                continue
                
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Use dev split if available, otherwise test
            if "dev" in task_df["Split"].values:
                plot_df = task_df[task_df["Split"] == "dev"]
            else:
                plot_df = task_df[task_df["Split"] == "test"]
                
            # Create a pivot table for the heatmap
            pivot = plot_df.pivot_table(
                values=primary_metric,
                index="Language_Name",
                columns="Model"
            )
            
            # Sort languages by average score
            language_order = pivot.mean(axis=1).sort_values(ascending=False).index
            pivot = pivot.reindex(language_order)
            
            # Create heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                linewidths=0.5
            )
            
            plt.title(f"{task} - {primary_metric} by Language and Model")
            plt.ylabel("Language")
            plt.xlabel("Model")
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, f"{task}_{primary_metric}_heatmap.png"), dpi=300)
            plt.close()
            
            # Create language comparison bar plot
            plt.figure(figsize=(14, 8))
            
            sns.barplot(
                data=plot_df,
                x="Language_Name",
                y=primary_metric,
                hue="Model"
            )
            
            plt.title(f"{task} - {primary_metric} by Language and Model")
            plt.ylabel(primary_metric)
            plt.xlabel("Language")
            plt.xticks(rotation=90)
            plt.legend(loc="best")
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(viz_dir, f"{task}_{primary_metric}_barplot.png"), dpi=300)
            plt.close()
            
        # Generate model comparison visualizations
        models = processed_results["models"]
        if len(models) > 1:
            # Create a model comparison plot across all tasks
            plt.figure(figsize=(12, 8))
            
            # Use dev split if available, otherwise test
            if "dev" in df["Split"].values:
                plot_df = df[df["Split"] == "dev"]
            else:
                plot_df = df[df["Split"] == "test"]
                
            # Calculate average scores for each task and model
            task_metrics = {
                "crosssum_in": "rougeL",
                "flores_in": "bleu",
                "xquad_in": "f1",
                "xorqa_in": "f1"
            }
            
            data = []
            for task, metric in task_metrics.items():
                task_data = plot_df[plot_df["Task"] == task]
                if task_data.empty or metric not in task_data.columns:
                    continue
                    
                for model in models:
                    model_data = task_data[task_data["Model"] == model]
                    if model_data.empty:
                        continue
                        
                    avg_score = model_data[metric].mean()
                    data.append({
                        "Model": model,
                        "Task": task,
                        "Score": avg_score,
                        "Metric": metric
                    })
            
            if data:
                comparison_df = pd.DataFrame(data)
                
                sns.barplot(
                    data=comparison_df,
                    x="Task",
                    y="Score",
                    hue="Model"
                )
                
                plt.title("Model Performance Comparison Across Tasks")
                plt.ylabel("Average Score")
                plt.xlabel("Task")
                plt.legend(loc="best")
                plt.tight_layout()
                
                # Save plot
                plt.savefig(os.path.join(viz_dir, "model_comparison.png"), dpi=300)
                plt.close()
                
                # Save comparison data
                comparison_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on IndicGenBench")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to IndicGenBench data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config JSON")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    evaluator = IndicGenBenchEvaluator(
        data_dir=args.data_dir,
        model_configs=config.get("models", []),
        languages=config.get("languages"),
        tasks=config.get("tasks"),
        splits=config.get("splits"),
        sample_size=config.get("sample_size"),
        output_dir=args.output_dir
    )
    
    evaluator.evaluate(num_workers=args.num_workers)

if __name__ == "__main__":
    main()