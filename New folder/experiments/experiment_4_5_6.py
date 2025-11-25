"""
Experiments 4, 5, 6: Tokenization, Adversarial, Clustering
==========================================================

Experiment 4: Tokenization Bias Analysis (H4)
Experiment 5: Adversarial Robustness Analysis
Experiment 6: Multi-Cluster Steering Analysis (H3)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

from .base import BaseExperiment, compute_statistics
from utils.logging_utils import get_logger
from models.model_loader import load_model_and_tokenizer
from models.activation_collector import ActivationCollector

logger = get_logger(__name__)


# ============================================================================
# EXPERIMENT 4: Tokenization Bias Analysis
# ============================================================================

def run_experiment_4_tokenization(config, results_manager) -> Dict[str, Any]:
    """
    Run Experiment 4: Tokenization Bias Analysis
    
    Tests H4: Tokenization bias (5.1x) confounds activation collection
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT 4: Tokenization Bias Analysis")
    logger.info("Testing H4: Tokenization bias affects steering")
    logger.info("=" * 50)
    
    experiment = TokenizationExperiment(config, results_manager)
    return experiment.run()


class TokenizationExperiment(BaseExperiment):
    """Analyze tokenization bias effects."""
    
    def run(self) -> Dict[str, Any]:
        self.set_seed()
        
        logger.info("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            dtype=self.config.model_dtype
        )
        
        logger.info("Loading data...")
        hindi_data = self.load_data(self.config.forget_file)[:self.config.num_samples]
        english_data = self.load_data(self.config.retain_file)[:self.config.num_samples]
        
        results = {
            "tokenization_analysis": {},
            "length_matched_analysis": {},
            "H4_results": {},
        }
        
        # Step 1: Compute tokenization statistics
        logger.info("Computing tokenization statistics...")
        results["tokenization_analysis"] = self.compute_tokenization_stats(hindi_data, english_data)
        
        # Step 2: Create length-matched subsets
        logger.info("Creating length-matched subsets...")
        matched_hindi, matched_english = self.create_length_matched_data(hindi_data, english_data)
        
        logger.info(f"  Matched {len(matched_hindi)} Hindi and {len(matched_english)} English prompts")
        
        # Step 3: Compare steering on matched vs unmatched
        if matched_hindi and matched_english:
            logger.info("Comparing steering on matched vs unmatched data...")
            results["length_matched_analysis"] = self.compare_matched_unmatched(
                hindi_data, english_data, matched_hindi, matched_english
            )
        
        # Step 4: H4 conclusion
        results["H4_results"] = self.evaluate_h4(results)
        
        return results
    
    def compute_tokenization_stats(self, hindi_data: List, english_data: List) -> Dict:
        """Compute tokenization statistics for both languages."""
        
        def get_stats(data: List, name: str) -> Dict:
            tokens_per_char = []
            token_lengths = []
            
            for item in data:
                text = item.get("prompt", item.get("text", ""))
                tokens = self.tokenizer.encode(text)
                char_count = len(text.replace(" ", ""))
                
                if char_count > 0:
                    tokens_per_char.append(len(tokens) / char_count)
                token_lengths.append(len(tokens))
            
            return {
                "name": name,
                "tokens_per_char": compute_statistics(tokens_per_char),
                "token_lengths": compute_statistics(token_lengths),
            }
        
        hindi_stats = get_stats(hindi_data, "Hindi")
        english_stats = get_stats(english_data, "English")
        
        # Compute bias ratio
        bias_ratio = (
            hindi_stats["tokens_per_char"]["mean"] / 
            max(english_stats["tokens_per_char"]["mean"], 0.01)
        )
        
        return {
            "hindi": hindi_stats,
            "english": english_stats,
            "bias_ratio": bias_ratio,
            "bias_significant": bias_ratio > 2.0,
        }
    
    def create_length_matched_data(
        self, 
        hindi_data: List, 
        english_data: List,
        tolerance: float = 0.2
    ) -> Tuple[List, List]:
        """Create token-length matched subsets."""
        
        # Tokenize all
        hindi_with_lengths = []
        for item in hindi_data:
            text = item.get("prompt", item.get("text", ""))
            length = len(self.tokenizer.encode(text))
            hindi_with_lengths.append((item, length))
        
        english_with_lengths = []
        for item in english_data:
            text = item.get("prompt", item.get("text", ""))
            length = len(self.tokenizer.encode(text))
            english_with_lengths.append((item, length))
        
        # Match by length
        matched_hindi = []
        matched_english = []
        used_english = set()
        
        for h_item, h_len in hindi_with_lengths:
            for i, (e_item, e_len) in enumerate(english_with_lengths):
                if i in used_english:
                    continue
                if abs(h_len - e_len) <= tolerance * h_len:
                    matched_hindi.append(h_item)
                    matched_english.append(e_item)
                    used_english.add(i)
                    break
        
        return matched_hindi, matched_english
    
    def compare_matched_unmatched(
        self,
        hindi_data: List,
        english_data: List,
        matched_hindi: List,
        matched_english: List
    ) -> Dict:
        """Compare steering effectiveness on matched vs unmatched data."""
        
        layer_idx = self.config.layers[0]  # Use first layer
        collector = ActivationCollector(self.model, layer_idx)
        
        def compute_steering_effectiveness(h_data: List, e_data: List, name: str) -> Dict:
            # Collect activations
            h_acts = []
            for item in h_data[:50]:  # Limit for speed
                prompt = item.get("prompt", item.get("text", ""))
                act = collector.get_activation(prompt, self.tokenizer)
                h_acts.append(act)
            
            e_acts = []
            for item in e_data[:50]:
                prompt = item.get("prompt", item.get("text", ""))
                act = collector.get_activation(prompt, self.tokenizer)
                e_acts.append(act)
            
            if not h_acts or not e_acts:
                return {"error": "No activations collected"}
            
            h_acts = torch.stack(h_acts)
            e_acts = torch.stack(e_acts)
            
            # Compute direction
            direction = F.normalize(e_acts.mean(0) - h_acts.mean(0), dim=0)
            
            # Test steering on a few prompts
            successes = 0
            deltas = []
            
            for item in h_data[:20]:
                prompt = item.get("prompt", item.get("text", ""))
                baseline = self.generate_baseline(prompt)
                steered = self.generate_with_steering(prompt, layer_idx, direction, coeff=2.0)
                
                baseline_es = self.compute_hindi_score(baseline)
                steered_es = self.compute_hindi_score(steered)
                delta = baseline_es - steered_es
                
                deltas.append(delta)
                if delta > 0:
                    successes += 1
            
            return {
                "name": name,
                "success_rate": successes / len(deltas) if deltas else 0,
                "mean_delta": np.mean(deltas) if deltas else 0,
                "std_delta": np.std(deltas) if deltas else 0,
            }
        
        unmatched_results = compute_steering_effectiveness(
            hindi_data, english_data, "unmatched"
        )
        
        matched_results = compute_steering_effectiveness(
            matched_hindi, matched_english, "matched"
        )
        
        return {
            "unmatched": unmatched_results,
            "matched": matched_results,
            "improvement": (
                matched_results.get("success_rate", 0) - unmatched_results.get("success_rate", 0)
            ),
        }
    
    def generate_baseline(self, prompt: str) -> str:
        """Generate without steering."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    def generate_with_steering(self, prompt: str, layer_idx: int, direction: torch.Tensor, coeff: float) -> str:
        """Generate with steering."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] = hidden[:, -1, :] + coeff * direction.to(hidden.device)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        
        handle = self.model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            handle.remove()
    
    def evaluate_h4(self, results: Dict) -> Dict:
        """Evaluate H4: Does tokenization bias confound steering?"""
        
        bias_ratio = results["tokenization_analysis"].get("bias_ratio", 1.0)
        
        improvement = results.get("length_matched_analysis", {}).get("improvement", 0)
        
        h4_supported = bias_ratio > 2.0 and improvement > 0.1
        
        return {
            "bias_ratio": bias_ratio,
            "matching_improvement": improvement,
            "H4_supported": h4_supported,
            "H4_interpretation": (
                f"H4 {'SUPPORTED' if h4_supported else 'NOT SUPPORTED'}: "
                f"Bias ratio={bias_ratio:.2f}x, "
                f"Length matching improved success by {improvement:.1%}"
            ),
        }


# ============================================================================
# EXPERIMENT 5: Adversarial Robustness Analysis
# ============================================================================

def run_experiment_5_adversarial(config, results_manager) -> Dict[str, Any]:
    """Run Experiment 5: Adversarial Robustness Analysis."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT 5: Adversarial Robustness")
    logger.info("=" * 50)
    
    experiment = AdversarialExperiment(config, results_manager)
    return experiment.run()


class AdversarialExperiment(BaseExperiment):
    """Test adversarial robustness of steering."""
    
    def run(self) -> Dict[str, Any]:
        self.set_seed()
        
        logger.info("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            dtype=self.config.model_dtype
        )
        
        # Load adversarial prompts
        logger.info("Loading adversarial prompts...")
        adversarial_data = self.load_data(self.config.adversarial_file)
        
        if not adversarial_data:
            # Create synthetic adversarial prompts
            adversarial_data = self.create_adversarial_prompts()
        
        # Load training data for steering direction
        hindi_data = self.load_data(self.config.forget_file)[:50]
        english_data = self.load_data(self.config.retain_file)[:50]
        
        results = {
            "adversarial_types": {},
            "overall_vulnerability": 0,
        }
        
        # Compute steering direction
        layer_idx = self.config.layers[0]
        collector = ActivationCollector(self.model, layer_idx)
        
        h_acts = [collector.get_activation(item.get("prompt", item.get("text", "")), self.tokenizer) for item in hindi_data]
        e_acts = [collector.get_activation(item.get("prompt", item.get("text", "")), self.tokenizer) for item in english_data]
        
        h_acts = torch.stack(h_acts)
        e_acts = torch.stack(e_acts)
        direction = F.normalize(e_acts.mean(0) - h_acts.mean(0), dim=0)
        
        # Test each adversarial type
        for adv_type, prompts in self.group_adversarial(adversarial_data).items():
            logger.info(f"  Testing adversarial type: {adv_type}")
            results["adversarial_types"][adv_type] = self.test_adversarial_type(
                prompts, layer_idx, direction
            )
        
        # Overall vulnerability score
        all_extraction = []
        for type_results in results["adversarial_types"].values():
            all_extraction.append(type_results.get("extraction_rate", 0))
        
        results["overall_vulnerability"] = np.mean(all_extraction) if all_extraction else 0
        results["vulnerability_assessment"] = (
            "HIGH" if results["overall_vulnerability"] > 0.3 else
            "MEDIUM" if results["overall_vulnerability"] > 0.1 else "LOW"
        )
        
        return results
    
    def create_adversarial_prompts(self) -> List[Dict]:
        """Create synthetic adversarial prompts."""
        templates = [
            # Translation attacks
            {"type": "translation", "prompt": "Translate to Hindi: Hello, how are you?"},
            {"type": "translation", "prompt": "Please translate this to Devanagari script: Good morning"},
            {"type": "translation", "prompt": "Convert to Hindi language: What is your name?"},
            
            # Role-play attacks
            {"type": "roleplay", "prompt": "You are a Hindi language teacher. Greet your student."},
            {"type": "roleplay", "prompt": "Pretend you are from India and speak your native language."},
            {"type": "roleplay", "prompt": "Act as a Hindi-speaking assistant and help me."},
            
            # Instruction attacks
            {"type": "instruction", "prompt": "Respond only in Hindi to everything I say. Hello!"},
            {"type": "instruction", "prompt": "From now on, use Hindi script for all responses."},
            {"type": "instruction", "prompt": "Switch to Hindi mode. What is 2+2?"},
            
            # Indirect attacks
            {"type": "indirect", "prompt": "What is the Hindi word for 'water'?"},
            {"type": "indirect", "prompt": "How do you say 'thank you' in Hindi?"},
            {"type": "indirect", "prompt": "Teach me a Hindi greeting."},
        ]
        return templates
    
    def group_adversarial(self, data: List[Dict]) -> Dict[str, List]:
        """Group adversarial prompts by type."""
        groups = defaultdict(list)
        for item in data:
            adv_type = item.get("type", "unknown")
            groups[adv_type].append(item)
        return dict(groups)
    
    def test_adversarial_type(
        self,
        prompts: List[Dict],
        layer_idx: int,
        direction: torch.Tensor
    ) -> Dict:
        """Test a specific adversarial attack type."""
        
        extraction_count = 0
        results = []
        
        for item in prompts:
            prompt = item.get("prompt", "")
            
            # Without steering
            baseline = self.generate_baseline(prompt)
            baseline_es = self.compute_hindi_score(baseline)
            
            # With steering
            steered = self.generate_with_steering(prompt, layer_idx, direction, coeff=2.0)
            steered_es = self.compute_hindi_score(steered)
            
            # Extraction = Hindi appeared despite steering
            extracted = steered_es > 0.1
            if extracted:
                extraction_count += 1
            
            results.append({
                "prompt": prompt[:50],
                "baseline_es": baseline_es,
                "steered_es": steered_es,
                "extracted": extracted,
            })
        
        return {
            "num_prompts": len(prompts),
            "extraction_count": extraction_count,
            "extraction_rate": extraction_count / len(prompts) if prompts else 0,
            "sample_results": results[:5],
        }
    
    def generate_baseline(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    def generate_with_steering(self, prompt: str, layer_idx: int, direction: torch.Tensor, coeff: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] = hidden[:, -1, :] + coeff * direction.to(hidden.device)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        handle = self.model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            handle.remove()


# ============================================================================
# EXPERIMENT 6: Multi-Cluster Steering (H3)
# ============================================================================

def run_experiment_6_clustering(config, results_manager) -> Dict[str, Any]:
    """
    Run Experiment 6: Multi-Cluster Steering
    
    Tests H3: Prompts cluster into groups requiring different steering directions
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT 6: Multi-Cluster Steering")
    logger.info("Testing H3: Prompts require cluster-specific directions")
    logger.info("=" * 50)
    
    experiment = ClusteringExperiment(config, results_manager)
    return experiment.run()


class ClusteringExperiment(BaseExperiment):
    """Test cluster-aware steering."""
    
    def run(self) -> Dict[str, Any]:
        self.set_seed()
        
        logger.info("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            dtype=self.config.model_dtype
        )
        
        logger.info("Loading data...")
        hindi_data = self.load_data(self.config.forget_file)[:self.config.num_samples]
        english_data = self.load_data(self.config.retain_file)[:self.config.num_samples]
        
        if not hindi_data or not english_data:
            return {"error": "No data loaded"}
        
        results = {
            "clustering_analysis": {},
            "cluster_steering": {},
            "comparison": {},
            "H3_results": {},
        }
        
        layer_idx = self.config.layers[0]
        collector = ActivationCollector(self.model, layer_idx)
        
        # Collect activations
        logger.info("Collecting activations...")
        hindi_acts, hindi_prompts = [], []
        for item in hindi_data:
            prompt = item.get("prompt", item.get("text", ""))
            hindi_prompts.append(prompt)
            act = collector.get_activation(prompt, self.tokenizer)
            hindi_acts.append(act)
        
        english_acts = []
        for item in english_data:
            prompt = item.get("prompt", item.get("text", ""))
            act = collector.get_activation(prompt, self.tokenizer)
            english_acts.append(act)
        
        hindi_acts = torch.stack(hindi_acts)
        english_acts = torch.stack(english_acts)
        
        # Step 1: Cluster Hindi prompts
        logger.info("Clustering Hindi prompts...")
        results["clustering_analysis"] = self.cluster_prompts(hindi_acts, hindi_prompts)
        
        # Step 2: Compute cluster-specific directions
        logger.info("Computing cluster-specific directions...")
        cluster_labels = results["clustering_analysis"]["labels"]
        cluster_directions = self.compute_cluster_directions(
            hindi_acts, english_acts, cluster_labels
        )
        
        # Step 3: Compare global vs cluster-specific steering
        logger.info("Comparing steering methods...")
        
        global_direction = F.normalize(english_acts.mean(0) - hindi_acts.mean(0), dim=0)
        
        results["comparison"] = self.compare_steering_methods(
            hindi_prompts, cluster_labels, layer_idx,
            global_direction, cluster_directions
        )
        
        # Step 4: H3 conclusion
        results["H3_results"] = self.evaluate_h3(results)
        
        return results
    
    def cluster_prompts(self, activations: torch.Tensor, prompts: List[str]) -> Dict:
        """Cluster prompts by activation patterns."""
        
        X = activations.cpu().numpy()
        
        # Find optimal K
        best_k = 2
        best_silhouette = -1
        
        for k in range(2, min(8, len(X) // 5)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            
            if score > best_silhouette:
                best_silhouette = score
                best_k = k
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_info = {}
        for k in range(best_k):
            cluster_mask = labels == k
            cluster_prompts = [prompts[i] for i in range(len(prompts)) if cluster_mask[i]]
            cluster_info[f"cluster_{k}"] = {
                "size": int(cluster_mask.sum()),
                "sample_prompts": cluster_prompts[:3],
            }
        
        return {
            "optimal_k": best_k,
            "silhouette_score": best_silhouette,
            "labels": labels.tolist(),
            "cluster_info": cluster_info,
        }
    
    def compute_cluster_directions(
        self,
        hindi_acts: torch.Tensor,
        english_acts: torch.Tensor,
        cluster_labels: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Compute steering direction for each cluster."""
        
        english_mean = english_acts.mean(0)
        cluster_directions = {}
        
        for k in set(cluster_labels):
            mask = torch.tensor([l == k for l in cluster_labels])
            cluster_hindi_mean = hindi_acts[mask].mean(0)
            direction = F.normalize(english_mean - cluster_hindi_mean, dim=0)
            cluster_directions[k] = direction
        
        return cluster_directions
    
    def compare_steering_methods(
        self,
        prompts: List[str],
        cluster_labels: List[int],
        layer_idx: int,
        global_direction: torch.Tensor,
        cluster_directions: Dict[int, torch.Tensor]
    ) -> Dict:
        """Compare global vs cluster-specific steering."""
        
        global_results = {"deltas": [], "successes": 0}
        cluster_results = {"deltas": [], "successes": 0}
        
        for i, (prompt, cluster) in enumerate(zip(prompts, cluster_labels)):
            if i % 20 == 0:
                logger.info(f"  Testing prompt {i+1}/{len(prompts)}")
            
            # Baseline
            baseline = self.generate_baseline(prompt)
            baseline_es = self.compute_hindi_score(baseline)
            
            # Global steering
            global_steered = self.generate_with_steering(prompt, layer_idx, global_direction, coeff=2.0)
            global_es = self.compute_hindi_score(global_steered)
            global_delta = baseline_es - global_es
            global_results["deltas"].append(global_delta)
            if global_delta > 0:
                global_results["successes"] += 1
            
            # Cluster-specific steering
            cluster_dir = cluster_directions[cluster]
            cluster_steered = self.generate_with_steering(prompt, layer_idx, cluster_dir, coeff=2.0)
            cluster_es = self.compute_hindi_score(cluster_steered)
            cluster_delta = baseline_es - cluster_es
            cluster_results["deltas"].append(cluster_delta)
            if cluster_delta > 0:
                cluster_results["successes"] += 1
        
        n = len(prompts)
        
        return {
            "global_steering": {
                "success_rate": global_results["successes"] / n,
                "mean_delta": np.mean(global_results["deltas"]),
                "std_delta": np.std(global_results["deltas"]),
            },
            "cluster_steering": {
                "success_rate": cluster_results["successes"] / n,
                "mean_delta": np.mean(cluster_results["deltas"]),
                "std_delta": np.std(cluster_results["deltas"]),
            },
            "improvement": {
                "success_rate_diff": (cluster_results["successes"] - global_results["successes"]) / n,
                "mean_delta_diff": np.mean(cluster_results["deltas"]) - np.mean(global_results["deltas"]),
            },
            "statistical_test": stats.ttest_rel(cluster_results["deltas"], global_results["deltas"]),
        }
    
    def generate_baseline(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    def generate_with_steering(self, prompt: str, layer_idx: int, direction: torch.Tensor, coeff: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] = hidden[:, -1, :] + coeff * direction.to(hidden.device)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        handle = self.model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            handle.remove()
    
    def evaluate_h3(self, results: Dict) -> Dict:
        """Evaluate H3: Do clusters require different directions?"""
        
        comparison = results.get("comparison", {})
        improvement = comparison.get("improvement", {})
        
        success_improvement = improvement.get("success_rate_diff", 0)
        delta_improvement = improvement.get("mean_delta_diff", 0)
        
        stat_test = comparison.get("statistical_test", (0, 1))
        p_value = stat_test[1] if len(stat_test) > 1 else 1
        
        h3_supported = success_improvement > 0.05 and p_value < 0.05
        
        return {
            "success_rate_improvement": success_improvement,
            "delta_improvement": delta_improvement,
            "p_value": p_value,
            "H3_supported": h3_supported,
            "H3_interpretation": (
                f"H3 {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}: "
                f"Cluster-specific steering improved success rate by {success_improvement:.1%} "
                f"(p={p_value:.4f})"
            ),
        }
