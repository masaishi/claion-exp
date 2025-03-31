import gc
import glob
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from claion.core.accent_evaluator import AccentEvaluator, EvaluatorConfig
from claion.data.utils import get_root_path
from claion.pipes.sb_sts import SpeechBrainSTSPipeline

# Basic setup
logger = logging.getLogger(__name__)
ROOT_PATH = get_root_path()


@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 1
    learning_rate: float = 0.0001
    max_grad_norm: float = 1.0
    epochs: int = 5
    max_files: Optional[int] = None
    file_pattern: str = "*.wav"
    evaluator_model: str = "openai/whisper-base"
    data_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "speechocean762" / "train" / "audios")
    output_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "outputs")
    cache_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "cache")
    model_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "models")
    use_wandb: bool = True
    project_name: str = "accent-maximization"
    experiment_name: str = field(default_factory=lambda: f"accent_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Create directories"""
        for dir_path in [self.output_dir, self.cache_dir, self.model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class FileProcessingResult:
    """File processing results"""

    file_name: str
    epoch: int
    input_en_score: float
    corrected_en_score: float
    improvement: float
    input_transcript: str
    corrected_transcript: str
    input_accents: dict
    corrected_accents: dict
    loss: Optional[float] = None


@dataclass
class TrainingResult:
    """Training results"""

    epoch_stats: List[dict] = field(default_factory=list)
    file_stats: dict = field(default_factory=dict)
    best_epoch: Optional[int] = None
    best_improvement: float = 0.0


class AccentTrainer:
    """Accent trainer class"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize components
        evaluator_config = EvaluatorConfig(model_path=config.evaluator_model, device=config.device)
        self.evaluator = AccentEvaluator(evaluator_config)
        self.sts_pipeline = SpeechBrainSTSPipeline(device=config.device)
        self.optimizer = torch.optim.Adam(self.sts_pipeline.model.parameters(), lr=config.learning_rate)

        # Enable gradient computation for model parameters
        for param in self.sts_pipeline.model.parameters():
            param.requires_grad = True

        # Setup wandb logging
        if self.config.use_wandb:
            self._setup_wandb()

    def _setup_wandb(self):
        """Setup wandb"""
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            logger.warning("WANDB_API_KEY not found. W&B logging disabled.")
            self.config.use_wandb = False
            return

        wandb.login(key=api_key)
        wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            config={k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
            tags=self.config.tags,
            notes=self.config.notes,
        )

    def _save_model(self, epoch):
        """Save model"""
        try:
            model_path = str(self.config.model_dir / f"sts_model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.sts_pipeline.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                model_path,
            )

            if self.config.use_wandb:
                wandb.log({"model_saved": True, "model_epoch": epoch})

            logger.info(f"Model saved for epoch {epoch}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _cleanup_memory(self):
        """Clean up memory"""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def process_file(self, audio_path, epoch, train_mode=True):
        """Process file"""
        file_name = Path(audio_path).stem

        # Determine input path
        input_path = audio_path if epoch == 0 else self.config.cache_dir / f"{file_name}_epoch{epoch - 1}.wav"

        try:
            # Evaluate input audio
            input_result = self.evaluator(str(input_path), transcribe_audio=True)
            input_en_score = input_result.accents.get("English", 0.0)

            # Set model mode
            self.sts_pipeline.model.train() if train_mode else self.sts_pipeline.model.eval()
            if train_mode:
                self.optimizer.zero_grad()

            # Process audio
            corrected_audio = self.sts_pipeline.generate_speech(input_path)
            if corrected_audio.ndim > 1 and corrected_audio.shape[0] == 1:
                corrected_audio = corrected_audio.squeeze(0)

            # Save to cache
            cache_path = str(self.config.cache_dir / f"{file_name}_epoch{epoch}.wav")
            torchaudio.save(
                cache_path,
                corrected_audio,
                sample_rate=self.sts_pipeline.sampling_rate,
            )

            # Evaluate corrected audio
            corrected_result = self.evaluator(cache_path, transcribe_audio=True)
            corrected_en_score = corrected_result.accents.get("English", 0.0)
            improvement = corrected_en_score - input_en_score

            # Calculate gradient and update in training mode
            if train_mode:
                loss = -improvement
                loss_tensor = torch.tensor(loss, device=self.config.device, requires_grad=True)
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.sts_pipeline.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Log gradient information
                if self.config.use_wandb:
                    total_grad_norm = 0.0
                    for p in self.sts_pipeline.model.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    wandb.log({"grad_norm": total_grad_norm**0.5, "loss": loss})

            # Reset model in evaluation mode
            if not train_mode:
                self.sts_pipeline.reset_model()

            self._cleanup_memory()

            # Return results
            return FileProcessingResult(
                file_name=file_name,
                epoch=epoch,
                input_en_score=input_en_score,
                corrected_en_score=corrected_en_score,
                improvement=improvement,
                input_transcript=input_result.transcript,
                corrected_transcript=corrected_result.transcript,
                input_accents=input_result.accents,
                corrected_accents=corrected_result.accents,
                loss=-improvement if train_mode else None,
            )

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            self._cleanup_memory()
            self.sts_pipeline.reset_model()
            return None

    def evaluate(self, test_audio_files=None, best_epoch=None):
        """Evaluate model"""
        logger.info("Starting evaluation...")

        # Load best model
        if best_epoch is not None:
            model_path = self.config.model_dir / f"sts_model_epoch_{best_epoch}.pt"
            if model_path.exists():
                logger.info(f"Loading best model from epoch {best_epoch}")
                checkpoint = torch.load(str(model_path))
                self.sts_pipeline.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.warning("Best model not found. Using current model.")

        self.sts_pipeline.model.eval()

        # Get test files
        if test_audio_files is None:
            test_audio_files = glob.glob(str(self.config.data_dir / "*.wav"))

        # Process each file
        eval_results = []
        for audio_path in tqdm(test_audio_files, desc="Evaluating"):
            result = self.process_file(audio_path, epoch=0, train_mode=False)
            if result:
                eval_results.append(result)

        # Calculate evaluation statistics
        if not eval_results:
            return {"eval_results": [], "error": "No files processed"}

        stats = {
            "avg_input_score": np.mean([r.input_en_score for r in eval_results]),
            "avg_corrected_score": np.mean([r.corrected_en_score for r in eval_results]),
            "avg_improvement": np.mean([r.improvement for r in eval_results]),
            "num_files_processed": len(eval_results),
            "eval_results": eval_results,
        }

        # Log output
        logger.info(
            f"Evaluation stats: input={stats['avg_input_score']:.4f}, corrected={stats['avg_corrected_score']:.4f}, improvement={stats['avg_improvement']:.4f}"
        )

        # Log to wandb
        if self.config.use_wandb:
            wandb.log(
                {
                    "eval/en_score_mean/input": stats["avg_input_score"],
                    "eval/en_score_mean/corrected": stats["avg_corrected_score"],
                    "eval/en_score_mean/improvement": stats["avg_improvement"],
                }
            )

        return stats

    def train(self):
        """Run training"""
        # Get audio files
        audio_files = glob.glob(str(self.config.data_dir / self.config.file_pattern))
        if self.config.max_files:
            audio_files = audio_files[: self.config.max_files]

        logger.info(f"Training with {len(audio_files)} audio files")

        # Initialize results
        training_result = TrainingResult()

        # Epoch loop
        for epoch in range(self.config.epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{self.config.epochs}")
            epoch_results = []

            # Batch processing
            for i in range(0, len(audio_files), self.config.batch_size):
                batch_files = audio_files[i : i + self.config.batch_size]
                logger.info(f"Processing batch {i // self.config.batch_size + 1}/{(len(audio_files) + self.config.batch_size - 1) // self.config.batch_size}")

                # 各ファイルの処理
                for audio_path in tqdm(batch_files, desc=f"Epoch {epoch + 1}"):
                    result = self.process_file(audio_path, epoch, train_mode=True)
                    if not result:
                        continue

                    epoch_results.append(result)

                    # Update file statistics
                    file_name = result.file_name
                    if file_name not in training_result.file_stats:
                        training_result.file_stats[file_name] = {
                            "original_en_score": result.input_en_score if epoch == 0 else None,
                            "epoch_scores": [],
                            "best_epoch": 0,
                            "best_score": 0,
                        }

                    # Update scores
                    training_result.file_stats[file_name]["epoch_scores"].append(result.corrected_en_score)
                    if result.corrected_en_score > training_result.file_stats[file_name]["best_score"]:
                        training_result.file_stats[file_name]["best_score"] = result.corrected_en_score
                        training_result.file_stats[file_name]["best_epoch"] = epoch

                    # wandbへのログ
                    if self.config.use_wandb:
                        wandb.log(
                            {
                                "file_name": result.file_name,
                                "epoch": epoch,
                                "input_en_score": result.input_en_score,
                                "corrected_en_score": result.corrected_en_score,
                                "improvement": result.improvement,
                                "loss": result.loss if result.loss is not None else 0.0,
                            }
                        )

            # Calculate epoch statistics
            if epoch_results:
                stats = {
                    "epoch": epoch,
                    "avg_input_score": np.mean([r.input_en_score for r in epoch_results]),
                    "avg_corrected_score": np.mean([r.corrected_en_score for r in epoch_results]),
                    "avg_improvement": np.mean([r.improvement for r in epoch_results]),
                    "num_files_processed": len(epoch_results),
                }

                training_result.epoch_stats.append(stats)

                # Update best epoch
                if stats["avg_improvement"] > training_result.best_improvement:
                    training_result.best_improvement = stats["avg_improvement"]
                    training_result.best_epoch = epoch

                # ログ出力
                logger.info(
                    f"Epoch {epoch + 1} stats: input={stats['avg_input_score']:.4f}, "
                    f"corrected={stats['avg_corrected_score']:.4f}, "
                    f"improvement={stats['avg_improvement']:.4f}"
                )

                # wandbへのログ
                if self.config.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "en_score_mean/input": stats["avg_input_score"],
                            "en_score_mean/corrected": stats["avg_corrected_score"],
                            "en_score_mean/improvement": stats["avg_improvement"],
                        }
                    )

            # Save model for each epoch
            self._save_model(epoch)
            self._cleanup_memory()

        # Save final outputs
        self._save_final_outputs(training_result)

        # Finish wandb session
        if self.config.use_wandb:
            wandb.finish()

        return training_result

    def _save_final_outputs(self, training_result):
        """Save final outputs"""
        logger.info("Saving final corrected audio files...")
        for file_name, stats in training_result.file_stats.items():
            best_epoch = stats["best_epoch"]
            best_cache_path = self.config.cache_dir / f"{file_name}_epoch{best_epoch}.wav"
            output_path = self.config.output_dir / f"{file_name}_corrected.wav"

            try:
                waveform, sr = torchaudio.load(str(best_cache_path))
                torchaudio.save(str(output_path), waveform, sr)
                logger.info(f"Saved {file_name} (best from epoch {best_epoch + 1})")
            except Exception as e:
                logger.error(f"Error saving {file_name}: {e}")

        # Calculate and log final statistics
        if training_result.file_stats:
            orig_scores = [stats.get("original_en_score", 0) for stats in training_result.file_stats.values() if stats.get("original_en_score") is not None]
            best_scores = [stats["best_score"] for stats in training_result.file_stats.values()]

            if orig_scores and best_scores:
                avg_orig = np.mean(orig_scores)
                avg_best = np.mean(best_scores)
                avg_improvement = avg_best - avg_orig

                logger.info(f"Final stats: original={avg_orig:.4f}, best={avg_best:.4f}, improvement={avg_improvement:.4f}")

                if self.config.use_wandb:
                    wandb.log(
                        {"final/en_score_mean/original": avg_orig, "final/en_score_mean/best": avg_best, "final/en_score_mean/improvement": avg_improvement}
                    )


def main():
    """Main function"""
    load_dotenv()

    # Configuration
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.0001,
        epochs=50,
        use_wandb=True,
        project_name="accent-maximization",
    )

    # Initialize and run trainer
    trainer = AccentTrainer(config)
    results = trainer.train()

    # Evaluate with best epoch
    best_epoch = results.best_epoch
    logger.info(f"Best epoch: {best_epoch} with improvement: {results.best_improvement:.4f}")

    if best_epoch is not None:
        eval_results = trainer.evaluate(best_epoch=best_epoch)
        logger.info(f"Evaluation with best model: improvement={eval_results['avg_improvement']:.4f}")

    # Results summary
    print(f"\nTraining Complete! Processed {len(results.file_stats)} files over {config.epochs} epochs")

    # Display top improved files
    improvements = [
        (file_name, stats["best_score"] - stats.get("original_en_score", 0))
        for file_name, stats in results.file_stats.items()
        if stats.get("original_en_score") is not None
    ]

    if improvements:
        improvements.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 most improved files:")
        for i, (file_name, improvement) in enumerate(improvements[:5]):
            stats = results.file_stats[file_name]
            print(f"{i + 1}. {file_name}: {improvement:.4f} improvement ({stats.get('original_en_score', 0):.4f} → {stats['best_score']:.4f})")


if __name__ == "__main__":
    main()
