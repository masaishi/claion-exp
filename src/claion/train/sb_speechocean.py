import glob
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from claion.data.utils import get_root_path

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get root path once at module level
ROOT_PATH = get_root_path()


@dataclass
class TrainingConfig:
    """Configuration for accent training process."""

    # Training parameters
    batch_size: int = 4
    learning_rate: float = 0.0001
    max_grad_norm: float = 1.0
    epochs: int = 10
    max_files: Optional[int] = None
    file_pattern: str = "*.wav"

    # Model configuration
    evaluator_model: str = "openai/whisper-base"

    # Paths configuration - directly using ROOT_PATH
    data_dir: Path = ROOT_PATH / "data" / "speechocean762" / "train" / "audios"
    output_dir: Path = ROOT_PATH / "data" / "outputs"
    cache_dir: Path = ROOT_PATH / "data" / "cache"
    model_dir: Path = ROOT_PATH / "data" / "models"

    # Logging configuration
    project_name: str = "accent-maximization"
    experiment_name: str = f"accent_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Computing configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessingResult:
    """Results from processing a single audio file."""

    file_name: str
    input_en_score: float
    corrected_en_score: float
    improvement: float
    input_transcript: str
    corrected_transcript: str
    input_accents: Dict[str, float]
    corrected_accents: Dict[str, float]
    epoch: int
    loss: Optional[float] = None


@dataclass
class TrainingResult:
    """Results from a complete training session."""

    best_epoch: int
    best_improvement: float
    file_stats: Dict[str, Dict] = None
    epoch_stats: List[Dict] = None

    def __post_init__(self):
        """Initialize collections if None."""
        if self.file_stats is None:
            self.file_stats = {}
        if self.epoch_stats is None:
            self.epoch_stats = []


class AccentTrainer:
    """
    Trainer to maximize English accent scores in speech.

    Uses dataclasses for configuration and results to improve
    type safety and code organization.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config

        # Initialize accent evaluator
        from claion.core.accent_evaluator import AccentEvaluator, EvaluatorConfig

        evaluator_config = EvaluatorConfig(model_path=self.config.evaluator_model, device=self.config.device)
        self.evaluator = AccentEvaluator(evaluator_config)

        # Initialize STS pipeline
        from claion.pipes.sb_sts import SpeechBrainSTSPipeline

        self.sts_pipeline = SpeechBrainSTSPipeline(device=self.config.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.sts_pipeline.model.parameters(), lr=self.config.learning_rate)

        # Enable gradient computation for the STS model
        for param in self.sts_pipeline.model.parameters():
            param.requires_grad = True

        # Setup wandb logging
        self._setup_wandb()

    def _setup_wandb(self):
        """Set up Weights & Biases logging."""
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY not found. Cannot continue without wandb.")

        wandb.login(key=api_key)

        # Convert dataclass to dict for wandb config
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")}

        # Convert Path objects to strings for wandb
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)

        wandb.init(project=self.config.project_name, name=self.config.experiment_name, config=config_dict)

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_model(self, epoch: int):
        """Save model checkpoint for the current epoch."""
        try:
            model_path = str(Path(self.config.model_dir) / f"sts_model_epoch_{epoch}.pt")
            logger.info(f"Saving model for epoch {epoch} to {model_path}")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.sts_pipeline.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                model_path,
            )

            wandb.log({"model_saved": True, "model_epoch": epoch})
            logger.info(f"Successfully saved model for epoch {epoch}")
        except Exception as e:
            logger.error(f"Error saving model for epoch {epoch}: {e}")

    def process_file(self, audio_path: str, epoch: int, train_mode: bool = True) -> Optional[ProcessingResult]:
        """
        Process a single audio file to maximize English accent score.

        Args:
            audio_path: Path to the audio file
            epoch: Current epoch number
            train_mode: Whether to run in training mode with backprop

        Returns:
            ProcessingResult or None if processing failed
        """
        file_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Determine input path based on epoch
        if epoch == 0:
            input_path = Path(audio_path)
        else:
            input_path = Path(self.config.cache_dir) / f"{file_name}_epoch{epoch - 1}.wav"

        try:
            # Evaluate input audio
            input_result = self.evaluator(str(input_path), transcribe_audio=True)
            input_en_score = input_result.accents.get("English", 0.0)

            # Set model mode
            if train_mode:
                self.sts_pipeline.model.train()
                self.optimizer.zero_grad()
            else:
                self.sts_pipeline.model.eval()

            # Process with STS pipeline
            corrected_audio = self.sts_pipeline.generate_speech(input_path)

            # Ensure audio has correct dimensions
            if corrected_audio.ndim > 1 and corrected_audio.shape[0] == 1:
                corrected_audio = corrected_audio.squeeze(0)

            # Save to cache
            cache_path = str(Path(self.config.cache_dir) / f"{file_name}_epoch{epoch}.wav")
            torchaudio.save(cache_path, corrected_audio, sample_rate=self.sts_pipeline.sampling_rate)

            # Evaluate corrected audio
            corrected_result = self.evaluator(cache_path, transcribe_audio=True)
            corrected_en_score = corrected_result.accents.get("English", 0.0)

            # Calculate improvement
            improvement = corrected_en_score - input_en_score

            if train_mode:
                # Maximize English score by minimizing negative improvement
                loss = -improvement
                loss_tensor = torch.tensor(loss, device=self.config.device, requires_grad=True)
                loss_tensor.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.sts_pipeline.model.parameters(), self.config.max_grad_norm)

                # Update model parameters
                self.optimizer.step()

                # Log to wandb
                wandb.log(
                    {
                        "loss": loss.item(),
                        "file_name": file_name,
                        "epoch": epoch,
                        "input_en_score": input_en_score,
                        "corrected_en_score": corrected_en_score,
                        "improvement": improvement,
                    }
                )

            # Reset model to free memory if not training
            if not train_mode:
                self.sts_pipeline.reset_model()

            self._cleanup_memory()

            # Return processing result
            return ProcessingResult(
                file_name=file_name,
                input_en_score=input_en_score,
                corrected_en_score=corrected_en_score,
                improvement=improvement,
                input_transcript=input_result.transcript,
                corrected_transcript=corrected_result.transcript,
                input_accents=input_result.accents,
                corrected_accents=corrected_result.accents,
                epoch=epoch,
                loss=-improvement if train_mode else None,
            )

        except Exception as e:
            logger.error(f"Error processing {file_name} at epoch {epoch}: {e}")
            self._cleanup_memory()
            self.sts_pipeline.reset_model()
            return None

    def train(self) -> TrainingResult:
        """
        Train the model to maximize English accent scores.

        Returns:
            TrainingResult with training statistics and best epoch info
        """
        # Get audio files
        audio_files = glob.glob(str(Path(self.config.data_dir) / self.config.file_pattern))

        # Limit files if specified
        if self.config.max_files:
            audio_files = audio_files[: self.config.max_files]

        logger.info(f"Found {len(audio_files)} audio files to process")

        # Initialize training result
        result = TrainingResult(best_epoch=0, best_improvement=0.0)

        # Training loop
        for epoch in range(self.config.epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{self.config.epochs}")

            epoch_results = []

            # Process files in batches
            for i in range(0, len(audio_files), self.config.batch_size):
                batch_files = audio_files[i : i + self.config.batch_size]

                for audio_path in tqdm(batch_files, desc=f"Epoch {epoch + 1}"):
                    process_result = self.process_file(audio_path, epoch, train_mode=True)

                    if process_result:
                        epoch_results.append(process_result)

                        # Update file stats
                        file_name = process_result.file_name
                        if file_name not in result.file_stats:
                            result.file_stats[file_name] = {
                                "original_score": process_result.input_en_score if epoch == 0 else None,
                                "best_score": 0,
                                "best_epoch": 0,
                                "epoch_scores": [],
                            }

                        # Track scores by epoch
                        result.file_stats[file_name]["epoch_scores"].append(process_result.corrected_en_score)

                        # Update best score for this file
                        if process_result.corrected_en_score > result.file_stats[file_name]["best_score"]:
                            result.file_stats[file_name]["best_score"] = process_result.corrected_en_score
                            result.file_stats[file_name]["best_epoch"] = epoch

            # Calculate epoch statistics
            if epoch_results:
                avg_input_score = np.mean([r.input_en_score for r in epoch_results])
                avg_corrected_score = np.mean([r.corrected_en_score for r in epoch_results])
                avg_improvement = np.mean([r.improvement for r in epoch_results])

                logger.info(f"Epoch {epoch + 1} statistics:")
                logger.info(f"  Average input English score: {avg_input_score:.4f}")
                logger.info(f"  Average corrected English score: {avg_corrected_score:.4f}")
                logger.info(f"  Average improvement: {avg_improvement:.4f}")

                # Save epoch statistics
                epoch_stats = {
                    "epoch": epoch,
                    "avg_input_score": avg_input_score,
                    "avg_corrected_score": avg_corrected_score,
                    "avg_improvement": avg_improvement,
                    "num_files_processed": len(epoch_results),
                }
                result.epoch_stats.append(epoch_stats)

                # Log to wandb
                wandb.log({"epoch": epoch, "avg_input_score": avg_input_score, "avg_corrected_score": avg_corrected_score, "avg_improvement": avg_improvement})

                # Update best epoch overall
                if avg_improvement > result.best_improvement:
                    result.best_improvement = avg_improvement
                    result.best_epoch = epoch

            # Save model
            self._save_model(epoch)
            self._cleanup_memory()

        # Save final outputs (best version for each file)
        logger.info("Saving final corrected audio files...")
        for file_name, stats in result.file_stats.items():
            best_epoch = stats["best_epoch"]
            best_cache_path = str(Path(self.config.cache_dir) / f"{file_name}_epoch{best_epoch}.wav")
            output_path = str(Path(self.config.output_dir) / f"{file_name}_corrected.wav")

            try:
                waveform, sr = torchaudio.load(best_cache_path)
                torchaudio.save(output_path, waveform, sr)
                logger.info(f"Saved {file_name} (best from epoch {best_epoch + 1})")
            except Exception as e:
                logger.error(f"Error saving {file_name}: {e}")

        # Log final stats
        orig_scores = [stats.get("original_score", 0) for stats in result.file_stats.values() if stats.get("original_score") is not None]
        best_scores = [stats["best_score"] for stats in result.file_stats.values()]

        if orig_scores and best_scores:
            avg_orig = np.mean(orig_scores)
            avg_best = np.mean(best_scores)
            avg_improvement = avg_best - avg_orig

            logger.info("Final statistics:")
            logger.info(f"  Average original English score: {avg_orig:.4f}")
            logger.info(f"  Average best English score: {avg_best:.4f}")
            logger.info(f"  Average improvement: {avg_improvement:.4f}")

            wandb.log({"final_avg_original": avg_orig, "final_avg_best": avg_best, "final_improvement": avg_improvement})

        # Close wandb
        wandb.finish()

        return result


def main():
    """Main function to run the accent trainer."""
    # Load environment variables
    load_dotenv()

    # Create configuration
    config = TrainingConfig()

    # Initialize and run trainer
    trainer = AccentTrainer(config)
    results = trainer.train()

    # Print summary
    print("\nTraining Complete!")
    print(f"Processed {len(results.file_stats)} files over {config.epochs} epochs")
    print(f"Best epoch: {results.best_epoch} with improvement: {results.best_improvement:.4f}")

    # Print top improvements
    improvements = [
        (file_name, stats["best_score"] - stats.get("original_score", 0))
        for file_name, stats in results.file_stats.items()
        if stats.get("original_score") is not None
    ]

    if improvements:
        improvements.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 most improved files:")
        for i, (file_name, improvement) in enumerate(improvements[:5]):
            stats = results.file_stats[file_name]
            print(f"{i + 1}. {file_name}: {improvement:.4f} improvement ({stats.get('original_score', 0):.4f} → {stats['best_score']:.4f})")


if __name__ == "__main__":
    main()
