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
from claion.core.accent_evaluator import EvaluatorConfig
from claion.data.utils import get_root_path

# Configure logger
logger = logging.getLogger(__name__)

# Get root path once at module level
ROOT_PATH = get_root_path()


@dataclass
class TrainingConfig:
    """Configuration for accent training process."""

    # Training parameters
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_grad_norm: float = 1.0
    epochs: int = 5
    max_files: Optional[int] = None  # This can still be None as it's a limit
    file_pattern: str = "*.wav"

    # Model configuration
    evaluator_model: str = "openai/whisper-base"

    # Paths configuration - directly using ROOT_PATH
    data_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "speechocean762" / "train" / "audios")
    output_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "outputs")
    cache_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "cache")
    model_dir: Path = field(default_factory=lambda: ROOT_PATH / "data" / "models")

    # Logging configuration
    use_wandb: bool = True  # Changed from False to True
    project_name: str = "accent-maximization"

    # Experiment tracking fields
    experiment_name: str = field(default_factory=lambda: f"accent_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Computing configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Validate the configuration and ensure directories exist."""
        # Ensure all directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResult:
    """Results from a training session."""

    epoch_stats: List[dict] = field(default_factory=list)
    file_stats: dict = field(default_factory=dict)
    best_epoch: Optional[int] = None
    best_improvement: float = 0.0


@dataclass
class FileProcessingResult:
    """Results from processing a single file."""

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


# Modified AccentTrainer class to use the dataclass
class AccentTrainer:
    """
    Trainer for SpeechBrainSTSPipeline to maximize English accent scores.

    Now uses a dataclass for configuration to improve type safety and code organization.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the Accent Trainer with a configuration dataclass."""
        # Store the configuration
        self.config = config

        # Initialize accent evaluator
        from claion.core.accent_evaluator import AccentEvaluator

        evaluator_config = EvaluatorConfig(model_path=self.config.evaluator_model, device=self.config.device)
        self.evaluator = AccentEvaluator(evaluator_config)

        # Initialize STS pipeline
        from claion.pipes.sb_sts import SpeechBrainSTSPipeline

        self.sts_pipeline = SpeechBrainSTSPipeline(device=self.config.device)

        # Initialize W&B if enabled (now enabled by default)
        self.use_wandb = self.config.use_wandb
        if self.use_wandb:
            self._setup_wandb()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.sts_pipeline.model.parameters(), lr=self.config.learning_rate)

        # Enable gradient computation for the STS model
        for param in self.sts_pipeline.model.parameters():
            param.requires_grad = True

    def _setup_wandb(self):
        """Set up Weights & Biases logging."""
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            run_name = self.config.experiment_name
            wandb.init(
                project=self.config.project_name,
                name=run_name,
                config={
                    "batch_size": self.config.batch_size,
                    "evaluator_model": self.config.evaluator_model,
                    "device": self.config.device,
                    "learning_rate": self.config.learning_rate,
                    "max_grad_norm": self.config.max_grad_norm,
                    "epochs": self.config.epochs,
                    "tags": self.config.tags,
                    "notes": self.config.notes,
                },
                tags=self.config.tags,
                notes=self.config.notes,
            )
        else:
            logger.warning("WANDB_API_KEY not found in environment variables. W&B logging disabled.")
            self.use_wandb = False

    def _initialize_evaluator(self):
        """Initialize the accent evaluator."""
        logger.info(f"Initializing Accent Evaluator with model: {self.config.evaluator_model}")
        eval_config = EvaluatorConfig(model_path=self.config.evaluator_model, device=self.config.device)
        return AccentEvaluator(eval_config)

    def _save_model_for_epoch(self, epoch):
        """Save the STS pipeline model for the current epoch."""
        try:
            model_path = str(Path(self.config.model_dir) / f"sts_model_epoch_{epoch}.pt")
            logger.info(f"Saving model for epoch {epoch} to {model_path}")

            # Save the model state dictionary and optimizer state
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.sts_pipeline.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                model_path,
            )

            # Log model saving in wandb
            if self.use_wandb:
                wandb.log({"model_saved": True, "model_epoch": epoch, "model_path": model_path})

            logger.info(f"Successfully saved model for epoch {epoch}")
        except Exception as e:
            logger.error(f"Error saving model for epoch {epoch}: {e}")

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def process_file(self, audio_path, epoch, train_mode=True):
        """
        Process a single audio file to maximize English accent score.

        Args:
            audio_path (str): Path to the audio file
            epoch (int): Current epoch number
            train_mode (bool): Whether to run in training mode with backprop

        Returns:
            dict: Results containing scores and improvement
        """
        file_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Determine input path based on epoch
        if epoch == 0:
            # First epoch uses original audio
            input_path = Path(audio_path)
        else:
            # Subsequent epochs use output from previous epoch
            prev_cache_path = str(Path(self.config.cache_dir) / f"{file_name}_epoch{epoch - 1}.wav")
            input_path = Path(prev_cache_path)

        # Process the file with error handling
        try:
            # Evaluate input audio
            input_result = self.evaluator(str(input_path), transcribe_audio=True)
            input_en_score = input_result.accents.get("English", 0.0)

            if train_mode:
                # Set model to training mode
                self.sts_pipeline.model.train()

                # Zero gradients
                self.optimizer.zero_grad()
            else:
                # Set model to evaluation mode
                self.sts_pipeline.model.eval()

            # Process with SpeechBrainSTSPipeline
            corrected_audio = self.sts_pipeline.generate_speech(input_path)

            # Ensure audio has correct dimensions
            if corrected_audio.ndim > 1 and corrected_audio.shape[0] == 1:
                corrected_audio = corrected_audio.squeeze(0)

            # Save to cache for this epoch
            cache_path = str(Path(self.config.cache_dir) / f"{file_name}_epoch{epoch}.wav")
            torchaudio.save(
                cache_path,
                corrected_audio,
                sample_rate=self.sts_pipeline.sampling_rate,
            )

            # Evaluate corrected audio
            corrected_result = self.evaluator(cache_path, transcribe_audio=True)
            corrected_en_score = corrected_result.accents.get("English", 0.0)

            # Calculate improvement (this will be our reward signal)
            improvement = corrected_en_score - input_en_score

            if train_mode:
                # We want to maximize the English accent score, so we'll minimize the negative score
                # This is our "loss" function: negative of the improvement in English accent score
                loss = -improvement

                # Manually create a scalar tensor for backpropagation
                loss_tensor = torch.tensor(loss, device=self.config.device, requires_grad=True)

                # Backward pass
                loss_tensor.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.sts_pipeline.model.parameters(), self.config.max_grad_norm)

                # Update model parameters
                self.optimizer.step()

                # Log gradients
                if self.use_wandb:
                    total_grad_norm = 0.0
                    for p in self.sts_pipeline.model.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm**0.5

                    wandb.log({"grad_norm": total_grad_norm, "loss": loss.item()})

            # Reset the model to free up memory
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
            logger.error(f"Error processing {file_name} at epoch {epoch}: {e}")
            # Clean up and reset pipeline after error
            self._cleanup_memory()
            self.sts_pipeline.reset_model()
            return None

    def evaluate(self, test_audio_files=None, best_epoch=None):
        """
        Evaluate the model on test audio files.

        Args:
            test_audio_files (list): List of test audio file paths. If None, use all files in data_dir.
            best_epoch (int): The best epoch to load model from. If None, use the current model.

        Returns:
            dict: Evaluation results
        """
        logger.info("Starting evaluation...")

        # Load the best model if specified
        if best_epoch is not None:
            best_model_path = str(Path(self.config.model_dir) / f"sts_model_epoch_{best_epoch}.pt")
            if os.path.exists(best_model_path):
                logger.info(f"Loading best model from epoch {best_epoch}")
                checkpoint = torch.load(best_model_path)
                self.sts_pipeline.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.warning(f"Best model from epoch {best_epoch} not found. Using current model.")

        # Set model to evaluation mode
        self.sts_pipeline.model.eval()

        # Get test files if not provided
        if test_audio_files is None:
            test_audio_files = glob.glob(str(Path(self.config.data_dir) / "*.wav"))
            logger.info(f"Using {len(test_audio_files)} files from data directory for evaluation")

        # Process each file in evaluation mode
        eval_results = []
        for audio_path in tqdm(test_audio_files, desc="Evaluating"):
            # Use epoch 0 for evaluation (doesn't matter as we're not updating weights)
            result = self.process_file(audio_path, epoch=0, train_mode=False)
            if result:
                eval_results.append(result)

        # Calculate evaluation statistics
        if eval_results:
            avg_input_score = np.mean([r.input_en_score for r in eval_results])
            avg_corrected_score = np.mean([r.corrected_en_score for r in eval_results])
            avg_improvement = np.mean([r.improvement for r in eval_results])

            logger.info("Evaluation statistics:")
            logger.info(f"  Average input English score: {avg_input_score:.4f}")
            logger.info(f"  Average corrected English score: {avg_corrected_score:.4f}")
            logger.info(f"  Average improvement: {avg_improvement:.4f}")

            # Log to W&B
            if self.use_wandb:
                wandb.log(
                    {
                        "eval/en_score_mean/input": avg_input_score,
                        "eval/en_score_mean/corrected": avg_corrected_score,
                        "eval/en_score_mean/improvement": avg_improvement,
                    }
                )

            return {
                "eval_results": eval_results,
                "avg_input_score": avg_input_score,
                "avg_corrected_score": avg_corrected_score,
                "avg_improvement": avg_improvement,
                "num_files_processed": len(eval_results),
            }

        return {"eval_results": [], "error": "No files were successfully processed during evaluation"}

    def train(self):
        """
        Train the SpeechBrainSTSPipeline over multiple epochs to maximize
        English accent scores for all audio files.

        Returns:
            TrainingResult: Training results with per-epoch statistics
        """
        # Get all audio files
        audio_files = glob.glob(str(Path(self.config.data_dir) / self.config.file_pattern))

        # Limit number of files if specified
        if self.config.max_files:
            audio_files = audio_files[: self.config.max_files]

        logger.info(f"Found {len(audio_files)} audio files to process")

        # Initialize results dictionary
        training_result = TrainingResult()

        # Training loop
        for epoch in range(self.config.epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{self.config.epochs}")

            # Process each file
            epoch_results = []

            # Process files in batches
            for i in range(0, len(audio_files), self.config.batch_size):
                batch_files = audio_files[i : i + self.config.batch_size]
                batch_size = len(batch_files)

                logger.info(
                    f"Processing batch {i // self.config.batch_size + 1}/{(len(audio_files) + self.config.batch_size - 1) // self.config.batch_size}, size: {batch_size}"
                )

                # Log batch size to W&B
                if self.use_wandb:
                    wandb.log({"batch_size": batch_size})

                for audio_path in tqdm(batch_files, desc=f"Epoch {epoch + 1}, Batch {i // self.config.batch_size + 1}"):
                    result = self.process_file(audio_path, epoch, train_mode=True)

                    if result:
                        epoch_results.append(result)

                        # Update file stats
                        file_name = result.file_name
                        if file_name not in training_result.file_stats:
                            training_result.file_stats[file_name] = {
                                "original_en_score": result.input_en_score if epoch == 0 else None,
                                "epoch_scores": [],
                                "best_epoch": 0,
                                "best_score": 0,
                            }

                        # Update epoch scores
                        training_result.file_stats[file_name]["epoch_scores"].append(result.corrected_en_score)

                        # Update best epoch if score improved
                        if result.corrected_en_score > training_result.file_stats[file_name]["best_score"]:
                            training_result.file_stats[file_name]["best_score"] = result.corrected_en_score
                            training_result.file_stats[file_name]["best_epoch"] = epoch

                        # Log individual file results to W&B
                        if self.use_wandb:
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
                training_result.epoch_stats.append(epoch_stats)

                # Update best epoch overall if improvement is better
                if avg_improvement > training_result.best_improvement:
                    training_result.best_improvement = avg_improvement
                    training_result.best_epoch = epoch

                # Log to W&B with explicit en_score_mean
                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "en_score_mean/input": avg_input_score,
                            "en_score_mean/corrected": avg_corrected_score,
                            "en_score_mean/improvement": avg_improvement,
                        }
                    )

            # Save model for this epoch
            self._save_model_for_epoch(epoch)

            # Clean up at the end of each epoch
            self._cleanup_memory()

        # Save final outputs (best version for each file)
        logger.info("Saving final corrected audio files...")
        for file_name, stats in training_result.file_stats.items():
            best_epoch = stats["best_epoch"]
            best_cache_path = str(Path(self.config.cache_dir) / f"{file_name}_epoch{best_epoch}.wav")
            output_path = str(Path(self.config.output_dir) / f"{file_name}_corrected.wav")

            # Copy best version to output dir
            try:
                waveform, sr = torchaudio.load(best_cache_path)
                torchaudio.save(output_path, waveform, sr)
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

                logger.info("Final statistics:")
                logger.info(f"  Average original English score: {avg_orig:.4f}")
                logger.info(f"  Average best English score: {avg_best:.4f}")
                logger.info(f"  Average improvement: {avg_improvement:.4f}")

                # Log to W&B with explicit en_score_mean
                if self.use_wandb:
                    wandb.log(
                        {"final/en_score_mean/original": avg_orig, "final/en_score_mean/best": avg_best, "final/en_score_mean/improvement": avg_improvement}
                    )

        # Close W&B
        if self.use_wandb:
            wandb.finish()

        return training_result


def main():
    """
    Main function to train the SpeechBrainSTSPipeline.
    """
    # Load environment variables for wandb
    load_dotenv()

    # Create training configuration
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.0001,
        epochs=50,
        use_wandb=True,  # This is now the default, but explicitly setting it for clarity
        project_name="accent-maximization",
    )

    # Initialize trainer with the configuration
    trainer = AccentTrainer(config)

    # Run the training
    results = trainer.train()

    # Find the best epoch based on improvement
    best_epoch = results.best_epoch
    best_improvement = results.best_improvement

    logger.info(f"Best epoch: {best_epoch} with improvement: {best_improvement:.4f}")

    # Evaluate the model using the best epoch
    if best_epoch is not None:
        eval_results = trainer.evaluate(best_epoch=best_epoch)
        logger.info(f"Evaluation results with best model from epoch {best_epoch}:")
        logger.info(f"  Average improvement: {eval_results['avg_improvement']:.4f}")

    # Print summary
    print("\nTraining Complete!")
    print(f"Processed {len(results.file_stats)} files over {config.epochs} epochs")

    # Print top improvements
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
