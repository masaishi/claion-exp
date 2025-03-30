import gc
import glob
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from claion.core.accent_evaluator import AccentEvaluator, EvaluatorConfig
from claion.data.utils import get_root_path
from claion.pipes.sb_sts import SpeechBrainSTSPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AccentTrainer:
    """
    Trainer for SpeechBrainSTSPipeline to maximize English accent scores.

    Key improvements:
    1. Uses a single SpeechBrainSTSPipeline instance with proper memory management
    2. Supresses unnecessary warnings
    3. Ensures models are downloaded only once and cached properly
    4. Implements proper error recovery
    """

    def __init__(
        self,
        data_dir=None,
        output_dir=None,
        cache_dir=None,
        evaluator_model="openai/whisper-base",
        device=None,
        use_wandb=True,
        project_name="accent-maximization",
    ):
        """Initialize the Accent Trainer."""
        # Set up device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set up directories using get_root_path
        root_path = get_root_path()
        self.data_dir = data_dir if data_dir else str(root_path / "data" / "speechocean762" / "train" / "audios")
        self.output_dir = output_dir if output_dir else str(root_path / "data" / "outputs")
        self.cache_dir = cache_dir if cache_dir else str(root_path / "data" / "cache")

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize accent evaluator
        self.evaluator_model = evaluator_model
        self.evaluator = self._initialize_evaluator()

        # Initialize STS pipeline once and reuse it
        logger.info("Initializing SpeechBrainSTSPipeline...")
        self.sts_pipeline = SpeechBrainSTSPipeline(device=self.device)

        # Initialize W&B
        self.use_wandb = use_wandb
        self.project_name = project_name
        if self.use_wandb:
            self._setup_wandb()

    def _setup_wandb(self):
        """Set up Weights & Biases logging."""
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"accent_training_{timestamp}"
            wandb.init(project=self.project_name, name=run_name)
        else:
            logger.warning("WANDB_API_KEY not found in environment variables. W&B logging disabled.")
            self.use_wandb = False

    def _initialize_evaluator(self):
        """Initialize the accent evaluator."""
        logger.info(f"Initializing Accent Evaluator with model: {self.evaluator_model}")
        eval_config = EvaluatorConfig(model_path=self.evaluator_model, device=self.device)
        return AccentEvaluator(eval_config)

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def process_file(self, audio_path, epoch):
        """
        Process a single audio file to maximize English accent score.

        Args:
            audio_path (str): Path to the audio file
            epoch (int): Current epoch number

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
            prev_cache_path = str(Path(self.cache_dir) / f"{file_name}_epoch{epoch - 1}.wav")
            input_path = Path(prev_cache_path)

        # Process the file with error handling
        try:
            # Evaluate input audio
            input_result = self.evaluator(str(input_path), transcribe_audio=True)
            input_en_score = input_result.accents.get("English", 0.0)

            # Process with SpeechBrainSTSPipeline
            corrected_audio = self.sts_pipeline.generate_speech(input_path)

            # Ensure audio has correct dimensions
            if corrected_audio.ndim > 1 and corrected_audio.shape[0] == 1:
                corrected_audio = corrected_audio.squeeze(0)

            # Save to cache for this epoch
            cache_path = str(Path(self.cache_dir) / f"{file_name}_epoch{epoch}.wav")
            torchaudio.save(
                cache_path,
                corrected_audio,
                sample_rate=self.sts_pipeline.sampling_rate,
            )

            # Evaluate corrected audio
            corrected_result = self.evaluator(cache_path, transcribe_audio=True)
            corrected_en_score = corrected_result.accents.get("English", 0.0)

            # Reset the model to free up memory
            self.sts_pipeline.reset_model()
            self._cleanup_memory()

            # Return results
            return {
                "file_name": file_name,
                "epoch": epoch,
                "input_en_score": input_en_score,
                "corrected_en_score": corrected_en_score,
                "improvement": corrected_en_score - input_en_score,
                "input_transcript": input_result.transcript,
                "corrected_transcript": corrected_result.transcript,
                "input_accents": input_result.accents,
                "corrected_accents": corrected_result.accents,
            }

        except Exception as e:
            logger.error(f"Error processing {file_name} at epoch {epoch}: {e}")
            # Clean up and reset pipeline after error
            self._cleanup_memory()
            self.sts_pipeline.reset_model()
            return None

    def train(self, epochs=5, max_files=None, file_pattern="*.wav"):
        """
        Train the SpeechBrainSTSPipeline over multiple epochs to maximize
        English accent scores for all audio files.

        Args:
            epochs (int): Number of epochs to train
            max_files (int): Maximum number of files to process (None for all)
            file_pattern (str): Pattern to match audio files

        Returns:
            dict: Training results with per-epoch statistics
        """
        # Get all audio files
        audio_files = glob.glob(str(Path(self.data_dir) / file_pattern))

        # Limit number of files if specified
        if max_files:
            audio_files = audio_files[:max_files]

        logger.info(f"Found {len(audio_files)} audio files to process")

        # Initialize results dictionary
        all_results = {"audio_files": audio_files, "epoch_stats": [], "file_stats": {}}

        # Training loop
        for epoch in range(epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{epochs}")

            # Process each file
            epoch_results = []
            for audio_path in tqdm(audio_files, desc=f"Epoch {epoch + 1}"):
                result = self.process_file(audio_path, epoch)

                if result:
                    epoch_results.append(result)

                    # Update file stats
                    file_name = result["file_name"]
                    if file_name not in all_results["file_stats"]:
                        all_results["file_stats"][file_name] = {
                            "original_en_score": result["input_en_score"] if epoch == 0 else None,
                            "epoch_scores": [],
                            "best_epoch": 0,
                            "best_score": 0,
                        }

                    # Update epoch scores
                    all_results["file_stats"][file_name]["epoch_scores"].append(result["corrected_en_score"])

                    # Update best epoch if score improved
                    if result["corrected_en_score"] > all_results["file_stats"][file_name]["best_score"]:
                        all_results["file_stats"][file_name]["best_score"] = result["corrected_en_score"]
                        all_results["file_stats"][file_name]["best_epoch"] = epoch

                    # Log to W&B
                    if self.use_wandb:
                        wandb.log(
                            {
                                "file_name": result["file_name"],
                                "epoch": epoch,
                                "input_en_score": result["input_en_score"],
                                "corrected_en_score": result["corrected_en_score"],
                                "improvement": result["improvement"],
                            }
                        )

            # Calculate epoch statistics
            if epoch_results:
                avg_input_score = np.mean([r["input_en_score"] for r in epoch_results])
                avg_corrected_score = np.mean([r["corrected_en_score"] for r in epoch_results])
                avg_improvement = np.mean([r["improvement"] for r in epoch_results])

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
                all_results["epoch_stats"].append(epoch_stats)

                # Log to W&B
                if self.use_wandb:
                    wandb.log(
                        {"epoch": epoch, "avg_input_score": avg_input_score, "avg_corrected_score": avg_corrected_score, "avg_improvement": avg_improvement}
                    )

            # Clean up at the end of each epoch
            self._cleanup_memory()

        # Save final outputs (best version for each file)
        logger.info("Saving final corrected audio files...")
        for file_name, stats in all_results["file_stats"].items():
            best_epoch = stats["best_epoch"]
            best_cache_path = str(Path(self.cache_dir) / f"{file_name}_epoch{best_epoch}.wav")
            output_path = str(Path(self.output_dir) / f"{file_name}_corrected.wav")

            # Copy best version to output dir
            try:
                waveform, sr = torchaudio.load(best_cache_path)
                torchaudio.save(output_path, waveform, sr)
                logger.info(f"Saved {file_name} (best from epoch {best_epoch + 1})")
            except Exception as e:
                logger.error(f"Error saving {file_name}: {e}")

        # Calculate and log final statistics
        if all_results["file_stats"]:
            orig_scores = [stats.get("original_en_score", 0) for stats in all_results["file_stats"].values() if stats.get("original_en_score") is not None]
            best_scores = [stats["best_score"] for stats in all_results["file_stats"].values()]

            if orig_scores and best_scores:
                avg_orig = np.mean(orig_scores)
                avg_best = np.mean(best_scores)
                avg_improvement = avg_best - avg_orig

                logger.info("Final statistics:")
                logger.info(f"  Average original English score: {avg_orig:.4f}")
                logger.info(f"  Average best English score: {avg_best:.4f}")
                logger.info(f"  Average improvement: {avg_improvement:.4f}")

                # Log to W&B
                if self.use_wandb:
                    wandb.log({"final_avg_original_score": avg_orig, "final_avg_best_score": avg_best, "final_avg_improvement": avg_improvement})

        # Close W&B
        if self.use_wandb:
            wandb.finish()

        return all_results


def main():
    """
    Main function to train the SpeechBrainSTSPipeline on all .wav files for 5 epochs.
    """
    # Load environment variables for wandb
    load_dotenv()

    # Setup trainer
    trainer = AccentTrainer(use_wandb=True, project_name="accent-maximization")

    # Run the training for 5 epochs on all .wav files
    results = trainer.train(epochs=5)

    # Print summary
    print("\nTraining Complete!")
    print(f"Processed {len(results['file_stats'])} files over 5 epochs")

    # Print top improvements
    improvements = [
        (file_name, stats["best_score"] - stats.get("original_en_score", 0))
        for file_name, stats in results["file_stats"].items()
        if stats.get("original_en_score") is not None
    ]

    if improvements:
        improvements.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 most improved files:")
        for i, (file_name, improvement) in enumerate(improvements[:5]):
            stats = results["file_stats"][file_name]
            print(f"{i + 1}. {file_name}: {improvement:.4f} improvement ({stats.get('original_en_score', 0):.4f} → {stats['best_score']:.4f})")


if __name__ == "__main__":
    main()
