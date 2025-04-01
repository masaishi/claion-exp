import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import Audio, Dataset
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, SpeechT5Processor, TrainerCallback

from claion.core.accent_evaluator import AccentEvaluator, EvaluatorConfig

# Constants
SAMPLING_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the accent evaluator
accent_config = EvaluatorConfig(model_path="openai/whisper-base", device=DEVICE.type)
accent_evaluator = AccentEvaluator(accent_config)

# Initialize the vocoder (used for generating speech during training)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

# Initialize the speaker embedding model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": DEVICE},
    savedir=os.path.join("/tmp", spk_model_name),
)


# Create a function to extract speaker embeddings
def extract_speaker_embedding(waveform):
    """Extract a speaker embedding using SpeechBrain."""
    with torch.no_grad():
        waveform_tensor = torch.tensor(waveform).unsqueeze(0).to(DEVICE)
        speaker_embeddings = speaker_model.encode_batch(waveform_tensor)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
    return speaker_embeddings.squeeze().cpu().numpy()


# Define a custom data collator for our speech-to-speech task
@dataclass
class SpeechToSpeechDataCollator:
    processor: SpeechT5Processor

    def __call__(self, features):
        # Extract the audio inputs and speaker embeddings
        input_values = [feature["input_values"] for feature in features]
        speaker_embeddings = [feature["speaker_embeddings"] for feature in features]

        # Pad the batch - FIXED: We need to properly format this
        batch = {
            "input_values": self.processor.pad({"input_values": input_values}, return_tensors="pt").input_values,
            "speaker_embeddings": torch.tensor(np.array(speaker_embeddings)),
            # For seq2seq training, we need a labels key
            "labels": self.processor.pad(
                {"input_values": input_values},  # In this case, target is the same as input
                return_tensors="pt",
            ).input_values,
        }

        return batch


# Define a callback to evaluate accent scores during training
class AccentScoreCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor, vocoder):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.vocoder = vocoder

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Compute accent scores on evaluation dataset."""
        model.eval()

        # Sample a few examples to evaluate
        sample_indices = np.random.choice(len(self.eval_dataset), min(5, len(self.eval_dataset)), replace=False)
        total_en_score = 0.0

        for idx in sample_indices:
            example = self.eval_dataset[idx]
            input_values = torch.tensor(example["input_values"]).unsqueeze(0).to(model.device)
            speaker_emb = torch.tensor(example["speaker_embeddings"]).unsqueeze(0).to(model.device)

            # Generate speech
            with torch.no_grad():
                generated_speech = model.generate_speech(input_values, speaker_emb, vocoder=self.vocoder)

            # Save the generated speech to a temporary file
            tmp_path = f"/tmp/generated_speech_{idx}.wav"
            if generated_speech.ndim == 1:
                generated_speech = generated_speech.unsqueeze(0)
            torchaudio.save(tmp_path, generated_speech.cpu(), sample_rate=SAMPLING_RATE)

            # Evaluate accent using the AccentEvaluator
            result = accent_evaluator(tmp_path)
            en_score = result.accents.get("English", 0.0)
            total_en_score += en_score

            # Clean up
            os.remove(tmp_path)

        avg_en_score = total_en_score / len(sample_indices)
        state.log_history.append({"eval_en_accent_score": avg_en_score})
        print(f"Average English accent score: {avg_en_score:.4f}")

        model.train()


# Function to prepare our dataset
def prepare_dataset(data_path, processor):
    """Prepare dataset for training."""
    # Load audio files from the specified path
    # This assumes data_path contains WAV files
    audio_files = list(Path(data_path).glob("*.wav"))

    # Create a simple dataset dictionary
    dataset_dict = {
        "audio": [str(f) for f in audio_files],
        "file_name": [f.stem for f in audio_files],
    }

    # Create a dataset object
    dataset = Dataset.from_dict(dataset_dict)

    # Add audio loading
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    def preprocess_function(examples):
        """Process audio files into model inputs."""
        audio_arrays = [example["array"] for example in examples["audio"]]
        processed_inputs = []

        for audio_array in audio_arrays:
            # Get input values using the processor
            inputs = processor(audio=audio_array, sampling_rate=SAMPLING_RATE, return_tensors="np")
            processed_inputs.append(inputs.input_values.squeeze())

        # Extract speaker embeddings
        speaker_embeddings = [extract_speaker_embedding(array) for array in audio_arrays]

        return {
            "input_values": processed_inputs,
            "speaker_embeddings": speaker_embeddings,
        }

    # Apply preprocessing to the dataset
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return processed_dataset


# Define a custom model wrapper for Seq2SeqTrainer compatibility
class SpeechT5ForSpeechToSpeechTraining(SpeechT5ForSpeechToSpeech):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_values=None,
        speaker_embeddings=None,
        labels=None,
        return_dict=None,
    ):
        # For training, we need to override the forward method to handle our inputs
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Use the standard SpeechT5 encoder
        encoder_outputs = self.encoder(
            input_values=input_values,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        # Compute loss if we have labels
        loss = None
        if labels is not None:
            # Generate the output using the decoder
            decoder_outputs = self.decoder(
                encoder_hidden_states=hidden_states,
                speaker_embeddings=speaker_embeddings,
                return_dict=return_dict,
            )

            # Simplified loss computation - mean squared error between input and output
            # This is a simplification - you may want to use a more sophisticated loss
            loss = torch.nn.functional.mse_loss(decoder_outputs.last_hidden_state, self.encoder(input_values=labels, return_dict=True).last_hidden_state)

        if not return_dict:
            output = (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return dict(
            loss=loss,
            last_hidden_state=hidden_states,
        )


# Custom loss function for accent optimization
class AccentOptimizationLoss(torch.nn.Module):
    def __init__(self, accent_evaluator, vocoder):
        super().__init__()
        self.accent_evaluator = accent_evaluator
        self.vocoder = vocoder

    def forward(self, model_outputs, speaker_embeddings, inputs):
        """Compute loss based on accent evaluator scores."""
        # This is a placeholder for demonstration
        # In a real implementation, you'd need to incorporate accent scores
        # into the training loop more directly
        return model_outputs.loss  # Default model loss


# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    # Since we can't directly compute accent scores here (needs generated audio),
    # we rely on the AccentScoreCallback for that metric
    return {"loss": eval_pred.loss}


# Main training function
def train_accent_correction_model(
    model_name="microsoft/speecht5_vc",
    train_data_path="data/speechocean762/train/audios",
    eval_data_path="data/speechocean762/valid/audios",
    output_dir="./accent_correction_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
):
    # Load model and processor
    processor = SpeechT5Processor.from_pretrained(model_name)

    # Create our custom model wrapper
    model = SpeechT5ForSpeechToSpeech.from_pretrained(model_name)
    model = model.to(DEVICE)

    # Prepare datasets
    train_dataset = prepare_dataset(train_data_path, processor)
    eval_dataset = prepare_dataset(eval_data_path, processor)

    # Print dataset info
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        gradient_accumulation_steps=gradient_accumulation_steps,
        predict_with_generate=False,  # We're using our custom generate method
    )

    # Create data collator
    data_collator = SpeechToSpeechDataCollator(processor=processor)

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Add our custom accent score callback
    trainer.add_callback(AccentScoreCallback(eval_dataset=eval_dataset, processor=processor, vocoder=vocoder))

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)

    return model, processor


if __name__ == "__main__":
    trained_model, processor = train_accent_correction_model()

    # Test the trained model on a sample file
    file_name = "000002"
    input_file = Path(f"data/speechocean762/test/audios/{file_name}.wav")

    # Load audio
    waveform, sr = torchaudio.load(str(input_file))
    if sr != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)(waveform)

    audio_array = waveform.squeeze().numpy()

    # Extract speaker embedding
    speaker_embeddings = extract_speaker_embedding(audio_array)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0).to(DEVICE)

    # Process audio for model input
    inputs = processor(audio=audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt").to(DEVICE)

    # Generate corrected speech
    with torch.no_grad():
        corrected_audio = trained_model.generate_speech(
            inputs.input_values,
            speaker_embeddings,
            vocoder=vocoder,
        )

    # Save the output
    if corrected_audio.ndim == 1:
        corrected_audio = corrected_audio.unsqueeze(0)

    output_dir = Path("data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)

    torchaudio.save(
        str(output_dir / f"{file_name}_corrected.wav"),
        corrected_audio.cpu(),
        sample_rate=SAMPLING_RATE,
    )

    # Evaluate the accent score
    result = accent_evaluator(str(output_dir / f"{file_name}_corrected.wav"))
    en_score = result.accents.get("English", 0.0)
    print(f"English accent score for {file_name}: {en_score:.4f}")
