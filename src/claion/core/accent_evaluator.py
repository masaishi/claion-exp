import re
from typing import Dict, List

import torch
import torchaudio
from iso639 import Lang
from torchaudio.transforms import Resample
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from claion.schemas.accent_evaluator_types import (
    AccentEvaluationResult,
    AudioInput,
    AudioProcessingResult,
    EvaluatorConfig,
    TokenProbabilities,
    TranscriptionResult,
)


class AccentEvaluator:
    def __init__(
        self,
        config: EvaluatorConfig = EvaluatorConfig(),
    ):
        """
        Initialize the accent evaluator with the specified model configuration.

        Args:
            config: EvaluatorConfig object containing model settings
        """
        self.config = config
        self.device = config.device
        self.model = WhisperForConditionalGeneration.from_pretrained(config.model_path).to(self.device)
        self.tokenizer = WhisperTokenizer.from_pretrained(config.model_path)
        self.processor = WhisperProcessor.from_pretrained(config.model_path)

        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.TRANSCRIBE_TOKEN_ID = 50258  # Transcribe task token ID (<|transcribe|>)

        print(f"Using device: {self.device}")
        print(f"Available audio backends: {torchaudio.list_audio_backends()}")

    def load_audio(self, audio_input: AudioInput) -> AudioProcessingResult:
        """
        Load and preprocess audio file.

        Args:
            audio_input: AudioInput object with file path and duration settings

        Returns:
            AudioProcessingResult containing the processed audio data
        """
        waveform, sample_rate = torchaudio.load(uri=audio_input.audio_file_path, backend="soundfile")
        max_samples = audio_input.max_duration * sample_rate
        waveform = waveform[:, :max_samples]

        if sample_rate != 16000:
            waveform = Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000

        input_features = self.processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features.to(self.device)

        return AudioProcessingResult(waveform=waveform, sample_rate=sample_rate, input_features=input_features)

    def _get_lang_tokens(self) -> List[str]:
        """Get language tokens from the tokenizer."""
        return [t for t in self.tokenizer.additional_special_tokens if len(t) == 6]

    def _get_model_logits(self, input_features: torch.Tensor) -> torch.Tensor:
        """Get model logits from input features."""
        decoder_input_ids = torch.full((input_features.shape[0], 1), self.TRANSCRIBE_TOKEN_ID, dtype=torch.long).to(self.device)
        return self.model(input_features, decoder_input_ids=decoder_input_ids).logits

    def get_tokens_probabilities(self, logits: torch.Tensor, token_ids: List[int]) -> TokenProbabilities:
        """
        Calculate token probabilities from logits.

        Args:
            logits: Model output logits
            token_ids: IDs of tokens to analyze

        Returns:
            TokenProbabilities with processed logits and probabilities
        """
        logits = logits.clone()
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, :, token_ids] = False
        logits[mask] = -float("inf")

        return TokenProbabilities(logits=logits, probabilities=logits.softmax(dim=-1).cpu())

    def get_sorted_results(self, logits: torch.Tensor, tokens: List[str]) -> Dict[str, float]:
        """
        Get sorted probability results for specified tokens.

        Args:
            logits: Model output logits
            tokens: List of tokens to analyze

        Returns:
            Dictionary of tokens and their probabilities, sorted by probability
        """
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_probs = self.get_tokens_probabilities(logits, token_ids)

        results = [
            {token: token_probs.probabilities[input_idx, 0, token_id].item() for token_id, token in zip(token_ids, tokens)}
            for input_idx in range(token_probs.logits.shape[0])
        ]

        summed_results = {token: sum([result[token] for result in results]) for token in tokens}
        average_results = {token: total / len(results) for token, total in summed_results.items()}
        sorted_results = dict(sorted(average_results.items(), key=lambda item: item[1], reverse=True))

        return sorted_results

    def get_language_accent(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Analyze language accent probabilities.

        Args:
            logits: Model output logits

        Returns:
            Dictionary of language names and their accent probabilities
        """
        lang_tokens = self._get_lang_tokens()
        lang_tokens = [lang for lang in lang_tokens if lang not in ["<|jw|>"]]
        sorted_results = self.get_sorted_results(logits, lang_tokens)

        return {Lang(lang[2:-2]).name: value for lang, value in sorted_results.items()}

    def transcribe(self, input_features: torch.Tensor) -> TranscriptionResult:
        """
        Transcribe audio from input features.

        Args:
            input_features: Processed audio features

        Returns:
            TranscriptionResult with transcript and confidence scores
        """
        generated_ids = self.model.generate(
            input_features,
            return_dict_in_generate=True,
            return_token_timestamps=True,
        )
        token_scores = generated_ids["scores"]
        tokens = generated_ids["sequences"][0]

        transcription = self.tokenizer.decode(tokens, skip_special_tokens=True, decode_with_timestamps=True)

        scores = [torch.max(score).item() for score in token_scores]
        token_scores_dict = {self.tokenizer.decode([token]): score for token, score in zip(tokens, scores)}

        special_token_pattern = re.compile(r"^<\|.*\|>$")
        word_scores = [score for token, score in token_scores_dict.items() if not special_token_pattern.match(token)]
        mean_score = sum(word_scores) / len(word_scores) if word_scores else 0

        return TranscriptionResult(
            transcript=transcription,
            token_scores=token_scores_dict,
            mean_score=mean_score,
        )

    def __call__(self, audio_file_path: str, max_duration: int = 60, transcribe_audio: bool = False) -> AccentEvaluationResult:
        """
        Evaluate accent in an audio file.

        Args:
            audio_file_path: Path to the audio file
            max_duration: Maximum duration to process in seconds
            transcribe_audio: Whether to perform transcription (can be disabled to save processing time)

        Returns:
            AccentEvaluationResult with accent analysis and transcription
        """
        audio_input = AudioInput(audio_file_path=audio_file_path, max_duration=max_duration)
        audio_processed = self.load_audio(audio_input)

        logits = self._get_model_logits(audio_processed.input_features)
        accent_result = self.get_language_accent(logits)

        # Optional transcription
        if transcribe_audio:
            transcription_results = self.transcribe(audio_processed.input_features)
            transcript = transcription_results.transcript
            token_scores = transcription_results.token_scores
            mean_score = transcription_results.mean_score
        else:
            transcript = ""
            token_scores = {}
            mean_score = 0.0

        return AccentEvaluationResult(
            accents=accent_result,
            transcript=transcript,
            token_scores=token_scores,
            mean_score=mean_score,
        )


if __name__ == "__main__":
    config = EvaluatorConfig(model_path="openai/whisper-base", device="cpu")
    evaluator = AccentEvaluator(config)
    result = evaluator("tests/data/000080.wav", transcribe_audio=True)

    import json

    print(json.dumps(result.dict(), indent=2))
