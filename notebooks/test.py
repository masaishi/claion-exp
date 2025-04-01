from pathlib import Path

import IPython.display as ipd
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor

# Define the path to your audio file
audio_file_path = Path("../data/speechocean762/train/audios/000002.wav")
audio_array, sampling_rate = sf.read(str(audio_file_path))

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

inputs = processor(audio=audio_array, sampling_rate=sampling_rate, return_tensors="pt")

# Create correct speaker embeddings - this is the key fix
# Use a properly shaped fixed embedding (simulating a speaker identity)
speaker_embeddings = torch.randn(1, 512)  # Correct shape for SpeechT5

# Process the audio
with torch.no_grad():
    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    # Get encoder outputs
    encoder_outputs = model.speecht5.encoder(input_values=input_values, attention_mask=attention_mask, return_dict=True)

    encoder_hidden_states = encoder_outputs.last_hidden_state

    # Create initial output sequence (zeros)
    bsz = input_values.size(0)
    output_sequence = encoder_hidden_states.new_zeros(bsz, 1, model.config.num_mel_bins)

    # Set parameters
    minlenratio = 0.0
    maxlenratio = 10.0
    threshold = 0.5

    # Calculate min and max lengths
    maxlen = int(encoder_hidden_states.size(1) * maxlenratio / model.config.reduction_factor)
    minlen = int(encoder_hidden_states.size(1) * minlenratio / model.config.reduction_factor)

    # For downsample encoder attention mask if needed
    if attention_mask is not None:
        encoder_attention_mask = attention_mask
    else:
        encoder_attention_mask = torch.ones_like(input_values, dtype=torch.int)

    # Start generation loop
    spectrogram = []
    past_key_values = None
    idx = 0

    while True:
        idx += 1

        # Run decoder prenet with correctly shaped speaker embeddings
        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

        # Run decoder
        decoder_out = model.speecht5.decoder.wrapped_decoder(
            hidden_states=decoder_hidden_states[:, -1:],
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )

        last_decoder_output = decoder_out.last_hidden_state.squeeze(1)
        past_key_values = decoder_out.past_key_values

        # Predict spectrum
        spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
        spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)
        spectrogram.append(spectrum)

        # Extend output sequence
        new_spectrogram = spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins)
        output_sequence = torch.cat((output_sequence, new_spectrogram), dim=1)

        # Check stop token probability
        prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

        # Check stopping conditions
        if idx < minlen:
            continue
        elif idx >= maxlen:
            break
        elif torch.sum(prob, dim=-1) >= threshold:
            break

    # Process generated spectrogram
    spectrograms = torch.stack(spectrogram)
    spectrograms = spectrograms.transpose(0, 1).flatten(1, 2)
    final_spectrogram = model.speech_decoder_postnet.postnet(spectrograms)

    # Convert from mel spectrogram back to waveform using Griffin-Lim algorithm
    # First convert to numpy
    mel_spec_np = final_spectrogram.squeeze().detach().numpy()

    # Original audio parameters for matching
    n_fft = 1024
    hop_length = 256
    win_length = 1024

    # Reconstruct the waveform from the mel spectrogram using Griffin-Lim
    # First, we need to convert the mel spectrogram back to linear spectrogram
    linear_spec = librosa.feature.inverse.mel_to_stft(mel_spec_np, sr=sampling_rate, n_fft=n_fft, power=1.0)

    # Use Griffin-Lim to reconstruct the audio signal
    waveform_np = librosa.griffinlim(
        linear_spec,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=32,  # More iterations give better quality but take longer
    )

    # Normalize the waveform to match input amplitude
    max_amp_orig = np.max(np.abs(audio_array))
    max_amp_new = np.max(np.abs(waveform_np))
    if max_amp_new > 0:
        waveform_np = waveform_np * (max_amp_orig / max_amp_new)

    # Save the generated audio
    output_audio_path = Path("../data/output_audio.wav")
    sf.write(str(output_audio_path), waveform_np, samplerate=sampling_rate)

    print(f"Audio saved to {output_audio_path}")
    print(f"Waveform shape: {waveform_np.shape}")

    # Display the waveform in the notebook
    print("Playing generated audio:")
    display(ipd.Audio(waveform_np, rate=sampling_rate))

    # Display the original audio for comparison
    print("Playing original audio:")
    display(ipd.Audio(audio_array, rate=sampling_rate))

    # Visualize the waveforms
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.title("Original Waveform")
    plt.plot(audio_array)

    plt.subplot(3, 1, 2)
    plt.title("Generated Waveform")
    plt.plot(waveform_np)

    # Also visualize the spectrogram
    plt.subplot(3, 1, 3)
    plt.title("Generated Mel Spectrogram")
    librosa.display.specshow(mel_spec_np, sr=sampling_rate, hop_length=hop_length, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.show()
