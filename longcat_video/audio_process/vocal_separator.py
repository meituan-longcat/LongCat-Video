import math
import librosa
import numpy as np
import torch

from audio_separator.separator.uvr_lib_v5 import spec_utils


def _demix_batched(model, mix, batch_size):
    model.initialize_model_settings()
    chunk_size = model.chunk_size
    overlap = model.overlap
    gen_size = chunk_size - 2 * model.trim
    pad = gen_size + model.trim - (mix.shape[-1] % gen_size)
    mixture = np.concatenate(
        (
            np.zeros((2, model.trim), dtype=np.float32),
            mix,
            np.zeros((2, pad), dtype=np.float32),
        ),
        axis=1,
    )
    step = int((1 - overlap) * chunk_size)
    starts = list(range(0, mixture.shape[-1], step))
    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros_like(result)

    for group_start in range(0, len(starts), batch_size):
        batch_parts = []
        batch_meta = []
        for start in starts[group_start : group_start + batch_size]:
            end = min(start + chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start
            window = None
            if overlap:
                window = np.tile(
                    np.hanning(chunk_size_actual)[None, None, :], (1, 2, 1)
                )

            mix_part = mixture[:, start:end]
            if chunk_size_actual != chunk_size:
                mix_part = np.pad(
                    mix_part,
                    ((0, 0), (0, chunk_size - chunk_size_actual)),
                )
            batch_parts.append(mix_part)
            batch_meta.append((start, end, chunk_size_actual, window))

        mix_wave = torch.from_numpy(np.stack(batch_parts)).to(
            model.torch_device, dtype=torch.float32
        )
        with torch.no_grad():
            batch_result = model.run_model(mix_wave)

        for index, (start, end, chunk_size_actual, window) in enumerate(batch_meta):
            chunk_result = batch_result[index : index + 1, ..., :chunk_size_actual]
            if window is not None:
                chunk_result *= window
                divider[..., start:end] += window
            else:
                divider[..., start:end] += 1
            result[..., start:end] += chunk_result

    source = (result / divider)[0, :, model.trim : -model.trim]
    source = source[:, : mix.shape[-1]]
    source *= model.compensate
    return source


def separate_vocals(vocal_separator, source_path, sample_rate=16000, batch_size=2, duration=None):
    model = vocal_separator.model_instance
    mix = model.prepare_mix(source_path)
    mix = spec_utils.normalize(
        wave=mix,
        max_peak=model.normalization_threshold,
        min_peak=-model.normalization_threshold,
    )
    if duration is not None:
        model.initialize_model_settings()
        target_samples = min(
            math.ceil(duration * model.sample_rate), mix.shape[-1]
        )
        mix = mix[..., : min(target_samples + model.chunk_size, mix.shape[-1])]

    source = _demix_batched(model, mix, batch_size)
    source = spec_utils.normalize(
        wave=source,
        max_peak=model.normalization_threshold,
        min_peak=-model.normalization_threshold,
    )
    mono = librosa.to_mono(source)
    if model.sample_rate != sample_rate:
        mono = librosa.resample(
            mono,
            orig_sr=model.sample_rate,
            target_sr=sample_rate,
        )
    if duration is not None:
        target_output_samples = math.ceil(target_samples * sample_rate / model.sample_rate)
        mono = mono[:target_output_samples]
    return mono.astype(np.float32, copy=False), sample_rate
