# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, cast

import numpy as np
import torch
import torchaudio.functional as F
from fairseq2.data._memory import MemoryBlock
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    FileMapper,
    read_sequence,
)
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.logging import get_log_writer
from fairseq2.models.hub import load_model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.batch_layout import BatchLayout
from numpy.typing import NDArray

from omnilingual_asr.datasets.utils.audio import add_waveform_processing
from omnilingual_asr.models.inference.align import (
    align_ctc,
    align_llm,
    chunk_waveform,
)
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.config import (
    ModelType,
    Wav2Vec2LlamaStreamingConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import (
    Wav2Vec2LlamaBeamSearchConfig,
    Wav2Vec2LlamaModel,
)
from omnilingual_asr.models.wav2vec2_llama.syntax import Modality, ModalityInput

log = get_log_writer(__name__)

AudioInput = (
    List[Path]
    | List[str]
    | List[str | Path]
    | List[bytes]
    | List[NDArray[np.int8]]
    | List[bytes | NDArray[np.int8]]
    | List[Dict[str, Any]]
)

MAX_ALLOWED_AUDIO_SEC: Final = 40


@dataclass
class ContextExample:
    """
    Represents a single context example with audio and text.

    Args:
        audio: Audio input (same formats as AudioInput)
        text: Corresponding text transcription
    """

    audio: str | Path | bytes | NDArray[np.int8] | Dict[str, Any]
    text: str


def resample_to_16khz(
    audio_data: Dict[str, Any], target_sample_rate: int = 16000
) -> Dict[str, Any]:
    """
    Resample audio waveform to target sample rate (16kHz by default).

    Args:
        audio_data: Dictionary containing 'waveform' and 'sample_rate' keys
        target_sample_rate: Target sample rate (default: 16000)

    Returns:
        Dictionary with resampled waveform and updated sample rate
    """
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data["sample_rate"]

    if current_sample_rate != target_sample_rate:
        log.debug(f"Resampling from {current_sample_rate}Hz to {target_sample_rate}Hz")

        # Different audio reading mechanisms can cause the shape to be either (channels, time)
        # or (time, channels). We go heuristically by the longer axis to enforce (channels, time)
        # is going to F.resample, and (time, channels) is returned
        assert len(waveform.shape) <= 2
        need_transpose = (
            len(waveform.shape) > 1 and waveform.shape[0] > waveform.shape[1]
        )
        if need_transpose:
            waveform = waveform.transpose(1, 0)
        waveform = F.resample(
            waveform,
            orig_freq=current_sample_rate,
            new_freq=target_sample_rate,
        )
        if need_transpose:
            waveform = waveform.transpose(1, 0)
        audio_data["sample_rate"] = target_sample_rate
        audio_data["waveform"] = waveform

    return audio_data


def repeat_to_max_len(
    lists: List[List[ContextExample]], max_len: int
) -> List[List[ContextExample]]:
    """Repeats each inner list of `lists` until it reaches the `max_len`.
    This is used to replicate context examples to fit the zero-shot model training setup, which always
    saw exactly `max_len` context examples for low-resource languages.

    If more than `max_len` examples are provided we trim them down to `max_len`.
    """

    def extend_list(lst):
        repetitions = (max_len // len(lst)) + 1
        return (lst * repetitions)[:max_len]

    return [extend_list(lst) for lst in lists]


def assert_max_length(
    audio_data: Dict[str, Any], target_sample_rate: int = 16000
) -> Dict[str, Any]:
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data["sample_rate"]
    waveform_len_s = len(waveform) / current_sample_rate
    if waveform_len_s > MAX_ALLOWED_AUDIO_SEC:
        raise ValueError(
            f"Audio length {waveform_len_s:.2f}s exceeds max {MAX_ALLOWED_AUDIO_SEC}s. Use chunk_len parameter to process long files."
        )
    return audio_data


class ASRInferencePipeline:
    def __init__(
        self,
        model_card: str | None,
        *,
        model: Wav2Vec2LlamaModel | Wav2Vec2AsrModel | None = None,
        tokenizer: Tokenizer | None = None,
        device: str | None | torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        beam_search_config: Wav2Vec2LlamaBeamSearchConfig | None = None,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            model_card: Model card name to load from the hub (mutually exclusive with model/tokenizer)
                Recommended to use model_card for inference !

            model: Pre-loaded Wav2Vec2LlamaModel instance (mutually exclusive with model_card)
            tokenizer: Pre-loaded Tokenizer instance (mutually exclusive with model_card)
            device: Device to run inference on
            dtype: Data type for model inference
            beam_search_config: Optional beam search configuration

        Raises:
            ValueError: If both model_card and (model/tokenizer) are provided, or if model/tokenizer are provided without each other
        """
        # Validate mutually exclusive arguments
        if model_card is not None and (model is not None or tokenizer is not None):
            raise ValueError(
                "model_card is mutually exclusive with model/tokenizer. "
                "Provide either model_card OR both model and tokenizer."
            )

        if (model is None) != (tokenizer is None):
            raise ValueError(
                "Both model and tokenizer must be provided together when not using model_card"
            )

        if model_card is None and (model is None or tokenizer is None):
            raise ValueError(
                "Must provide either model_card OR both model and tokenizer"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        # Load or use provided model and tokenizer
        if model_card is not None:
            log.info(f"Loading model from model card: {model_card}")
            self.model = load_model(model_card, device=self.device, dtype=self.dtype)

            log.info(f"Loading tokenizer from model card: {model_card}")
            self.tokenizer = load_tokenizer(model_card)
        else:
            assert isinstance(tokenizer, Tokenizer)
            assert isinstance(model, (Wav2Vec2LlamaModel, Wav2Vec2AsrModel))
            log.info("Using provided model and tokenizer")
            self.model = model
            self.model = self.model.to(device=self.device)
            self.tokenizer = tokenizer

        self.model.eval()

        # Set up beam search
        if beam_search_config is None:
            beam_search_config = Wav2Vec2LlamaBeamSearchConfig(
                nbest=1,
                length_norm=False,
            )
        else:
            assert isinstance(beam_search_config, Wav2Vec2LlamaBeamSearchConfig)

        # cursed logic
        self.streaming_config: Wav2Vec2LlamaStreamingConfig = (
            Wav2Vec2LlamaStreamingConfig()
        )
        if isinstance(self.model, Wav2Vec2LlamaModel) and hasattr(
            self.model, "streaming_config"
        ):
            self.streaming_config = self.model.streaming_config

        self.beam_search_generator = None
        if isinstance(self.model, Wav2Vec2LlamaModel):
            self.beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
                model=self.model,
                config=beam_search_config,
                streaming_config=self.streaming_config,
            )

        assert self.tokenizer is not None
        self.token_decoder = self.tokenizer.create_decoder(skip_special_tokens=True)
        self.token_encoder = self.tokenizer.create_encoder()

        self.audio_decoder = AudioDecoder(dtype=torch.float32)
        self.file_mapper = FileMapper(cached_fd_count=200)
        pad_idx = getattr(self.tokenizer.vocab_info, "pad_idx", 0)
        text_collate_opts = CollateOptionsOverride("text", pad_value=pad_idx)

        self.full_collater = Collater(
            pad_value=0,
            overrides=[text_collate_opts],
        )
        self.collater_audio = Collater(pad_value=0)
        self.collater_text = Collater(pad_value=pad_idx)

        model_source = (
            f"model_card={model_card}" if model_card else "provided model/tokenizer"
        )
        log.info(
            f"Pipeline initialized on {self.device} with dtype {self.dtype} using {model_source}"
        )

    def _create_batch_simple(
        self, wavs_langs: List[Tuple[torch.Tensor, str | None]]
    ) -> Seq2SeqBatch:
        """Create a Seq2SeqBatch from audio tensors using fairseq2 utilities."""
        audio_examples = []
        for item in wavs_langs:
            audio_examples.append(
                {
                    "audio_feature": item[0],
                    "text": torch.tensor(
                        [0], dtype=torch.int64
                    ),  # Dummy text for inference
                }
            )

        collated_data = self.full_collater(audio_examples)

        audio_data = collated_data["audio_feature"]
        text_data = collated_data["text"]

        example = {"lang": [item[1] for item in wavs_langs]}
        if all(x is None for x in example["lang"]):
            example = {}

        return Seq2SeqBatch(
            source_seqs=audio_data["seqs"].to(self.device, self.dtype),
            source_seq_lens=audio_data["seq_lens"],
            target_seqs=text_data["seqs"].to(self.device),
            target_seq_lens=text_data["seq_lens"],
            example=example,
        )

    def _apply_model_wav2vec2asr(self, batch: Seq2SeqBatch) -> List[str]:
        batch_layout = BatchLayout(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )

        logits, bl_out = self.model(batch.source_seqs, batch_layout)
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = []

        for i in range(pred_ids.shape[0]):
            seq = pred_ids[i][: bl_out.seq_lens[i]]
            mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
            mask[1:] = seq[1:] != seq[:-1]
            decoded_ids = seq[mask]
            transcriptions.append(self.token_decoder(decoded_ids))
        return transcriptions

    def _apply_model_wav2vec2llama(self, batch: Seq2SeqBatch) -> List[str]:
        assert self.beam_search_generator is not None
        assert isinstance(self.model, Wav2Vec2LlamaModel)

        if self.streaming_config is not None and self.streaming_config.is_streaming:
            segment_samples = int(
                self.streaming_config.segment_secs * self.streaming_config.sample_rate
            )
            source_seq_lens = torch.tensor(batch.source_seq_lens, device=self.device)
            n_segments = torch.ceil(source_seq_lens / segment_samples).int()

            assert isinstance(batch.example, dict)
            batch.example["n_segments"] = n_segments

            audio_segments = torch.split(batch.source_seqs, segment_samples, 1)
            audio_embeddings: List[ModalityInput] = []
            for i, segment in enumerate(audio_segments):
                seg_lengths = torch.clamp(
                    source_seq_lens - i * segment_samples,
                    min=0,
                    max=segment_samples,
                )
                modality_input = ModalityInput(
                    modality=Modality.AUDIO,
                    seqs=segment.to(self.device),
                    seq_lens=seg_lengths.tolist(),
                    loss=False,
                    embedded=False,
                )
                embedded = self.model.embed_inputs(
                    [modality_input], dtype=segment.dtype
                )[0]
                audio_embeddings.append(embedded)

            hypothesis_tokens, hypothesis_lens = (
                self.beam_search_generator.generate_hypotheses(
                    decoder_context_inputs=None,
                    decoder_context_seq_lens=None,
                    audio_embeddings=audio_embeddings,
                    batch=batch,
                )
            )
        else:
            (decoder_context, decoder_context_seq_lens, audio_embeddings) = self.model(  # type: ignore
                batch, return_decoder_inputs=True
            )

            hypothesis_tokens, hypothesis_lens = (
                self.beam_search_generator.generate_hypotheses(
                    decoder_context_inputs=decoder_context,
                    decoder_context_seq_lens=decoder_context_seq_lens,
                    audio_embeddings=None,
                    batch=None,
                )
            )

        transcriptions = []
        for i in range(hypothesis_tokens.shape[0]):
            seq_len = hypothesis_lens[i] if hypothesis_lens is not None else 0
            tokens = hypothesis_tokens[i, :seq_len]
            text = self.token_decoder(tokens)
            transcriptions.append(text)

        return transcriptions

    def _apply_model(self, batch: Seq2SeqBatch) -> List[str]:
        """Apply model forward pass to the batch."""
        if isinstance(self.model, Wav2Vec2LlamaModel):
            transcriptions = self._apply_model_wav2vec2llama(batch)
        elif isinstance(self.model, Wav2Vec2AsrModel):
            transcriptions = self._apply_model_wav2vec2asr(batch)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return transcriptions

    def _build_audio_wavform_pipeline(
        self, inp_list: AudioInput, check_max_length: bool = True
    ) -> DataPipelineBuilder:
        """Process audio inputs using fairseq2.data pipeline similar to ASR task."""
        builder = read_sequence(inp_list)

        need_to_decode = True
        first_element = inp_list[0]
        if isinstance(first_element, (Path, str)):
            builder = builder.map(str)
            builder = builder.map(self.file_mapper)
        elif isinstance(first_element, (bytes, np.ndarray)):
            if isinstance(first_element, np.ndarray):
                assert first_element.dtype in [
                    np.uint8,
                    np.int8,
                ], "Only uint8 numpy arrays are supported"
            builder = builder.map(lambda x: {"data": MemoryBlock(x)})
        elif isinstance(first_element, dict):
            need_to_decode = False
            log.info("Processing pre-decoded audio dictionaries")
            builder = builder.map(
                lambda x: {
                    "data": {
                        "waveform": torch.tensor(x["waveform"]),
                        "sample_rate": int(x["sample_rate"]),
                    }
                }
            )
        else:
            raise ValueError(f"Unsupported input type: {type(first_element)}")

        if need_to_decode:
            builder = builder.map(self.audio_decoder, selector="data")

        builder = builder.map(resample_to_16khz, selector="data")

        non_streaming = not self.streaming_config.is_streaming
        if non_streaming and check_max_length:
            builder = builder.map(assert_max_length, selector="data")

        builder = add_waveform_processing(
            builder,
            normalize_audio=True,
            dtype=self.dtype,
            selector="data.waveform",
            spec_aug_p=None,
            spec_aug_freq_mask_param=0,
            spec_aug_time_mask_param=0,
        )
        builder = builder.map(lambda x: x["data"]["waveform"])
        return builder

    def _process_context_audio(
        self, context_examples: List[ContextExample]
    ) -> Dict[str, Any] | None:
        """
        Process context audio examples into tensors.

        Args:
            context_examples: List of context examples with audio and text

        Returns:
            Dictionary of collater audio tensors
            keys = (seqs, seq_lens, is_ragged)
        """
        if not context_examples:
            return None

        raw_audio = cast(AudioInput, [example.audio for example in context_examples])
        builder = self._build_audio_wavform_pipeline(raw_audio)
        context_audio_tensors = list(builder.and_return())

        collated_audio = self.collater_audio(context_audio_tensors)
        collated_audio["seqs"] = collated_audio["seqs"].to(self.device, self.dtype)
        collated_audio["seq_lens"] = torch.tensor(
            collated_audio["seq_lens"], device=self.device
        )
        return collated_audio

    def _process_context_text(
        self, context_examples: List[ContextExample]
    ) -> List[torch.Tensor]:
        """
        Process context text examples into tokenized tensors.

        Args:
            context_examples: List of context examples with audio and text

        Returns:
            Dictionary of collater tokenized text
            keys = (seqs, seq_lens, is_ragged)
        """
        if not context_examples:
            return []

        context_text_tensors = []
        for example in context_examples:
            text_tensor = self.token_encoder(example.text)
            context_text_tensors.append(text_tensor)
        collated_text = self.collater_text(context_text_tensors)
        collated_text["seqs"] = collated_text["seqs"].to(self.device)
        collated_text["seq_lens"] = torch.tensor(
            collated_text["seq_lens"], device=self.device
        )
        return collated_text

    def _create_batch_with_context(
        self, combined_batch: List[Tuple[torch.Tensor, List[ContextExample]]]
    ) -> Seq2SeqBatch:
        """
        Create a Seq2SeqBatch with zero-shot context support.
        """
        batch = self._create_batch_simple([(item[0], None) for item in combined_batch])  # type: ignore[index]

        context_audio = []
        context_text = []
        for combined_item in combined_batch:
            context_examples = combined_item[1]  # type: ignore[index]
            context_audio_tensors = self._process_context_audio(context_examples)
            context_text_tensors = self._process_context_text(context_examples)
            context_audio.append(context_audio_tensors)
            context_text.append(context_text_tensors)

        batch.example["context_audio"] = context_audio  # type: ignore[index]
        batch.example["context_text"] = context_text  # type: ignore[index]
        return batch

    def _embed_audio_segment(self, wav_segment: torch.Tensor, input_lang: str | None) -> ModalityInput:
        """
        Embed a raw waveform segment into a ModalityInput using the model's audio encoder.
        Used by the unlimited streaming sliding window path.
        """
        assert isinstance(self.model, Wav2Vec2LlamaModel)
        batch_data = [(wav_segment, input_lang)]
        seq2seq_batch = self._create_batch_simple(batch_data)
        audio_mod = ModalityInput(
            modality=Modality.AUDIO,
            seqs=seq2seq_batch.source_seqs,
            seq_lens=seq2seq_batch.source_seq_lens,
            loss=False,
            embedded=False,
        )
        return self.model.embed_inputs([audio_mod], dtype=self.dtype)[0]

    def _transcribe_unlimited_streaming_chunk(
        self,
        wav_segment: torch.Tensor,
        input_lang: str | None,
        historical_audio_embeddings: List[ModalityInput],
        historical_text_tokens: List[ModalityInput],
    ) -> str:
        """
        Run transcription for one chunk on an unlimited streaming model,
        passing historical audio embeddings and text tokens as context.
        """
        assert isinstance(self.model, Wav2Vec2LlamaModel)
        assert self.beam_search_generator is not None

        current_audio_emb = self._embed_audio_segment(wav_segment, input_lang)
        n_total = torch.tensor(
            [len(historical_audio_embeddings) + 1],
            device=self.device,
            dtype=torch.int32,
        )
        langs = [input_lang] if input_lang else None

        tokens, lens = self.beam_search_generator.generate_hypotheses_one_segment_streaming(
            new_audio_embeddings=current_audio_emb.seqs,
            new_audio_embedding_seq_lens=current_audio_emb.seq_lens,
            n_total_segments=n_total,
            langs=langs,  # type: ignore
            previous_audio_embeddings=historical_audio_embeddings,
            previous_text_tokens=historical_text_tokens,
        )
        return self.token_decoder(tokens[0, : lens[0]])

    def _update_unlimited_history(
        self,
        wav_segment: torch.Tensor,
        safe_text: str,
        safe_duration_sec: float,
        input_lang: str | None,
        historical_audio_embeddings: List[ModalityInput],
        historical_text_tokens: List[ModalityInput],
        max_history: int,
    ) -> None:
        """
        Append the safe portion of the current chunk to history (in-place),
        then trim to `max_history` segments.

        Only the audio up to `safe_duration_sec` and the corresponding safe text
        are added — dropped words are excluded from history so the model doesn't
        see them as ground truth context in future chunks.
        """
        assert isinstance(self.model, Wav2Vec2LlamaModel)

        safe_samples = int(safe_duration_sec * 16000)
        safe_wav = wav_segment[:safe_samples]

        safe_audio_emb = self._embed_audio_segment(safe_wav, input_lang)

        safe_text_tokens_t = (
            self.token_encoder(safe_text).unsqueeze(0).to(self.device, torch.int64)
        )
        safe_text_modality = ModalityInput(
            modality=Modality.TEXT,
            seqs=safe_text_tokens_t,
            seq_lens=[safe_text_tokens_t.size(1)],
            loss=False,
            embedded=False,
        )

        historical_audio_embeddings.append(safe_audio_emb)
        historical_text_tokens.append(safe_text_modality)

        # Trim to window
        if len(historical_audio_embeddings) > max_history:
            historical_audio_embeddings.pop(0)
            historical_text_tokens.pop(0)

    @torch.inference_mode()
    def transcribe(
        self,
        inp: AudioInput,
        *,
        lang: List[str | None] | List[str] | List[None] | None = None,
        batch_size: int = 2,
        chunk_len: float | None = None,
        overlap_drop_sec: float = 1.0,
    ) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """
        Transcribes `AudioInput` into text by preprocessing (decoding, resample to 16kHz,
        converting to mono, normalizing) each input sample and performing inference with
        `self.model`.

        Returns text and extracted timestamps.

        For unlimited streaming LLM models (streaming_config.is_streaming=True), chunking
        uses a dynamic sliding window: the next window start is determined by the last word
        that ends safely before `chunk_end - overlap_drop_sec`. History (audio embeddings +
        text tokens) from previous safe chunks is fed as context to each new chunk, capped
        at `streaming_config.n_context_segments`.

        For non-streaming models, non-overlapping chunks via `chunk_waveform` are used.

        Args:
            `inp`: Audio input in different forms.
                - `List[ Path | str ]`: Audio file paths
                - `List[ bytes ]`: Raw audio data
                - `List[ np.ndarray ]`: Audio data as uint8 numpy array
                - `List[ dict[str, Any] ]`: Pre-decoded audio with 'waveform' and 'sample_rate' keys
            `lang`: Language code for the input audios (e.g., 'eng_Latn', ...) (default: None)
            `batch_size`: Number of audio samples to process in each batch (used for non-streaming path).
            `chunk_len`: Maximum window length in seconds. Required for audio longer than
                MAX_ALLOWED_AUDIO_SEC. For streaming models this is the upper-bound window;
                actual advancement is determined dynamically by timestamps.
            `overlap_drop_sec`: For the streaming sliding window, words whose end timestamp
                falls within this many seconds of the chunk boundary are considered unsafe
                and dropped (not committed to output or history). The next window starts from
                the last safe word's end time. Defaults to 1.0s.

        Returns:
            Tuple[List[str], List[List[Dict[str, Any]]]]:
                - List of transcribed texts.
                - List of List of timestamp dicts (e.g. [{'word': 'Hello', 'start': 0.0, 'end': 0.5}])
        """
        if len(inp) == 0:
            return [], []

        # fmt: off
        is_ctc_model = isinstance(self.model, Wav2Vec2AsrModel)
        is_llm_model = isinstance(self.model, Wav2Vec2LlamaModel)
        is_llm_zs_model = is_llm_model and self.model.model_type == ModelType.ZERO_SHOT
        is_unlimited_streaming = (
            is_llm_model
            and self.streaming_config is not None
            and self.streaming_config.is_streaming
        )

        if is_ctc_model and lang:
            log.info(f"Found {lang=} with a CTC model. Ignoring.")
        if is_llm_model and not lang:
            log.info("Using an LLM model without a `lang` code can lead to degraded transcription quality.")
        if is_llm_zs_model:
            raise NotImplementedError(
                "Model does not support inference without context conditioning. "
                "Please use `.transcribe_with_context()` instead of `.transcribe()`."
            )

        if not lang:
            lang = [None] * len(inp)

        assert len(lang) == len(inp), (
            f"`lang` must be a list of the same length as `inp` ({len(inp)}), "
            f"but is {len(lang)}."
        )
        # fmt: on

        final_transcripts = []
        final_timestamps = []

        for idx, (input_item, input_lang) in enumerate(zip(inp, lang)):
            single_input: AudioInput = cast(AudioInput, [input_item])
            p = self._build_audio_wavform_pipeline(
                single_input, check_max_length=False
            ).and_return()
            waveform = next(iter(p))  # Tensor[T]

            duration = waveform.shape[0] / 16000.0

            # ----------------------------------------------------------------
            # Unlimited streaming model → dynamic sliding window
            # ----------------------------------------------------------------
            if is_unlimited_streaming and chunk_len is not None and duration > chunk_len:
                max_history = getattr(
                    self.streaming_config, "n_context_segments", 1
                )

                historical_audio_embeddings: List[ModalityInput] = []
                historical_text_tokens: List[ModalityInput] = []

                chunk_start = 0.0
                input_text_parts: List[str] = []
                input_timestamps: List[Dict[str, Any]] = []

                while chunk_start < duration:
                    chunk_end = min(duration, chunk_start + chunk_len)
                    start_sample = int(chunk_start * 16000)
                    chunk_samples = int((chunk_end - chunk_start) * 16000)
                    wav_segment = waveform[start_sample : start_sample + chunk_samples]

                    # Transcribe with history as context
                    text = self._transcribe_unlimited_streaming_chunk(
                        wav_segment,
                        input_lang,
                        historical_audio_embeddings,
                        historical_text_tokens,
                    )

                    if not text.strip():
                        # Nothing transcribed — advance by a fixed stride and continue
                        chunk_start += chunk_len - overlap_drop_sec
                        continue

                    # Align current chunk's audio vs its own transcription only
                    chunk_ts: List[Dict[str, Any]] = []
                    try:
                        chunk_ts = align_llm(self, wav_segment, text, input_lang)
                    except Exception as e:
                        log.warning(
                            f"Alignment failed for streaming chunk at {chunk_start:.2f}s "
                            f"of input {idx}: {e}"
                        )

                    is_last_chunk = chunk_end >= duration

                    if not is_last_chunk and chunk_ts:
                        # Determine safe boundary: words must end before this threshold
                        safe_threshold = (chunk_end - chunk_start) - overlap_drop_sec
                        safe_ts = [w for w in chunk_ts if w["end"] <= safe_threshold]

                        if safe_ts:
                            # Commit safe words; next window starts at last safe word's end
                            committed_ts = safe_ts
                            next_start = chunk_start + safe_ts[-1]["end"]
                            safe_duration_sec = safe_ts[-1]["end"]
                        else:
                            # Fallback: no safe words found — commit all, advance by fixed stride
                            log.debug(
                                f"No safe words found at chunk_start={chunk_start:.2f}s; "
                                f"falling back to fixed stride."
                            )
                            committed_ts = chunk_ts
                            next_start = chunk_start + chunk_len - overlap_drop_sec
                            safe_duration_sec = chunk_len - overlap_drop_sec
                    else:
                        # Last chunk — commit everything
                        committed_ts = chunk_ts
                        next_start = duration
                        safe_duration_sec = chunk_end - chunk_start

                    # Build committed text from committed timestamps
                    is_word_level = committed_ts and "word" in committed_ts[0]
                    word_parts = [
                        w.get("word", w.get("char", "")) for w in committed_ts
                    ]
                    if is_word_level:
                        committed_text = " ".join(word_parts)
                    else:
                        committed_text = "".join(word_parts)

                    # Adjust timestamps to global timeline and record
                    for w in committed_ts:
                        input_timestamps.append({
                            "word": w.get("word", w.get("char", "")),
                            "start": w["start"] + chunk_start,
                            "end": w["end"] + chunk_start,
                        })

                    if committed_text.strip():
                        input_text_parts.append(committed_text)

                        # Update history with only the safe portion of audio + text.
                        # Dropped words are intentionally excluded so future chunks don't
                        # see them as committed context.
                        self._update_unlimited_history(
                            wav_segment=wav_segment,
                            safe_text=committed_text,
                            safe_duration_sec=safe_duration_sec,
                            input_lang=input_lang,
                            historical_audio_embeddings=historical_audio_embeddings,
                            historical_text_tokens=historical_text_tokens,
                            max_history=max_history,
                        )

                    chunk_start = next_start

                full_transcript = " ".join(t for t in input_text_parts if t.strip())
                final_transcripts.append(full_transcript)
                final_timestamps.append(input_timestamps)
                continue  # Move to next input item

            # ----------------------------------------------------------------
            # Non-streaming (or unlimited streaming without chunking needed)
            # ----------------------------------------------------------------
            if chunk_len is not None and duration > chunk_len:
                chunks = chunk_waveform(waveform, 16000, chunk_len)
            else:
                if duration > MAX_ALLOWED_AUDIO_SEC and chunk_len is None:
                    raise ValueError(
                        f"Audio {idx} duration {duration:.2f}s > {MAX_ALLOWED_AUDIO_SEC}s. "
                        f"Provide chunk_len parameter."
                    )
                chunks = [(waveform, 0.0)]

            input_text_parts = []
            input_timestamps = []

            chunk_waveforms = [c[0] for c in chunks]
            offsets = [c[1] for c in chunks]

            for i in range(0, len(chunk_waveforms), batch_size):
                batch_wavs = chunk_waveforms[i : i + batch_size]
                batch_offsets = offsets[i : i + batch_size]

                batch_data = [(w, input_lang) for w in batch_wavs]
                seq2seq_batch = self._create_batch_simple(batch_data)
                texts = self._apply_model(seq2seq_batch)

                for j, text in enumerate(texts):
                    wav_segment = batch_wavs[j]
                    offset = batch_offsets[j]

                    if not text.strip():
                        input_text_parts.append("")
                        continue

                    chunk_ts = []
                    try:
                        if isinstance(self.model, Wav2Vec2AsrModel):
                            chunk_ts = align_ctc(self.model, wav_segment, 16000, text)
                        elif isinstance(self.model, Wav2Vec2LlamaModel):
                            chunk_ts = align_llm(self, wav_segment, text, input_lang)
                    except Exception as e:
                        log.warning(
                            f"Alignment failed for chunk {i + j} of input {idx}: {e}"
                        )
                        chunk_ts = []

                    for w in chunk_ts:
                        input_timestamps.append({
                            "word": w.get("word", w.get("char", "")),
                            "start": w["start"] + offset,
                            "end": w["end"] + offset,
                        })

                    input_text_parts.append(text)

            full_transcript = " ".join(t for t in input_text_parts if t.strip())
            final_transcripts.append(full_transcript)
            final_timestamps.append(input_timestamps)

        return final_transcripts, final_timestamps

    @torch.inference_mode()
    def transcribe_with_context(
        self,
        inp: AudioInput,
        context_examples: List[List[ContextExample]],
        *,
        batch_size: int = 1,
    ) -> List[str]:
        """
        Transcribes `AudioInput` into text by preprocessing (decoding, resample to 16kHz, converting to mono, normalizing)
        each input sample and its `context_examples` and performing inference with `self.model` by leveraging the context examples
        as a "reference" on how to transcribe.

        The zero-shot model was trained on up to 30s samples per context example, with **10** examples per training sample.
        Please provide at least a single context example per input sample which is replicated until we reach 10 in total.
        If >10 samples are provided we crop to the first ten.

        Warning: Only works for the `omniASR_LLM_7B_ZS` model.

        Args:
            `inp`: Audio input in different forms.
                - `List[ Path | str ]`: Audio file paths
                - `List[ bytes ]`: Raw audio data
                - `List[ np.ndarray ]`: Audio data as uint8 numpy array
                - `List[ dict[str, Any] ]`: Pre-decoded audio with 'waveform' and 'sample_rate' keys
            `context_examples`: A list of context examples for each input audio.
                - `List[ List[ContextExample] ]`: Each inner list contains context examples (audio-text pairs) for that specific audio.
            `batch_size`: Number of audio samples to process in each batch.

        Returns:
            `List[str]`: Transcribed texts.
        """
        if len(inp) == 0:
            return []

        # fmt: off
        is_ctc_model = isinstance(self.model, Wav2Vec2AsrModel)
        is_llm_model = isinstance(self.model, Wav2Vec2LlamaModel)

        if is_ctc_model:
            raise NotImplementedError("CTC models do not support context conditioning. Please use `.transcribe()` instead of `.transcribe_with_context()`.")
        if is_llm_model and self.model.model_type != ModelType.ZERO_SHOT:
            raise NotImplementedError("LLM models do not support context conditioning. Please use `.transcribe()` instead of `.transcribe_with_context()`.")

        assert len(inp) == len(context_examples), f"Number of audio inputs ({len(inp)}) must match number of context examples {len(context_examples)}."

        for i, examples in enumerate(context_examples):
            assert len(examples) > 0, f"Input index {i} has no context examples, but needs at least one."
            if len(examples) > 10:
                log.info(f"Found {len(examples)} context examples for input index {i}, but can only process 10. Ignoring extra.")
        # fmt: on

        max_context_example_per_sample = 10
        context_examples = repeat_to_max_len(
            context_examples, max_len=max_context_example_per_sample
        )

        combined_builder = DataPipeline.zip(
            [
                self._build_audio_wavform_pipeline(inp).and_return(),
                read_sequence(context_examples).and_return(),
            ]
        )
        combined_builder = combined_builder.bucket(batch_size)
        combined_builder = combined_builder.map(self._create_batch_with_context)
        combined_builder = combined_builder.prefetch(1)
        combined_builder = combined_builder.map(self._apply_model)
        combined_builder = combined_builder.yield_from(
            lambda seq: read_sequence(seq).and_return()
        )
        return list(combined_builder.and_return())