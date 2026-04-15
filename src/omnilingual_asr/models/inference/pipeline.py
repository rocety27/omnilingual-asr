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
    """
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data["sample_rate"]

    if current_sample_rate != target_sample_rate:
        print(f"  [resample] {current_sample_rate}Hz → {target_sample_rate}Hz  shape={tuple(waveform.shape)}")
        log.debug(f"Resampling from {current_sample_rate}Hz to {target_sample_rate}Hz")

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
        print(f"  [resample] done  new shape={tuple(waveform.shape)}")
    else:
        print(f"  [resample] already at {target_sample_rate}Hz, no-op  shape={tuple(waveform.shape)}")

    return audio_data


def repeat_to_max_len(
    lists: List[List[ContextExample]], max_len: int
) -> List[List[ContextExample]]:
    """Repeats each inner list of `lists` until it reaches the `max_len`."""

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
        """
        print(f"\n{'='*60}")
        print(f"[init] Initializing ASRInferencePipeline")
        print(f"[init]   model_card  : {model_card}")
        print(f"[init]   device      : {device}")
        print(f"[init]   dtype       : {dtype}")
        print(f"{'='*60}")

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
            print(f"[init] device auto-selected: {device}")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        if model_card is not None:
            print(f"[init] Loading model from hub: {model_card} ...")
            log.info(f"Loading model from model card: {model_card}")
            self.model = load_model(model_card, device=self.device, dtype=self.dtype)
            print(f"[init] Model loaded: {type(self.model).__name__}")

            print(f"[init] Loading tokenizer from hub: {model_card} ...")
            log.info(f"Loading tokenizer from model card: {model_card}")
            self.tokenizer = load_tokenizer(model_card)
            print(f"[init] Tokenizer loaded: {type(self.tokenizer).__name__}")
        else:
            assert isinstance(tokenizer, Tokenizer)
            assert isinstance(model, (Wav2Vec2LlamaModel, Wav2Vec2AsrModel))
            print(f"[init] Using pre-loaded model: {type(model).__name__}")
            log.info("Using provided model and tokenizer")
            self.model = model
            self.model = self.model.to(device=self.device)
            self.tokenizer = tokenizer

        self.model.eval()
        print(f"[init] Model set to eval mode")

        if beam_search_config is None:
            beam_search_config = Wav2Vec2LlamaBeamSearchConfig(
                nbest=1,
                length_norm=False,
            )
            print(f"[init] Using default beam search config (nbest=1, length_norm=False)")
        else:
            assert isinstance(beam_search_config, Wav2Vec2LlamaBeamSearchConfig)
            print(f"[init] Using provided beam search config")

        # cursed logic
        self.streaming_config: Wav2Vec2LlamaStreamingConfig = (
            Wav2Vec2LlamaStreamingConfig()
        )
        if isinstance(self.model, Wav2Vec2LlamaModel) and hasattr(
            self.model, "streaming_config"
        ):
            self.streaming_config = self.model.streaming_config
            print(f"[init] Streaming config loaded from model:")
            print(f"[init]   is_streaming       : {self.streaming_config.is_streaming}")
            print(f"[init]   segment_secs       : {getattr(self.streaming_config, 'segment_secs', 'N/A')}")
            print(f"[init]   n_context_segments : {getattr(self.streaming_config, 'n_context_segments', 'N/A')}")
        else:
            print(f"[init] No streaming config on model — using defaults (is_streaming=False)")

        self.beam_search_generator = None
        if isinstance(self.model, Wav2Vec2LlamaModel):
            self.beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
                model=self.model,
                config=beam_search_config,
                streaming_config=self.streaming_config,
            )
            print(f"[init] Beam search generator created: {type(self.beam_search_generator).__name__}")

        assert self.tokenizer is not None
        self.token_decoder = self.tokenizer.create_decoder(skip_special_tokens=True)
        self.token_encoder = self.tokenizer.create_encoder()

        self.audio_decoder = AudioDecoder(dtype=torch.float32)
        self.file_mapper = FileMapper(cached_fd_count=200)
        pad_idx = getattr(self.tokenizer.vocab_info, "pad_idx", 0)
        text_collate_opts = CollateOptionsOverride("text", pad_value=pad_idx)

        self.full_collater = Collater(pad_value=0, overrides=[text_collate_opts])
        self.collater_audio = Collater(pad_value=0)
        self.collater_text = Collater(pad_value=pad_idx)

        model_source = (
            f"model_card={model_card}" if model_card else "provided model/tokenizer"
        )
        print(f"[init] Pipeline ready on {self.device} ({self.dtype}) using {model_source}")
        print(f"{'='*60}\n")
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
                    "text": torch.tensor([0], dtype=torch.int64),  # Dummy text for inference
                }
            )

        collated_data = self.full_collater(audio_examples)
        audio_data = collated_data["audio_feature"]
        text_data = collated_data["text"]

        example = {"lang": [item[1] for item in wavs_langs]}
        if all(x is None for x in example["lang"]):
            example = {}

        batch = Seq2SeqBatch(
            source_seqs=audio_data["seqs"].to(self.device, self.dtype),
            source_seq_lens=audio_data["seq_lens"],
            target_seqs=text_data["seqs"].to(self.device),
            target_seq_lens=text_data["seq_lens"],
            example=example,
        )
        print(f"  [batch] created  n={len(wavs_langs)}  "
              f"source_seqs={tuple(batch.source_seqs.shape)}  "
              f"seq_lens={list(batch.source_seq_lens)}  "
              f"langs={[item[1] for item in wavs_langs]}")
        return batch

    def _apply_model_wav2vec2asr(self, batch: Seq2SeqBatch) -> List[str]:
        print(f"  [model/CTC] forward pass  input shape={tuple(batch.source_seqs.shape)}")
        batch_layout = BatchLayout(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )

        logits, bl_out = self.model(batch.source_seqs, batch_layout)
        print(f"  [model/CTC] logits shape={tuple(logits.shape)}")
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = []

        for i in range(pred_ids.shape[0]):
            seq = pred_ids[i][: bl_out.seq_lens[i]]
            mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
            mask[1:] = seq[1:] != seq[:-1]
            decoded_ids = seq[mask]
            text = self.token_decoder(decoded_ids)
            transcriptions.append(text)
            print(f"  [model/CTC] item[{i}] → {repr(text[:80])}")

        return transcriptions

    def _apply_model_wav2vec2llama(self, batch: Seq2SeqBatch) -> List[str]:
        assert self.beam_search_generator is not None
        assert isinstance(self.model, Wav2Vec2LlamaModel)

        print(f"  [model/LLM] forward pass  input shape={tuple(batch.source_seqs.shape)}  "
              f"is_streaming={self.streaming_config.is_streaming}")

        if self.streaming_config is not None and self.streaming_config.is_streaming:
            segment_samples = int(
                self.streaming_config.segment_secs * self.streaming_config.sample_rate
            )
            source_seq_lens = torch.tensor(batch.source_seq_lens, device=self.device)
            n_segments = torch.ceil(source_seq_lens / segment_samples).int()
            print(f"  [model/LLM] streaming mode  segment_samples={segment_samples}  "
                  f"n_segments={n_segments.tolist()}")

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
                print(f"  [model/LLM] embedded segment[{i}]  shape={tuple(embedded.seqs.shape)}")

            hypothesis_tokens, hypothesis_lens = (
                self.beam_search_generator.generate_hypotheses(
                    decoder_context_inputs=None,
                    decoder_context_seq_lens=None,
                    audio_embeddings=audio_embeddings,
                    batch=batch,
                )
            )
        else:
            print(f"  [model/LLM] non-streaming mode — encoding audio + building decoder context")
            (decoder_context, decoder_context_seq_lens, audio_embeddings) = self.model(  # type: ignore
                batch, return_decoder_inputs=True
            )
            print(f"  [model/LLM] decoder context shape={tuple(decoder_context.shape)}  "
                  f"seq_lens={decoder_context_seq_lens}")

            hypothesis_tokens, hypothesis_lens = (
                self.beam_search_generator.generate_hypotheses(
                    decoder_context_inputs=decoder_context,
                    decoder_context_seq_lens=decoder_context_seq_lens,
                    audio_embeddings=None,
                    batch=None,
                )
            )

        print(f"  [model/LLM] hypothesis_tokens shape={tuple(hypothesis_tokens.shape)}")
        transcriptions = []
        for i in range(hypothesis_tokens.shape[0]):
            seq_len = hypothesis_lens[i] if hypothesis_lens is not None else 0
            tokens = hypothesis_tokens[i, :seq_len]
            text = self.token_decoder(tokens)
            transcriptions.append(text)
            print(f"  [model/LLM] item[{i}] seq_len={seq_len} → {repr(text[:80])}")

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
            print(f"  [model] CUDA cache cleared")
        return transcriptions

    def _build_audio_wavform_pipeline(
        self, inp_list: AudioInput, check_max_length: bool = True
    ) -> DataPipelineBuilder:
        """Process audio inputs using fairseq2.data pipeline similar to ASR task."""
        first_element = inp_list[0]
        print(f"  [audio_pipeline] building  n={len(inp_list)}  "
              f"input_type={type(first_element).__name__}  check_max_length={check_max_length}")

        builder = read_sequence(inp_list)

        need_to_decode = True
        if isinstance(first_element, (Path, str)):
            print(f"  [audio_pipeline] mode: file path → FileMapper + AudioDecoder")
            builder = builder.map(str)
            builder = builder.map(self.file_mapper)
        elif isinstance(first_element, (bytes, np.ndarray)):
            if isinstance(first_element, np.ndarray):
                assert first_element.dtype in [np.uint8, np.int8], \
                    "Only uint8 numpy arrays are supported"
            print(f"  [audio_pipeline] mode: bytes/ndarray → MemoryBlock + AudioDecoder")
            builder = builder.map(lambda x: {"data": MemoryBlock(x)})
        elif isinstance(first_element, dict):
            need_to_decode = False
            print(f"  [audio_pipeline] mode: pre-decoded dict (waveform+sample_rate) — skipping AudioDecoder")
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
            print(f"  [audio_pipeline] applying AudioDecoder")
            builder = builder.map(self.audio_decoder, selector="data")

        print(f"  [audio_pipeline] applying resample_to_16khz")
        builder = builder.map(resample_to_16khz, selector="data")

        non_streaming = not self.streaming_config.is_streaming
        if non_streaming and check_max_length:
            print(f"  [audio_pipeline] applying assert_max_length (max={MAX_ALLOWED_AUDIO_SEC}s)")
            builder = builder.map(assert_max_length, selector="data")
        elif not non_streaming:
            print(f"  [audio_pipeline] skipping max_length check (streaming model)")
        elif not check_max_length:
            print(f"  [audio_pipeline] skipping max_length check (check_max_length=False)")

        print(f"  [audio_pipeline] applying waveform normalization + dtype cast ({self.dtype})")
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
        print(f"  [audio_pipeline] pipeline ready")
        return builder

    def _process_context_audio(
        self, context_examples: List[ContextExample]
    ) -> Dict[str, Any] | None:
        if not context_examples:
            return None

        print(f"  [context_audio] processing {len(context_examples)} examples")
        raw_audio = cast(AudioInput, [example.audio for example in context_examples])
        builder = self._build_audio_wavform_pipeline(raw_audio)
        context_audio_tensors = list(builder.and_return())
        print(f"  [context_audio] decoded {len(context_audio_tensors)} tensors")

        collated_audio = self.collater_audio(context_audio_tensors)
        collated_audio["seqs"] = collated_audio["seqs"].to(self.device, self.dtype)
        collated_audio["seq_lens"] = torch.tensor(
            collated_audio["seq_lens"], device=self.device
        )
        print(f"  [context_audio] collated  seqs={tuple(collated_audio['seqs'].shape)}  "
              f"seq_lens={collated_audio['seq_lens'].tolist()}")
        return collated_audio

    def _process_context_text(
        self, context_examples: List[ContextExample]
    ) -> List[torch.Tensor]:
        if not context_examples:
            return []

        print(f"  [context_text] tokenizing {len(context_examples)} texts")
        context_text_tensors = []
        for i, example in enumerate(context_examples):
            text_tensor = self.token_encoder(example.text)
            context_text_tensors.append(text_tensor)
            print(f"  [context_text]   [{i}] {repr(example.text[:60])}  n_tokens={text_tensor.shape[0]}")

        collated_text = self.collater_text(context_text_tensors)
        collated_text["seqs"] = collated_text["seqs"].to(self.device)
        collated_text["seq_lens"] = torch.tensor(
            collated_text["seq_lens"], device=self.device
        )
        print(f"  [context_text] collated  seqs={tuple(collated_text['seqs'].shape)}")
        return collated_text

    def _create_batch_with_context(
        self, combined_batch: List[Tuple[torch.Tensor, List[ContextExample]]]
    ) -> Seq2SeqBatch:
        print(f"\n  [batch_with_context] building  n={len(combined_batch)}")
        batch = self._create_batch_simple([(item[0], None) for item in combined_batch])  # type: ignore[index]

        context_audio = []
        context_text = []
        for i, combined_item in enumerate(combined_batch):
            context_examples = combined_item[1]  # type: ignore[index]
            print(f"  [batch_with_context] item[{i}]: {len(context_examples)} context examples")
            context_audio_tensors = self._process_context_audio(context_examples)
            context_text_tensors = self._process_context_text(context_examples)
            context_audio.append(context_audio_tensors)
            context_text.append(context_text_tensors)

        batch.example["context_audio"] = context_audio  # type: ignore[index]
        batch.example["context_text"] = context_text  # type: ignore[index]
        print(f"  [batch_with_context] done")
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
        embedded = self.model.embed_inputs([audio_mod], dtype=self.dtype)[0]
        print(f"  [embed] wav shape={tuple(wav_segment.shape)} "
              f"({wav_segment.shape[0]/16000:.2f}s) → embedded shape={tuple(embedded.seqs.shape)}")
        return embedded

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

        n_history = len(historical_audio_embeddings)
        print(f"  [stream/transcribe] chunk duration={wav_segment.shape[0]/16000:.2f}s  "
              f"history_segments={n_history}  lang={input_lang}")

        current_audio_emb = self._embed_audio_segment(wav_segment, input_lang)
        n_total = torch.tensor(
            [n_history + 1],
            device=self.device,
            dtype=torch.int32,
        )
        langs = [input_lang] if input_lang else None

        print(f"  [stream/transcribe] calling generate_hypotheses_one_segment_streaming  "
              f"n_total_segments={n_total.item()}")
        tokens, lens = self.beam_search_generator.generate_hypotheses_one_segment_streaming(
            new_audio_embeddings=current_audio_emb.seqs,
            new_audio_embedding_seq_lens=current_audio_emb.seq_lens,
            n_total_segments=n_total,
            langs=langs,  # type: ignore
            previous_audio_embeddings=historical_audio_embeddings,
            previous_text_tokens=historical_text_tokens,
        )
        text = self.token_decoder(tokens[0, : lens[0]])
        print(f"  [stream/transcribe] output → {repr(text[:100])}")
        return text

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

        print(f"  [history] updating — safe_duration={safe_duration_sec:.2f}s  "
              f"safe_samples={safe_samples}  "
              f"safe_text={repr(safe_text[:60])}")

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
        print(f"  [history] tokenized safe_text: n_tokens={safe_text_tokens_t.size(1)}")

        historical_audio_embeddings.append(safe_audio_emb)
        historical_text_tokens.append(safe_text_modality)

        if len(historical_audio_embeddings) > max_history:
            historical_audio_embeddings.pop(0)
            historical_text_tokens.pop(0)
            print(f"  [history] trimmed oldest segment  "
                  f"current_size={len(historical_audio_embeddings)}  max={max_history}")
        else:
            print(f"  [history] appended  "
                  f"current_size={len(historical_audio_embeddings)}  max={max_history}")

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
        Transcribes AudioInput into text.

        For unlimited streaming LLM models (streaming_config.is_streaming=True), uses a
        dynamic sliding window with history context. For non-streaming models, uses
        non-overlapping chunks via chunk_waveform.
        """
        print(f"\n{'='*60}")
        print(f"[transcribe] called  n_inputs={len(inp)}  lang={lang}  "
              f"batch_size={batch_size}  chunk_len={chunk_len}  overlap_drop_sec={overlap_drop_sec}")

        if len(inp) == 0:
            print(f"[transcribe] empty input — returning early")
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

        print(f"[transcribe] model flags: CTC={is_ctc_model}  LLM={is_llm_model}  "
              f"ZeroShot={is_llm_zs_model}  UnlimitedStreaming={is_unlimited_streaming}")

        if is_ctc_model and lang:
            log.info(f"Found {lang=} with a CTC model. Ignoring.")
            print(f"[transcribe] WARNING: lang ignored for CTC model")
        if is_llm_model and not lang:
            log.info("Using an LLM model without a `lang` code can lead to degraded transcription quality.")
            print(f"[transcribe] WARNING: no lang provided for LLM — quality may be degraded")
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
            print(f"\n{'-'*60}")
            print(f"[transcribe] ── input[{idx}]  lang={input_lang} ──")

            single_input: AudioInput = cast(AudioInput, [input_item])
            print(f"[transcribe] input[{idx}] building audio pipeline ...")
            p = self._build_audio_wavform_pipeline(
                single_input, check_max_length=False
            ).and_return()
            waveform = next(iter(p))  # Tensor[T]

            duration = waveform.shape[0] / 16000.0
            print(f"[transcribe] input[{idx}] waveform ready  "
                  f"shape={tuple(waveform.shape)}  duration={duration:.2f}s")

            # ----------------------------------------------------------------
            # Unlimited streaming model → dynamic sliding window
            # ----------------------------------------------------------------
            if is_unlimited_streaming and chunk_len is not None and duration > chunk_len:
                print(f"\n[transcribe] input[{idx}] → UNLIMITED STREAMING / SLIDING WINDOW path")
                print(f"[transcribe]   duration={duration:.2f}s  chunk_len={chunk_len}s  "
                      f"overlap_drop_sec={overlap_drop_sec}s")

                max_history = getattr(self.streaming_config, "n_context_segments", 1)
                print(f"[transcribe]   max_history={max_history} segment(s)")

                historical_audio_embeddings: List[ModalityInput] = []
                historical_text_tokens: List[ModalityInput] = []

                chunk_start = 0.0
                input_text_parts: List[str] = []
                input_timestamps: List[Dict[str, Any]] = []
                chunk_idx = 0

                while chunk_start < duration:
                    chunk_end = min(duration, chunk_start + chunk_len)
                    start_sample = int(chunk_start * 16000)
                    chunk_samples = int((chunk_end - chunk_start) * 16000)
                    wav_segment = waveform[start_sample : start_sample + chunk_samples]
                    is_last_chunk = chunk_end >= duration

                    print(f"\n[transcribe] input[{idx}] chunk[{chunk_idx}]  "
                          f"{chunk_start:.2f}s → {chunk_end:.2f}s  "
                          f"({chunk_samples} samples)  last_chunk={is_last_chunk}")

                    # Transcribe with history as context
                    text = self._transcribe_unlimited_streaming_chunk(
                        wav_segment,
                        input_lang,
                        historical_audio_embeddings,
                        historical_text_tokens,
                    )

                    if not text.strip():
                        advance = chunk_len - overlap_drop_sec
                        print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                              f"empty transcript — advancing {advance:.2f}s (fixed stride)")
                        chunk_start += advance
                        chunk_idx += 1
                        continue

                    # Align current chunk's audio vs its own transcription only
                    print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                          f"aligning audio ↔ transcript (local chunk only, no history) ...")
                    chunk_ts: List[Dict[str, Any]] = []
                    try:
                        chunk_ts = align_llm(self, wav_segment, text, input_lang)
                        print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                              f"alignment returned {len(chunk_ts)} word timestamps")
                        if chunk_ts:
                            print(f"[transcribe]   first={chunk_ts[0]}  last={chunk_ts[-1]}")
                    except Exception as e:
                        log.warning(
                            f"Alignment failed for streaming chunk at {chunk_start:.2f}s "
                            f"of input {idx}: {e}"
                        )
                        print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                              f"alignment FAILED: {e}")

                    if not is_last_chunk and chunk_ts:
                        safe_threshold = (chunk_end - chunk_start) - overlap_drop_sec
                        safe_ts = [w for w in chunk_ts if w["end"] <= safe_threshold]
                        n_dropped = len(chunk_ts) - len(safe_ts)

                        print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                              f"safe_threshold={safe_threshold:.2f}s  "
                              f"safe={len(safe_ts)}  dropped={n_dropped}")

                        if safe_ts:
                            committed_ts = safe_ts
                            next_start = chunk_start + safe_ts[-1]["end"]
                            safe_duration_sec = safe_ts[-1]["end"]
                            print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                                  f"committing {len(committed_ts)} words  "
                                  f"next_start={next_start:.2f}s  "
                                  f"safe_duration={safe_duration_sec:.2f}s")
                        else:
                            # Fallback: no safe words — commit all, advance by fixed stride
                            committed_ts = chunk_ts
                            next_start = chunk_start + chunk_len - overlap_drop_sec
                            safe_duration_sec = chunk_len - overlap_drop_sec
                            print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                                  f"FALLBACK: no safe words found — committing all {len(committed_ts)} words  "
                                  f"fixed stride next_start={next_start:.2f}s")
                            log.debug(
                                f"No safe words found at chunk_start={chunk_start:.2f}s; "
                                f"falling back to fixed stride."
                            )
                    else:
                        # Last chunk (or no timestamps) — commit everything
                        committed_ts = chunk_ts
                        next_start = duration
                        safe_duration_sec = chunk_end - chunk_start
                        reason = "last chunk" if is_last_chunk else "no timestamps from alignment"
                        print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                              f"committing all {len(committed_ts)} words ({reason})")

                    # Build committed text from committed timestamps
                    is_word_level = committed_ts and "word" in committed_ts[0]
                    word_parts = [w.get("word", w.get("char", "")) for w in committed_ts]
                    committed_text = " ".join(word_parts) if is_word_level else "".join(word_parts)

                    print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                          f"committed_text={repr(committed_text[:80])}")

                    # Adjust timestamps to global timeline and record
                    for w in committed_ts:
                        input_timestamps.append({
                            "word": w.get("word", w.get("char", "")),
                            "start": w["start"] + chunk_start,
                            "end": w["end"] + chunk_start,
                        })
                    print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] "
                          f"global timestamps appended  total_so_far={len(input_timestamps)}")

                    if committed_text.strip():
                        input_text_parts.append(committed_text)
                        print(f"[transcribe] input[{idx}] chunk[{chunk_idx}] updating history ...")
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
                    chunk_idx += 1

                full_transcript = " ".join(t for t in input_text_parts if t.strip())
                print(f"\n[transcribe] input[{idx}] STREAMING DONE  "
                      f"chunks_processed={chunk_idx}  total_words={len(input_timestamps)}")
                print(f"[transcribe] input[{idx}] full transcript: {repr(full_transcript[:120])}")
                final_transcripts.append(full_transcript)
                final_timestamps.append(input_timestamps)
                continue

            # ----------------------------------------------------------------
            # Non-streaming path
            # ----------------------------------------------------------------
            print(f"\n[transcribe] input[{idx}] → NON-STREAMING path")

            if chunk_len is not None and duration > chunk_len:
                print(f"[transcribe] input[{idx}] splitting into non-overlapping chunks "
                      f"(chunk_len={chunk_len}s) ...")
                chunks = chunk_waveform(waveform, 16000, chunk_len)
                print(f"[transcribe] input[{idx}] got {len(chunks)} chunks")
            else:
                if duration > MAX_ALLOWED_AUDIO_SEC and chunk_len is None:
                    raise ValueError(
                        f"Audio {idx} duration {duration:.2f}s > {MAX_ALLOWED_AUDIO_SEC}s. "
                        f"Provide chunk_len parameter."
                    )
                print(f"[transcribe] input[{idx}] single chunk (duration={duration:.2f}s)")
                chunks = [(waveform, 0.0)]

            input_text_parts = []
            input_timestamps = []

            chunk_waveforms = [c[0] for c in chunks]
            offsets = [c[1] for c in chunks]

            for i in range(0, len(chunk_waveforms), batch_size):
                batch_wavs = chunk_waveforms[i : i + batch_size]
                batch_offsets = offsets[i : i + batch_size]

                print(f"\n[transcribe] input[{idx}] non-streaming batch[{i//batch_size}]  "
                      f"chunks {i}–{i+len(batch_wavs)-1}  "
                      f"offsets={[f'{o:.2f}s' for o in batch_offsets]}")

                batch_data = [(w, input_lang) for w in batch_wavs]
                seq2seq_batch = self._create_batch_simple(batch_data)
                texts = self._apply_model(seq2seq_batch)

                for j, text in enumerate(texts):
                    wav_segment = batch_wavs[j]
                    offset = batch_offsets[j]

                    print(f"[transcribe] input[{idx}] chunk[{i+j}] offset={offset:.2f}s  "
                          f"raw_text={repr(text[:80])}")

                    if not text.strip():
                        print(f"[transcribe] input[{idx}] chunk[{i+j}] empty — skipping alignment")
                        input_text_parts.append("")
                        continue

                    print(f"[transcribe] input[{idx}] chunk[{i+j}] aligning ...")
                    chunk_ts = []
                    try:
                        if isinstance(self.model, Wav2Vec2AsrModel):
                            chunk_ts = align_ctc(self.model, wav_segment, 16000, text)
                        elif isinstance(self.model, Wav2Vec2LlamaModel):
                            chunk_ts = align_llm(self, wav_segment, text, input_lang)
                        print(f"[transcribe] input[{idx}] chunk[{i+j}] "
                              f"alignment: {len(chunk_ts)} timestamps")
                        if chunk_ts:
                            print(f"[transcribe]   first={chunk_ts[0]}  last={chunk_ts[-1]}")
                    except Exception as e:
                        log.warning(
                            f"Alignment failed for chunk {i + j} of input {idx}: {e}"
                        )
                        print(f"[transcribe] input[{idx}] chunk[{i+j}] alignment FAILED: {e}")
                        chunk_ts = []

                    for w in chunk_ts:
                        input_timestamps.append({
                            "word": w.get("word", w.get("char", "")),
                            "start": w["start"] + offset,
                            "end": w["end"] + offset,
                        })

                    input_text_parts.append(text)

            full_transcript = " ".join(t for t in input_text_parts if t.strip())
            print(f"\n[transcribe] input[{idx}] NON-STREAMING DONE  "
                  f"total_words={len(input_timestamps)}")
            print(f"[transcribe] input[{idx}] full transcript: {repr(full_transcript[:120])}")
            final_transcripts.append(full_transcript)
            final_timestamps.append(input_timestamps)

        print(f"\n{'='*60}")
        print(f"[transcribe] ALL DONE  n_outputs={len(final_transcripts)}")
        print(f"{'='*60}\n")
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
        Transcribes AudioInput using zero-shot context conditioning.
        Only works for the omniASR_LLM_7B_ZS model.
        """
        print(f"\n{'='*60}")
        print(f"[transcribe_with_context] called  n_inputs={len(inp)}  batch_size={batch_size}")

        if len(inp) == 0:
            print(f"[transcribe_with_context] empty input — returning early")
            return []

        # fmt: off
        is_ctc_model = isinstance(self.model, Wav2Vec2AsrModel)
        is_llm_model = isinstance(self.model, Wav2Vec2LlamaModel)

        if is_ctc_model:
            raise NotImplementedError("CTC models do not support context conditioning. Please use `.transcribe()` instead of `.transcribe_with_context()`.")
        if is_llm_model and self.model.model_type != ModelType.ZERO_SHOT:
            raise NotImplementedError("LLM models do not support context conditioning. Please use `.transcribe()` instead of `.transcribe_with_context()`.")

        assert len(inp) == len(context_examples), \
            f"Number of audio inputs ({len(inp)}) must match number of context examples {len(context_examples)}."

        for i, examples in enumerate(context_examples):
            assert len(examples) > 0, f"Input index {i} has no context examples, but needs at least one."
            if len(examples) > 10:
                log.info(f"Found {len(examples)} context examples for input index {i}, but can only process 10. Ignoring extra.")
                print(f"[transcribe_with_context] input[{i}]: {len(examples)} examples → capping to 10")
        # fmt: on

        max_context_example_per_sample = 10
        context_examples = repeat_to_max_len(
            context_examples, max_len=max_context_example_per_sample
        )
        print(f"[transcribe_with_context] context padded/trimmed to {max_context_example_per_sample} per input")

        print(f"[transcribe_with_context] building zipped audio+context pipeline ...")
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
        print(f"[transcribe_with_context] running pipeline ...")
        results = list(combined_builder.and_return())
        print(f"[transcribe_with_context] done  n_outputs={len(results)}")
        for i, r in enumerate(results):
            print(f"[transcribe_with_context] output[{i}]: {repr(r[:100])}")
        print(f"{'='*60}\n")
        return results