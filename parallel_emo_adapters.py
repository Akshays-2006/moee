import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
)

@dataclass
class EmotionAdapterConfig:
    # Emotion space — matches GoEmotions top-5 or any custom grouping
    num_emotions: int = 5
    emotion_labels: List[str] = field(default_factory=lambda: [
        "joy", "sadness", "anger", "fear", "surprise"
    ])

    # LoRA hyperparams
    lora_r: int = 8
    lora_alpha: int = 16          # scaling = alpha / r
    lora_dropout: float = 0.05

    # Ablation flags (matches your sketch: encoder-only / decoder-only / both)
    inject_encoder: bool = True
    inject_decoder: bool = True
    inject_cross_attention: bool = False  # cross-attn in decoder (optional)

    # Model names
    t5_model_name: str = "google/flan-t5-base"
    roberta_model_name: str = "roberta-base"

    # Training
    freeze_t5_base: bool = True    # only LoRA params + classifier are trained
    freeze_classifier: bool = False  # set True to use pre-fine-tuned classifier

class EmotionLoRALayer(nn.Module):
    """
    E parallel low-rank adapters for a single linear projection.

    For emotion e with weight p_e:
        delta_e(x) = (dropout(x) @ A_e.T @ B_e.T) * scaling

    Soft-gated output (no loop — single batched einsum):
        delta(x, p) = Σ_e  p_e · delta_e(x)

    Weight init: A ~ Kaiming, B = 0  →  delta = 0 at initialisation,
    so the model starts identical to the frozen base.
    """

    def __init__(self, in_features: int, out_features: int, cfg: EmotionAdapterConfig):
        super().__init__()
        self.E = cfg.num_emotions
        self.r = cfg.lora_r
        self.scaling = cfg.lora_alpha / cfg.lora_r
        self.dropout = nn.Dropout(cfg.lora_dropout)

        # Stacked matrices: (E, r, D_in)  and  (E, D_out, r)
        self.lora_A = nn.Parameter(torch.empty(self.E, self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.E, out_features, self.r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero → delta starts at 0 ✓

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x       : (B, S, D_in)   input activations
        weights : (B, E)         soft emotion probabilities from classifier

        Returns
        -------
        delta   : (B, S, D_out)  soft-gated LoRA correction to add to base output
        """
        B, S, D_in = x.shape
        x_d = self.dropout(x).view(B * S, D_in)   # (B*S, D_in)

        # Step 1 — project down:  (B*S, D_in) × (E, D_in, r).T → (B*S, E, r)
        A_T = self.lora_A.permute(0, 2, 1)                         # (E, D_in, r)
        inter = torch.einsum("bi,eir->ber", x_d, A_T)              # (B*S, E, r)

        # Step 2 — project up:   (B*S, E, r) × (E, D_out, r) → (B*S, E, D_out)
        delta = torch.einsum("ber,eor->beo", inter, self.lora_B)    # (B*S, E, D_out)
        delta = (delta * self.scaling).view(B, S, self.E, -1)       # (B, S, E, D_out)

        # Step 3 — soft gate over emotions
        w = weights.unsqueeze(1).unsqueeze(-1)   # (B, 1, E, 1) — broadcast over S, D_out
        return (delta * w).sum(dim=2)             # (B, S, D_out)

class EmotionAdaptedLinear(nn.Module):
    """
    Wraps a frozen nn.Linear with an EmotionLoRALayer.

        y = base(x) + lora(x, emotion_weights)

    emotion_weights is injected via set_emotion_weights() before the T5
    forward pass — avoids threading it through every T5 internal call.
    """

    def __init__(self, base_linear: nn.Linear, cfg: EmotionAdapterConfig):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora = EmotionLoRALayer(
            base_linear.in_features, base_linear.out_features, cfg
        )
        self._emotion_weights: Optional[torch.Tensor] = None

    def set_emotion_weights(self, weights: torch.Tensor):
        self._emotion_weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._emotion_weights is not None:
            out = out + self.lora(x, self._emotion_weights)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Emotion Classifier  (RoBERTa → soft distribution over E emotions)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionClassifier(nn.Module):
    """
    RoBERTa fine-tuned on GoEmotions.
    Returns a soft probability distribution  p ∈ Δ^E  via softmax.

    Training note
    -------------
    Fine-tune separately on GoEmotions with BCE loss (multi-label) first,
    then optionally keep training end-to-end with the adapter model.
    Your sketch mentions BCE loss / KL-Div — use KL-Div when you have a
    target soft distribution; BCE when you have binary multi-label targets.
    """

    def __init__(self, cfg: EmotionAdapterConfig):
        super().__init__()
        self.encoder = RobertaForSequenceClassification.from_pretrained(
            cfg.roberta_model_name,
            num_labels=cfg.num_emotions,
            ignore_mismatched_sizes=True,
        )
        if cfg.freeze_classifier:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, E) soft emotion probabilities."""
        logits = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits                             # (B, E)
        return F.softmax(logits, dim=-1)     # probabilities sum to 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class SoftEmotionAdaptedT5(nn.Module):
    """
    Full pipeline:

        1. RoBERTa  →  p = softmax(logits)          (B, E)
        2. Broadcast p to all EmotionAdaptedLinear layers
        3. T5 forward with soft-gated LoRA at every q/k/v projection
        4. Return loss + logits + emotion_weights (for analysis)

    Only parameters that require grad:
        - LoRA A and B matrices   (injected into T5 attention layers)
        - RoBERTa classifier head (unless freeze_classifier=True)
    """

    def __init__(self, cfg: EmotionAdapterConfig):
        super().__init__()
        self.cfg = cfg

        # ── Base T5 ──────────────────────────────────────────────────────────
        self.t5 = T5ForConditionalGeneration.from_pretrained(cfg.t5_model_name)
        if cfg.freeze_t5_base:
            for p in self.t5.parameters():
                p.requires_grad_(False)

        # ── Emotion Classifier ───────────────────────────────────────────────
        self.emotion_clf = EmotionClassifier(cfg)

        # ── Inject LoRA into attention projections ───────────────────────────
        self._adapted_layers: List[EmotionAdaptedLinear] = []
        self._inject_adapters()

        # Sanity check
        self.print_trainable_summary()

    # ── Adapter injection ─────────────────────────────────────────────────────

    def _inject_adapters(self):
        """
        Walk T5 encoder/decoder blocks and replace q/k/v Linear layers
        with EmotionAdaptedLinear wrappers.

        T5 block structure (both encoder and decoder):
            block[i].layer[0].SelfAttention    ← q, k, v, o
            block[i].layer[1].EncDecAttention  ← cross-attn (decoder only)
            block[i].layer[-1]                 ← FFN
        """
        if self.cfg.inject_encoder:
            for block in self.t5.encoder.block:
                self._replace_qkv(block.layer[0].SelfAttention)

        if self.cfg.inject_decoder:
            for block in self.t5.decoder.block:
                self._replace_qkv(block.layer[0].SelfAttention)
                if self.cfg.inject_cross_attention:
                    self._replace_qkv(block.layer[1].EncDecAttention)

    def _replace_qkv(self, attn_module: nn.Module):
        """Replace q, k, v projections in a T5Attention module."""
        for proj_name in ["q", "k", "v"]:
            original_linear = getattr(attn_module, proj_name)
            adapted = EmotionAdaptedLinear(original_linear, self.cfg)
            setattr(attn_module, proj_name, adapted)
            self._adapted_layers.append(adapted)

    # ── Emotion weight broadcast ──────────────────────────────────────────────

    def _broadcast_weights(self, weights: torch.Tensor):
        """Push emotion weights to every adapted layer before T5 forward."""
        for layer in self._adapted_layers:
            layer.set_emotion_weights(weights)

    def _clear_weights(self):
        """Optionally clear after forward to avoid stale state."""
        for layer in self._adapted_layers:
            layer.set_emotion_weights(None)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        # T5 inputs
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # Emotion classifier inputs (may differ in tokenizer from T5)
        emotion_input_ids: Optional[torch.Tensor] = None,
        emotion_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # 1. Emotion distribution
        emo_ids  = emotion_input_ids     if emotion_input_ids     is not None else input_ids
        emo_mask = emotion_attention_mask if emotion_attention_mask is not None else attention_mask
        emotion_weights = self.emotion_clf(emo_ids, emo_mask)   # (B, E)

        # 2. Broadcast to LoRA layers
        self._broadcast_weights(emotion_weights)

        # 3. T5 forward
        out = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        # 4. Clear (optional — prevents state leak across batches)
        self._clear_weights()

        return {
            "loss":            out.loss,
            "logits":          out.logits,
            "emotion_weights": emotion_weights,   # (B, E) — for analysis / logging
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        emotion_input_ids: Optional[torch.Tensor] = None,
        emotion_attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (generated_ids, emotion_weights).
        Pass max_new_tokens, num_beams, etc. via generate_kwargs.
        """
        emo_ids  = emotion_input_ids     if emotion_input_ids     is not None else input_ids
        emo_mask = emotion_attention_mask if emotion_attention_mask is not None else attention_mask

        emotion_weights = self.emotion_clf(emo_ids, emo_mask)
        self._broadcast_weights(emotion_weights)

        generated = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        self._clear_weights()
        return generated, emotion_weights

    # ── Utilities ─────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        lora_only = sum(
            p.numel()
            for layer in self._adapted_layers
            for p in layer.lora.parameters()
        )
        print(f"─── Trainable parameter summary ───────────────────────")
        print(f"  Total params   : {total:>12,}")
        print(f"  Trainable      : {trainable:>12,}  ({100*trainable/total:.2f}%)")
        print(f"  LoRA only      : {lora_only:>12,}  ({100*lora_only/total:.2f}%)")
        print(f"  Adapted layers : {len(self._adapted_layers)}")
        print(f"───────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Emotion Entropy Utility  (your sketch: temperature / entropy analysis)
# ─────────────────────────────────────────────────────────────────────────────

def emotion_entropy(weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Shannon entropy of the emotion distribution.

    High entropy  → ambiguous / blended emotion (e.g. sarcasm)
    Low entropy   → clear single emotion

    Args
        weights : (B, E) soft emotion probabilities
    Returns
        entropy : (B,)   in nats  [0, log(E)]
    """
    return -(weights * (weights + eps).log()).sum(dim=-1)


def emotion_temperature_scale(
    weights: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    Re-scale emotion logits with temperature before softmax.
    temperature < 1 → sharper (more committed to dominant emotion)
    temperature > 1 → softer  (more spread across emotions)

    Useful at inference time to control emotional specificity.
    """
    log_p = weights.log()
    return F.softmax(log_p / temperature, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Minimal training loop sketch  (EmpatheticDialogues)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SoftEmotionAdaptedT5,
    dataloader,            # yields batches from EmpatheticDialogues
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
) -> float:
    """
    Minimal training loop. The dataloader should yield dicts with keys:
        t5_input_ids, t5_attention_mask, labels,
        emo_input_ids, emo_attention_mask
    (T5 and RoBERTa use different tokenizers, so pass both.)
    """
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            input_ids            = batch["t5_input_ids"],
            attention_mask       = batch["t5_attention_mask"],
            labels               = batch["labels"],
            emotion_input_ids    = batch["emo_input_ids"],
            emotion_attention_mask = batch["emo_attention_mask"],
        )

        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % log_every == 0:
            ent = emotion_entropy(out["emotion_weights"]).mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | mean emotion entropy {ent:.3f} nats")

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
random_seed = 42
torch.manual_seed(random_seed)

if __name__ == "__main__":
    cfg = EmotionAdapterConfig(
        t5_model_name      = "google/flan-t5-base",
        roberta_model_name = "roberta-base",
        num_emotions       = 5,
        lora_r             = 8,
        lora_alpha         = 16,
        inject_encoder     = True,
        inject_decoder     = False,   # ← change for ablation
        freeze_t5_base     = True,
        freeze_classifier  = False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SoftEmotionAdaptedT5(cfg).to(device)

    # Dummy batch — replace with real EmpatheticDialogues loader
    B, S = 2, 32
    dummy = {
        "input_ids":              torch.randint(0, 32128, (B, S),).to(device),
        "attention_mask":         torch.ones(B, S, dtype=torch.long).to(device),
        "labels":                 torch.randint(0, 32128, (B, S)).to(device),
        "emotion_input_ids":      torch.randint(0, 50265, (B, S)).to(device),
        "emotion_attention_mask": torch.ones(B, S, dtype=torch.long).to(device),
    }

    out = model(
        input_ids              = dummy["input_ids"],
        attention_mask         = dummy["attention_mask"],
        labels                 = dummy["labels"],
        emotion_input_ids      = dummy["emotion_input_ids"],
        emotion_attention_mask = dummy["emotion_attention_mask"],
    )

    print(f"\nLoss  : {out['loss'].item():.4f}")
    print(f"Logits: {out['logits'].shape}")
    print(f"Emotion weights (batch 0): {out['emotion_weights'][0].detach().cpu().numpy().round(3)}")
    print(f"Entropy (batch 0): {emotion_entropy(out['emotion_weights'])[0].item():.3f} nats")

    # Temperature scaling demo
    sharp  = emotion_temperature_scale(out["emotion_weights"], temperature=0.3)
    diffuse = emotion_temperature_scale(out["emotion_weights"], temperature=2.0)
    print(f"\nSharp  (T=0.3): {sharp[0].detach().cpu().numpy().round(3)}")
    print(f"Diffuse (T=2.0): {diffuse[0].detach().cpu().numpy().round(3)}")