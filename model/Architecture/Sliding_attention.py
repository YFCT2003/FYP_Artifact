# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：Sliding_attention.py
@Date：2026/4/9 17:33
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM


# ==========================================
# 0. Common utility: remove special tokens
# ==========================================
def remove_special_tokens(hidden_states, input_ids, tokenizer):
    """
    Remove CLS, EOS, PAD tokens, keep only actual amino acid residue embeddings.
    Args:
        hidden_states: [B, seq_len, d] hidden states output by ESM
        input_ids:     [B, seq_len]    input token ids
        tokenizer:     EsmTokenizer instance
    Returns:
        padded: [B, max_real_len, d]  embeddings after removing special tokens (re-padded)
        mask:   [B, max_real_len]     True = real residue, False = padding
    """
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    special_mask = (input_ids != cls_id) & (input_ids != eos_id) & (input_ids != pad_id)

    results = []
    for i in range(hidden_states.size(0)):
        real_tokens = hidden_states[i][special_mask[i]]  # [real_len, d]
        results.append(real_tokens)

    padded = nn.utils.rnn.pad_sequence(results, batch_first=True, padding_value=0.0)
    mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=hidden_states.device)
    for i, r in enumerate(results):
        mask[i, :len(r)] = True

    return padded, mask


# ==========================================
# 1. pHLA Encoder (fine-tuned ESM)
# ==========================================
class PHLA_encoder(nn.Module):
    """
    Load the ESM model fine-tuned with Epitope MLM and extract pHLA residue-level embeddings.
    Input format is the paired encoding from tokenizer(hla_seq, epitope_seq).
    Output pure amino acid embeddings after removing all special tokens.
    """

    def __init__(self, model_path, model_name, freeze=True):
        super().__init__()
        # Load fine-tuned weights: first build a same-architecture model, then load the weights
        esm_mlm = EsmForMaskedLM.from_pretrained(model_name)

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Compatible with PHLAForMLM wrapper format (esm_mlm. prefix) and DDP format (module. prefix)
        state_dict = {k.replace('module.', '').replace('esm_mlm.', ''): v for k, v in state_dict.items()}
        esm_mlm.load_state_dict(state_dict)

        # Keep only the encoder backbone, discard the MLM head
        self.encoder = esm_mlm.esm
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      [B, seq_len]  tokenizer output
            attention_mask:  [B, seq_len]  tokenizer output
        Returns:
            embed: [B, L_phla, d_esm]  pure residue embeddings
            mask:  [B, L_phla]         True = real residue
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, seq_len, d_esm]
        return remove_special_tokens(hidden, input_ids, self.tokenizer)


# ==========================================
# 2. TCR Encoder (pre-trained ESM, not fine-tuned)
# ==========================================
class TCR_encoder(nn.Module):
    """
    Load pre-trained ESM model, extract CDR3beta residue-level embeddings.
    Input is a single CDR3beta sequence.
    """

    def __init__(self, model_name, freeze=True):
        super().__init__()
        self.encoder = EsmModel.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask:  [B, seq_len]
        Returns:
            embed: [B, L_tcr, d_esm]  pure residue embeddings
            mask:  [B, L_tcr]         True = real residue
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return remove_special_tokens(hidden, input_ids, self.tokenizer)


# ==========================================
# 3. Sliding Cross-Attention
# ==========================================
class SlidingCrossAttention(nn.Module):
    """
    Dual-stream sliding cross-attention.
    Stream 1 (forward): TCR (query) -> pHLA (key/value), TCR aggregates information from pHLA
    Stream 2 (reverse): pHLA (query) -> TCR (key/value), pHLA aggregates information from TCR
    Both streams share the spatial bias (transpose relationship), each has independent projection weights.

    Args:
        d_in:     input embedding dimension (ESM hidden size)
        d_model:  internal working dimension for attention
        n_heads:  number of multi-head attention heads (d_model must be divisible by n_heads)
        L_Q:      TCR maximum sequence length (after removing special tokens)
        L_K:      pHLA maximum sequence length (after removing special tokens)
        sigma:    Gaussian kernel bandwidth
        n_iter:   number of sliding iterations
        d_ff:     feed-forward network hidden layer dimension
    """

    def __init__(self, d_in, d_model, n_heads, L_Q, L_K, sigma=1.0, window_T=None, n_iter=3, d_ff=512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sigma = sigma
        self.window_T = window_T  # hard truncation threshold, None means not used, recommended value 0.1~0.2
        self.n_iter = n_iter

        # Input projection: d_in -> d_model (skip when dimensions are the same, preserve ESM embedding semantics)
        self.tcr_input_proj = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)
        self.phla_input_proj = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)

        # Stream 1 (forward): TCR -> pHLA
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Stream 2 (reverse): pHLA -> TCR
        self.W_Q_v = nn.Linear(d_model, d_model, bias=False)
        self.W_K_v = nn.Linear(d_model, d_model, bias=False)
        self.W_V_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj_v = nn.Linear(d_model, d_model, bias=False)

        # Spatial coordinates
        self.L_Q = L_Q
        self.L_K = L_K
        # pHLA coordinates are fixed
        self.register_buffer('k_pos', torch.arange(L_K).float())

        # LayerNorm + FFN for stream 1
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # LayerNorm + FFN for stream 2
        self.norm1_v = nn.LayerNorm(d_model)
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2_v = nn.LayerNorm(d_model)

    def compute_init_pos(self, tcr_mask):
        """
        Dynamically generate initial positions at the 1/3 mark according to the actual length of each sequence.
        The middle 1/3 of residues are mapped to [0, L_K-1], both ends are set to -10 (far from pHLA).
        Args:
            tcr_mask: [B, L_Q]  True = real residue
        Returns:
            q_pos: [B, L_Q]  initial coordinates independent for each sequence
        """
        B, L_Q = tcr_mask.shape
        seq_lens = tcr_mask.sum(-1).float()  # [B]
        pos = torch.arange(L_Q, device=tcr_mask.device).float().unsqueeze(0)  # [1, L_Q]

        starts = (seq_lens / 3).unsqueeze(1)   # [B, 1]
        ends = (seq_lens * 2 / 3).unsqueeze(1)  # [B, 1]

        # Linearly map inside the middle segment to [0, L_K-1]
        normalized = (pos - starts) / (ends - starts).clamp(min=1)
        q_pos = normalized * (self.L_K - 1)

        # Both ends and padding are set to far away
        outside = (pos < starts) | (pos >= ends) | (~tcr_mask)
        q_pos = q_pos.masked_fill(outside, -10.0)

        return q_pos

    def compute_space(self, q_pos):
        """
        Gaussian kernel spatial bias matrix, with optional hard truncation.
        Args:
            q_pos: [B, L_Q] per-sample coordinates
        Returns:
            space: [B, L_Q, L_K]
        """
        q = q_pos.unsqueeze(-1)            # [B, L_Q, 1]
        k = self.k_pos.view(1, 1, -1)     # [1, 1, L_K]
        space = torch.exp(-(q - k) ** 2 / (2 * self.sigma ** 2))  # [B, L_Q, L_K]

        # Hard truncation: positions below window_T are set to 0 (do not participate in attention bias)
        if self.window_T is not None:
            space = space.masked_fill(space < self.window_T, 0.0)

        return space

    def forward(self, tcr_embed, phla_embed, tcr_mask=None, phla_mask=None):
        """
        Dual-stream sliding cross-attention.
        Args:
            tcr_embed:   [B, L_Q, d_in]   TCR residue embeddings
            phla_embed:  [B, L_K, d_in]   pHLA residue embeddings
            tcr_mask:    [B, L_Q]          True = real residue (optional)
            phla_mask:   [B, L_K]          True = real residue (optional)
        Returns:
            tcr_output:  [B, L_Q, d_model]  updated TCR representation (fused with pHLA information)
            phla_output: [B, L_K, d_model]  updated pHLA representation (fused with TCR information)
            attn:        [B, n_heads, L_Q, L_K]  forward attention weights
        """
        B = tcr_embed.size(0)

        # ---- Input projection ----
        tcr_proj = self.tcr_input_proj(tcr_embed)     # [B, L_Q, d_model]
        phla_proj = self.phla_input_proj(phla_embed)   # [B, L_K, d_model]
        residual_tcr = tcr_proj
        residual_phla = phla_proj

        # ---- Stream 1 (forward): TCR queries pHLA ----
        Q = self.W_Q(tcr_proj).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)   # [B, H, L_Q, d_head]
        K = self.W_K(phla_proj).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L_K, d_head]
        V = self.W_V(phla_proj).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L_K, d_head]

        # ---- Stream 2 (reverse): pHLA queries TCR ----
        Q_v = self.W_Q_v(phla_proj).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L_K, d_head]
        K_v = self.W_K_v(tcr_proj).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)   # [B, H, L_Q, d_head]
        V_v = self.W_V_v(tcr_proj).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)   # [B, H, L_Q, d_head]

        # ---- Sequence attention ----
        amino_attn = Q @ K.transpose(-1, -2) / (self.d_head ** 0.5)      # [B, H, L_Q, L_K]
        amino_attn_v = Q_v @ K_v.transpose(-1, -2) / (self.d_head ** 0.5)  # [B, H, L_K, L_Q]

        # padding mask (masked positions do not compute attn)
        if phla_mask is not None:
            pad_mask_phla = ~phla_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_K]
            amino_attn = amino_attn.masked_fill(pad_mask_phla, -np.inf)
        if tcr_mask is not None:
            pad_mask_tcr = ~tcr_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_Q]
            amino_attn_v = amino_attn_v.masked_fill(pad_mask_tcr, -np.inf)

        # ---- Spatial bias + iterative sliding (per-sample positions) ----
        q_pos = self.compute_init_pos(tcr_mask)  # [B, L_Q]
        space_attn = self.compute_space(q_pos)                      # [B, L_Q, L_K]

        for _ in range(self.n_iter):
            combined = amino_attn + space_attn.unsqueeze(1)  # [B, H, L_Q, L_K]
            weights = torch.softmax(combined, dim=-1)
            weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

            new_pos = (weights * self.k_pos.view(1, 1, 1, -1)).sum(-1)  # [B, H, L_Q]
            att_sum = weights.sum(-1)                                    # [B, H, L_Q]
            new_pos = new_pos + (1 - att_sum) * q_pos.unsqueeze(1)
            q_pos = new_pos.mean(dim=1)  # [B, L_Q]
            space_attn = self.compute_space(q_pos)

        # ---- Final attention (both streams share spatial bias, transpose relationship) ----
        # Stream 1: TCR -> pHLA
        final_scores = amino_attn + space_attn.unsqueeze(1)
        final_scores = torch.clamp(final_scores, min=-50, max=50)  # exp(50)~5e21, safely far from overflow
        if phla_mask is not None:
            final_scores = final_scores.masked_fill(pad_mask_phla, -np.inf)

        attn = torch.softmax(final_scores, dim=-1)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)

        # Stream 2: pHLA -> TCR (spatial bias transposed)
        final_scores_v = amino_attn_v + space_attn.transpose(-1, -2).unsqueeze(1)  # [B, H, L_K, L_Q]
        final_scores_v = torch.clamp(final_scores_v, min=-50, max=50)
        if tcr_mask is not None:
            final_scores_v = final_scores_v.masked_fill(pad_mask_tcr, -np.inf)

        attn_v = torch.softmax(final_scores_v, dim=-1)
        attn_v = torch.where(torch.isnan(attn_v), torch.zeros_like(attn_v), attn_v)

        # ---- Stream 1: aggregate pHLA -> update TCR ----
        att_sum_fwd = attn.sum(-1).unsqueeze(-1)  # [B, H, L_Q, 1]
        context = attn @ V + (1 - att_sum_fwd) * Q
        context = context * att_sum_fwd
        context = context.transpose(1, 2).reshape(B, -1, self.d_model)
        tcr_output = self.out_proj(context)
        tcr_output = self.norm1(tcr_output + residual_tcr)
        tcr_output = self.norm2(self.ffn(tcr_output) + tcr_output)

        # ---- Stream 2: aggregate TCR -> update pHLA ----
        att_sum_rev = attn_v.sum(-1).unsqueeze(-1)  # [B, H, L_K, 1]
        context_v = attn_v @ V_v + (1 - att_sum_rev) * Q_v
        context_v = context_v * att_sum_rev
        context_v = context_v.transpose(1, 2).reshape(B, -1, self.d_model)
        phla_output = self.out_proj_v(context_v)
        phla_output = self.norm1_v(phla_output + residual_phla)
        phla_output = self.norm2_v(self.ffn_v(phla_output) + phla_output)

        return tcr_output, phla_output, attn


# ==========================================
# 4. Top-level model
# ==========================================
class PLMSlidingModel(nn.Module):
    """
    Complete pHLA-TCR binding prediction model (dual-stream).
    pHLA encoder (fine-tuned ESM) + TCR encoder (pre-trained ESM) + dual-stream Sliding Cross-Attention + classification head

    Outputs:
        - logits: binding prediction
        - phla_output: pHLA representation fused with TCR interaction information (for downstream use)
    """

    def __init__(self, phla_model_path, phla_model_name, tcr_model_name,
                 d_model=256, n_heads=8, L_Q=23, L_K=43,
                 sigma=1.0, window_T=None, n_iter=3, d_ff=None,
                 freeze_phla=True, freeze_tcr=True, classifier_dropout=0.0):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # Encoders
        self.phla_encoder = PHLA_encoder(phla_model_path, phla_model_name, freeze=freeze_phla)
        self.tcr_encoder = TCR_encoder(tcr_model_name, freeze=freeze_tcr)

        # Get ESM hidden layer dimension
        d_esm_phla = self.phla_encoder.encoder.config.hidden_size
        d_esm_tcr = self.tcr_encoder.encoder.config.hidden_size

        # Dual-stream Sliding Cross-Attention
        self.sliding_attn = SlidingCrossAttention(
            d_in=d_esm_tcr,
            d_model=d_model,
            n_heads=n_heads,
            L_Q=L_Q,
            L_K=L_K,
            sigma=sigma,
            window_T=window_T,
            n_iter=n_iter,
            d_ff=d_ff
        )
        # If pHLA and TCR use different sizes of ESM, need to separately project pHLA
        if d_esm_phla != d_esm_tcr:
            self.phla_proj = nn.Linear(d_esm_phla, d_esm_tcr)
        else:
            self.phla_proj = nn.Identity()

        # Classification head: concatenate pooled representations of TCR and pHLA
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 2)
        )

    @staticmethod
    def _pad_or_truncate(embed, mask, target_len):
        """
        Pad or truncate [B, L, d] embeddings to a fixed length target_len, and update the mask accordingly.
        Ensure that the input dimensions received by SlidingCrossAttention match L_Q / L_K.
        """
        B, L, d = embed.shape
        if L >= target_len:
            return embed[:, :target_len, :], mask[:, :target_len]
        pad_len = target_len - L
        embed = torch.cat([embed, torch.zeros(B, pad_len, d, device=embed.device)], dim=1)
        mask = torch.cat([mask, torch.zeros(B, pad_len, dtype=torch.bool, device=mask.device)], dim=1)
        return embed, mask

    def _masked_mean_pool(self, embed, mask):
        """Take average over real residues, ignore padding"""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            return (embed * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        return embed.mean(1)

    def forward(self, phla_input_ids, phla_attention_mask, tcr_input_ids, tcr_attention_mask,
                return_attn=False):
        """
        Args:
            phla_input_ids:      [B, seq_len]
            phla_attention_mask:  [B, seq_len]
            tcr_input_ids:       [B, seq_len]
            tcr_attention_mask:   [B, seq_len]
            return_attn:         bool, set to True during inference to return the attention matrix for interpretability analysis
        Returns:
            logits:      [B, 2]                      binding / non-binding
            phla_output: [B, L_K, d_model]           pHLA representation fused with TCR information
            attn:        [B, n_heads, L_Q, L_K]      only returned when return_attn=True
        """
        L_Q = self.sliding_attn.L_Q
        L_K = self.sliding_attn.L_K

        # Encoding
        phla_embed, phla_mask = self.phla_encoder(phla_input_ids, phla_attention_mask)
        tcr_embed, tcr_mask = self.tcr_encoder(tcr_input_ids, tcr_attention_mask)

        # Align to the fixed length required by SlidingCrossAttention
        tcr_embed, tcr_mask = self._pad_or_truncate(tcr_embed, tcr_mask, L_Q)
        phla_embed, phla_mask = self._pad_or_truncate(phla_embed, phla_mask, L_K)

        # Align dimensions
        phla_embed = self.phla_proj(phla_embed)

        # Dual-stream Sliding Cross-Attention
        tcr_output, phla_output, attn = self.sliding_attn(
            tcr_embed, phla_embed, tcr_mask, phla_mask
        )

        # Classification: concatenate pooled representations from both sides
        tcr_pooled = self._masked_mean_pool(tcr_output, tcr_mask)    # [B, d_model]
        phla_pooled = self._masked_mean_pool(phla_output, phla_mask)  # [B, d_model]
        combined = torch.cat([tcr_pooled, phla_pooled], dim=-1)       # [B, d_model * 2]

        logits = self.classifier(combined)  # [B, 2]

        if return_attn:
            return logits, phla_output, attn
        return logits, phla_output
