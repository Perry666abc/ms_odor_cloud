import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleSinusoidalMzEncoding(nn.Module):
    """
    Multi-Scale Continuous Fourier/Sinusoidal Encoding for Mass Spectrometry.
    Key Intellectual Property & Core Feature:
    Provides continuous, resolution-invariant coordinate representations without discrete binning.
    Covers wavelengths from min_wavelength (e.g. 0.005 Da for high-resolution isotopic peaks)
    to max_wavelength (e.g. 2000.0 Da for full molecular mass range).
    """
    def __init__(self, d_model, min_wavelength=0.005, max_wavelength=2000.0):
        super().__init__()
        self.d_model = d_model
        num_bands = d_model // 2
        
        # Angular frequencies omega = 2*pi / wavelength spaced logarithmically
        log_min_freq = math.log(2.0 * math.pi / max_wavelength)
        log_max_freq = math.log(2.0 * math.pi / min_wavelength)
        omegas = torch.exp(torch.linspace(log_min_freq, log_max_freq, num_bands))
        self.register_buffer('omegas', omegas)

    def forward(self, mzs):
        """
        mzs: (batch_size, seq_len) or (batch_size, 1)
        Returns: (batch_size, seq_len, d_model)
        """
        # (B, S, 1) * (num_bands,) -> (B, S, num_bands)
        angles = mzs.unsqueeze(-1) * self.omegas
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pe

class MSOdorTransformer(nn.Module):
    """
    End-to-End Mass Spectrometry Transformer for Multi-Label Odor Prediction & Chemical Semantic Alignment.
    
    Core Features Preserved:
    1. Multi-scale continuous Fourier M/Z encoding (high-res capable down to 0.005 Da).
    2. Explicit continuous Neutral Loss (Delta m/z) feature embedding.
    3. Physically normalized continuous Collision Energy (CE) & Precursor m/z embeddings.
    4. Missing metadata tokenization for GC-MS vs LC-MS/MS decoupling.
    5. Pre-LayerNorm Transformer backbone for gradient stability.
    6. Multimodal projection heads aligning with Open-POM 256-dim chemical manifold (SCENT arXiv:2605.27009).
    """
    def __init__(self, latent_dim=256, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # 1. Continuous Peak Feature Encoders
        self.mz_encoder = MultiScaleSinusoidalMzEncoding(d_model)
        self.nl_encoder = MultiScaleSinusoidalMzEncoding(d_model)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        # Fuse continuous m/z + intensity + neutral loss
        self.peak_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Continuous Metadata Embeddings & Missing Tokens
        self.precursor_mz_encoder = MultiScaleSinusoidalMzEncoding(d_model)
        self.missing_pmz_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Continuous Collision Energy (scaled by 0.01 for numerical conditioning)
        self.ce_mlp = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.missing_ce_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Instrument mode: 0: GC-EI, 1: LC-ESI-Pos, 2: LC-ESI-Neg, 3: Other/Unknown
        self.instrument_emb = nn.Embedding(4, d_model)
        
        # 3. Global Context CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 4. Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for superior gradient flow & stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Multimodal Alignment Head (Stage 1: SCENT Contrastive InfoNCE Alignment)
        # Projects MS CLS representation to 256-dim space matching Open-POM
        self.align_dim = 256
        self.align_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.align_dim),
            nn.LayerNorm(self.align_dim)
        )
        
        # 6. Pretrained POM Perception Distillation Head (Stage 2: 256-dim Principal Odor Map)
        self.pom_dim = 256
        self.pom_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pom_dim),
            nn.LayerNorm(self.pom_dim)
        )
        
        # 7. Downstream 138-dim Odor Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.pom_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 138)
        )
        
        # Legacy latent head for backward compatibility
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def extract_features(self, mzs, intensities, neutral_losses, precursor_mz, has_precursor, ce, has_ce, instrument_mode, padding_mask=None):
        """Extracts pooled global CLS representation (B, d_model)."""
        B, S = mzs.size()
        
        # Encode Continuous Peaks
        mz_emb = self.mz_encoder(mzs)                           # (B, S, D)
        int_emb = self.intensity_mlp(intensities.unsqueeze(-1)) # (B, S, D)
        nl_emb = self.nl_encoder(neutral_losses)                # (B, S, D)
        
        peak_combined = torch.cat([mz_emb, int_emb, nl_emb], dim=-1) # (B, S, 3D)
        peak_emb = self.peak_fuse(peak_combined)                      # (B, S, D)
        
        # Encode Continuous Metadata
        pmz_encoded = self.precursor_mz_encoder(precursor_mz)         # (B, 1, D)
        pmz_emb = has_precursor.unsqueeze(-1) * pmz_encoded + (1.0 - has_precursor.unsqueeze(-1)) * self.missing_pmz_token
        
        # Normalize continuous CE by 0.01 (0~100 eV -> 0~1.0) for stable MLP conditioning
        ce_scaled = ce * 0.01
        ce_encoded = self.ce_mlp(ce_scaled.unsqueeze(-1))             # (B, 1, D)
        ce_emb = has_ce.unsqueeze(-1) * ce_encoded + (1.0 - has_ce.unsqueeze(-1)) * self.missing_ce_token
        
        inst_emb = self.instrument_emb(instrument_mode)               # (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)
        
        # Concatenate prefix tokens with continuous peak tokens
        seq = torch.cat([cls_tokens, pmz_emb, ce_emb, inst_emb, peak_emb], dim=1) # (B, S+4, D)
        
        if padding_mask is not None:
            prefix_mask = torch.zeros(B, 4, dtype=torch.bool, device=seq.device)
            full_mask = torch.cat([prefix_mask, padding_mask], dim=1)
        else:
            full_mask = None
            
        out = self.transformer_encoder(seq, src_key_padding_mask=full_mask)
        cls_out = out[:, 0, :] # (B, D)
        return cls_out

    def get_aligned_embedding(self, mzs, intensities, neutral_losses, precursor_mz, has_precursor, ce, has_ce, instrument_mode, padding_mask=None, normalize=True):
        """Extracts 256-dimensional representation aligned with Open-POM."""
        cls_out = self.extract_features(
            mzs, intensities, neutral_losses,
            precursor_mz, has_precursor,
            ce, has_ce, instrument_mode,
            padding_mask=padding_mask
        )
        emb = self.align_head(cls_out) # (B, 256)
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(self, mzs, intensities, neutral_losses, precursor_mz, has_precursor, ce, has_ce, instrument_mode, padding_mask=None, return_all=False):
        """
        Forward pass.
        If return_all=False: returns POM representation (B, pom_dim).
        If return_all=True: returns dict with logits, pom_pred, align_pred, and cls_out.
        """
        cls_out = self.extract_features(
            mzs, intensities, neutral_losses,
            precursor_mz, has_precursor,
            ce, has_ce, instrument_mode,
            padding_mask=padding_mask
        )
        
        pom_pred = self.pom_head(cls_out)       # (B, 256)
        align_pred = self.align_head(cls_out)   # (B, 256)
        logits = self.classifier(pom_pred)      # (B, 138)
        
        if return_all:
            return {
                'logits': logits,
                'pom_pred': pom_pred,
                'align_pred': align_pred,
                'cls_out': cls_out
            }
        
        if self.latent_dim != 256:
            return self.head(cls_out)
        
        return pom_pred

class MolecularStructureEncoder(nn.Module):
    """
    Molecular Structure Encoder for Stage 1 Multimodal Contrastive Pre-training.
    Projects molecular representations (Open-POM / GNN embeddings / fingerprints)
    into the shared multimodal alignment space s_i in R^{256}.
    """
    def __init__(self, in_dim=256, align_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        if in_dim == align_dim:
            # Identity or residual adapter when already in 256-dim Open-POM space
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, align_dim),
                nn.LayerNorm(align_dim)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, align_dim),
                nn.LayerNorm(align_dim)
            )

    def forward(self, mol_features):
        """
        mol_features: (B, in_dim) Open-POM or molecular embedding tensor
        Returns: (B, align_dim) L2-normalized vector s_i
        """
        s = self.encoder(mol_features)
        return F.normalize(s, p=2, dim=-1)

class PretrainedPOMPerceptionTeacher(nn.Module):
    """
    Pretrained Open-POM 256-dim Perception Manifold Teacher.
    Bridges the Open-POM manifold with 138-dimensional downstream odor logits.
    """
    def __init__(self, pom_dim=256, num_classes=138, hidden_dim=256, prior_prob=None):
        super().__init__()
        self.pom_dim = pom_dim
        self.num_classes = num_classes
        
        # Perceptual Decoder: POM 256-dim -> 138-dim Odor Logits
        self.decoder = nn.Sequential(
            nn.Linear(pom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Multi-label encoder fallback
        self.encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pom_dim),
            nn.LayerNorm(pom_dim)
        )
        
        if prior_prob is not None:
            eps = 1e-5
            p = prior_prob.clamp(min=eps, max=1.0 - eps)
            bias_val = torch.log(p / (1.0 - p))
            self.decoder[-1].bias.data.copy_(bias_val)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        logits = self.decode(latent)
        return logits, latent

# Alias for full backward compatibility
OdorAutoencoder = PretrainedPOMPerceptionTeacher
