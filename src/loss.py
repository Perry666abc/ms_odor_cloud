import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for extreme multi-label classification.
    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    
    Key mechanisms:
    1. Asymmetric focusing: gamma_neg > gamma_pos to heavily suppress gradient from easy negatives (96%+ of data).
    2. Probability margin shifting: max(p - margin, 0) completely zeros out tiny false positive gradients.
    3. Retains strong gradients for rare positive classes (gamma_pos ~ 0 or 1).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, reduction='mean', pos_weight=None):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes)
        targets: (batch_size, num_classes), multi-hot float tensor in {0, 1}
        """
        # Calculating Probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping for Negative classes
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic BCE terms with eps clamping for numerical stability
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing Weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.gamma_pos > 0:
                pt_pos = xs_pos
                asym_w_pos = (1.0 - pt_pos).pow(self.gamma_pos)
            else:
                asym_w_pos = 1.0
                
            if self.gamma_neg > 0:
                pt_neg = 1.0 - xs_neg  # shifted probability
                pt_neg = pt_neg.clamp(min=0.0)
                asym_w_neg = pt_neg.pow(self.gamma_neg)
            else:
                asym_w_neg = 1.0
                
            asym_weights = targets * asym_w_pos + (1.0 - targets) * asym_w_neg
            loss = loss * asym_weights

        # Optional Class Weights
        if self.pos_weight is not None:
            w = self.pos_weight.to(logits.device)
            loss = loss * (targets * w + (1.0 - targets) * 1.0)

        loss = -loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MultiLabelFocalLoss(nn.Module):
    """
    Cost-sensitive Multi-Label Focal Loss with class weights alpha.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = targets * alpha + (1.0 - targets) * 1.0
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultimodalInfoNCELoss(nn.Module):
    """
    CLIP-Style Symmetric InfoNCE Contrastive Loss for Mass Spectrometry & Molecular Structure Alignment.
    Reference: SCENT (arXiv:2605.27009), Radford et al. (CLIP, ICML 2021).
    
    Formula:
        L_align = - 1/(2B) * sum_i [ log( exp(s_i^T m_i / tau) / sum_j exp(s_i^T m_j / tau) )
                                   + log( exp(s_i^T m_i / tau) / sum_j exp(s_j^T m_i / tau) ) ]
    """
    def __init__(self, init_tau=0.03, learnable_tau=True):
        super(MultimodalInfoNCELoss, self).__init__()
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau, dtype=torch.float32)))
        else:
            self.register_buffer('log_tau', torch.log(torch.tensor(init_tau, dtype=torch.float32)))
            
    def forward(self, ms_emb, mol_emb):
        """
        ms_emb: (B, D) float tensor from MS Transformer
        mol_emb: (B, D) float tensor from Molecular Structure Encoder (e.g. MolFormer / Morgan FP)
        """
        B = ms_emb.size(0)
        device = ms_emb.device
        
        # L2 Normalization
        ms_norm = F.normalize(ms_emb, p=2, dim=-1)
        mol_norm = F.normalize(mol_emb, p=2, dim=-1)
        
        # Bounded temperature for numerical stability: tau in [0.01, 1.0]
        tau = torch.exp(self.log_tau).clamp(min=0.01, max=1.0)
        
        # Cosine similarity matrix: (B, B)
        # sim[i, j] = s_i^T m_j / tau
        sim = torch.matmul(mol_norm, ms_norm.t()) / tau
        
        # Ground-truth diagonal targets: pair i matches pair i
        targets = torch.arange(B, dtype=torch.long, device=device)
        
        # Symmetric InfoNCE loss: (Mol-to-MS + MS-to-Mol) / 2
        loss_mol2ms = F.cross_entropy(sim, targets)
        loss_ms2mol = F.cross_entropy(sim.t(), targets)
        loss_align = 0.5 * (loss_mol2ms + loss_ms2mol)
        
        # Accuracy metrics for monitoring retrieval during training
        with torch.no_grad():
            acc_mol2ms = (sim.argmax(dim=-1) == targets).float().mean()
            acc_ms2mol = (sim.t().argmax(dim=-1) == targets).float().mean()
            mean_acc = 0.5 * (acc_mol2ms + acc_ms2mol)
            
        return loss_align, mean_acc, tau.item()

class POMManifoldDistillationLoss(nn.Module):
    """
    Feature Distillation Loss aligning MS Transformer with Pretrained Open-POM 256-dim Perceptual Manifold.
    Combines Mean Squared Error (MSE) and Cosine Distance to constrain both vector magnitude
    and geometric orientation in the human Principal Odor Map (POM).
    """
    def __init__(self, alpha_mse=1.0, alpha_cos=0.5):
        super(POMManifoldDistillationLoss, self).__init__()
        self.alpha_mse = alpha_mse
        self.alpha_cos = alpha_cos
        
    def forward(self, pred_pom, teacher_pom):
        """
        pred_pom: (B, 256) predicted POM representation from MS Transformer
        teacher_pom: (B, 256) frozen POM representation from Pretrained Open-POM Teacher (detached)
        """
        # 1. MSE loss for absolute coordinate alignment
        loss_mse = F.mse_loss(pred_pom, teacher_pom.detach())
        
        # 2. Cosine distance loss for perceptual angle/direction alignment
        pred_norm = F.normalize(pred_pom, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_pom.detach(), p=2, dim=-1)
        loss_cos = 1.0 - (pred_norm * teacher_norm).sum(dim=-1).mean()
        
        total_distill_loss = self.alpha_mse * loss_mse + self.alpha_cos * loss_cos
        return total_distill_loss, loss_mse.item(), loss_cos.item()

