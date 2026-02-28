# ============================================================================
# SETUP & INSTALLATIONS
# ============================================================================
try:
    import kagglehub
except ImportError:
    !pip install -q kagglehub

!pip install -q shap openpyxl wfdb imbalanced-learn

# ============================================================================
# IMPORTS
# ============================================================================
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import shap
import kagglehub
import ast
import wfdb
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score, balanced_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm


def print_banner(title, char="=", width=60):
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def minmax_normalize(tensor, dim=None, eps=1e-8):
    """ Perform min-max normalization on a tensor to scale values to [0, 1]. """
    if dim is not None:
        t_min = tensor.min(dim=dim, keepdim=True)[0]
        t_max = tensor.max(dim=dim, keepdim=True)[0]
    else:
        t_min, t_max = tensor.min(), tensor.max()
    return (tensor - t_min) / (t_max - t_min + eps)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    DATASET = "khyeh0719/ptb-xl-dataset"
    
    # Data parameters
    SIGNAL_LENGTH = 1000 # Target length for ECG signals (time steps)
    RPM_SIZE = 72 # Dimension of square RPM matrix (72x72)
    SAMPLING_RATE = 100 # sampling frequency of ECG signals in Hz
    
    # Model parameters
    NUM_CLASSES = 5
    IN_CHANNELS = 1
    OUT_FEATURES = 64
    KERNEL_SIZE = 3
    POOL_SIZE = 2
    CONV_STRIDE = 1
    POOL_STRIDE = 2
    PADDING = 1
    DROPOUT = 0.4 #adjusted dropout to 0.4 from  prevent regularization from holding back learning
    MOMENTUM = 0.1
    NUM_RESIDUAL_LAYERS = 3
    FEATURE_DIM = 256
    
    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class imbalance handling
    OVERSAMPLE_RATIO = 0.3     # SMOTE: minority classes to 30% of majority
    WEIGHT_DECAY = 1e-4     # L2 regularization
    
    # Focal Loss parameters
    USE_FOCAL_LOSS = True  #opted to use focal loss instead of cross entropy to boost minority classes
    FOCAL_GAMMA = 2.5 #minority classes underperformed requiring us to adjust to 2.5
    FOCAL_LABEL_SMOOTHING = 0.1
    USE_CLASS_BALANCED_LOSS = True  # Use effective number weighting
    CB_BETA = 0.9999  # Class-balanced loss beta parameter
    
    # Debug mode for quick testing(small datasets)
    DEBUG_MODE = False
    DEBUG_SAMPLES = 500
    
    # PTB-XL diagnostic superclasses
    # NORM: Normal ECG, MI: Myocardial Infarction, STTC: ST/T Change
    # CD: Conduction Disturbance, HYP: Hypertrophy
    CLASS_MAP = {
        "NORM": 0,
        "MI": 1,
        "STTC": 2,
        "CD": 3,
        "HYP": 4
    }
    CLASS_NAMES = list(CLASS_MAP.keys())

config = Config()

# ============================================================================
# DATA LOADING - PTB-XL DATASET
# ============================================================================
def download_ptbxl_dataset():
    return kagglehub.dataset_download(config.DATASET)


def explore_dataset_structure(path, max_depth=2, max_files=10):
    print_banner("DATASET STRUCTURE")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        if level > max_depth:
            break
        indent = '  ' * level
        print(f"{indent} {os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files[:max_files]:
            print(f"{subindent} {file}")
        if len(files) > max_files:
            print(f"{subindent}... and {len(files) - max_files} more files")


def load_ptbxl_metadata(dataset_path):
    csv_files = list(Path(dataset_path).rglob("ptbxl_database.csv"))
    if not csv_files:
        raise FileNotFoundError("ptbxl_database.csv not found!")
    csv_path = csv_files[0]
    base_path = csv_path.parent
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp_files = list(Path(dataset_path).rglob("scp_statements.csv")) # Load SCP statements (diagnostic code definitions)
    scp_df = pd.read_csv(scp_files[0], index_col=0) if scp_files else None
    return df, base_path, scp_df


def aggregate_diagnostic(y_dic, scp_df):
    """
    Extract diagnostic superclasses from SCP codes dictionary.
    
    PTB-XL uses SCP (Standard Communications Protocol) codes for diagnoses.
    This function maps fine-grained SCP codes to diagnostic superclasses.
    """    
    return list({
        scp_df.loc[key].diagnostic_class
        for key in y_dic
        if key in scp_df.index and pd.notna(scp_df.loc[key].diagnostic_class)
    })


def prepare_ptbxl_labels(df, scp_df):
    """
    Prepare PTB-XL labels by aggregating to diagnostic superclasses.
    
    Process:
    1. Map SCP codes to diagnostic superclasses
    2. Filter records with exactly one superclass (unambiguous labels)
    3. Map superclass names to integer indices
    """    
    if scp_df is None:
        raise ValueError("SCP statements required for label preparation!")
    df['diagnostic_superclass'] = df.scp_codes.apply(
        lambda x: aggregate_diagnostic(x, scp_df)
    )
    df_filtered = df[df.diagnostic_superclass.apply(len) == 1].copy()
    df_filtered['label'] = df_filtered.diagnostic_superclass.apply(lambda x: x[0])
    df_filtered = df_filtered[df_filtered.label.isin(config.CLASS_MAP.keys())]
    df_filtered['label_idx'] = df_filtered.label.map(config.CLASS_MAP)
    return df_filtered


def load_ecg_signals(df, base_path, lead_idx=1): #Loads raw ECG signals from WFDB format with 12 leads; it loads a single lead for each record
    signals, valid_indices, labels = [], [], []
    filename_col = 'filename_lr' if 'filename_lr' in df.columns else 'filename_hr'
    df_to_load = df.head(config.DEBUG_SAMPLES) if config.DEBUG_MODE else df
    
    for idx, row in tqdm(df_to_load.iterrows(), total=len(df_to_load), desc="Loading signals"):
        try:
            record_path = os.path.join(base_path, row[filename_col])
            record_path = record_path.replace('.dat', '').replace('.hea', '')
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, lead_idx]
            signals.append(signal.astype(np.float32))
            valid_indices.append(idx)
            labels.append(row['label_idx'])
        except Exception:
            continue
    return signals, valid_indices, labels


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def adjust_signal_length(signals, target_length, min_length_ratio=0.8):
    """
    Adjust ECG signals to target length by truncation or padding.
    
    Strategy:
    - Signals >= target_length: Truncate to target_length
    - Signals >= target_length * min_length_ratio: Pad with last value
    - Signals < minimum threshold: Discard
    """    
    adjusted = []
    for sig in signals:
        sig_tensor = torch.from_numpy(sig).float()
        if len(sig_tensor) >= target_length:
            adjusted.append(sig_tensor[:target_length]) # Truncate long signals
        elif len(sig_tensor) >= target_length * min_length_ratio:
            padding = target_length - len(sig_tensor)
            adjusted.append(torch.cat([sig_tensor, sig_tensor[-1].repeat(padding)])) # Pad short signals with last value
            # Signals too short are discarded
    return adjusted


def zscore_normalize(x_stacked, eps=1e-8): #handles physiological variability and noise
    means = x_stacked.mean(dim=1, keepdim=True)
    stds = x_stacked.std(dim=1, keepdim=True, unbiased=False)
    return (x_stacked - means) / (stds + eps)


def preprocess_signals(signals, target_length): #adjust length and normalize.
    adjusted = adjust_signal_length(signals, target_length)
    if not adjusted:
        raise ValueError("No signals with suitable length found")
    x_stacked = torch.stack(adjusted)
    return zscore_normalize(x_stacked)


def create_rpm_representations(x_normalized, m=72):
    """
    Create Recurrence Plot Matrix (RPM) representations from 1D ECG signals.
    
    RPM captures temporal dynamics by computing pairwise distances between
    downsampled signal points. The resulting 2D matrix can be processed by CNNs.
    
    Process:
    1. Downsample signal to m points using adaptive average pooling
    2. Compute pairwise Euclidean distance matrix (m x m)
    3. Normalize to [0, 1] range
    
    Mathematical formulation:
        RPM[i,j] = |x_downsampled[i] - x_downsampled[j]|
    """    
    x_downsampled = F.adaptive_avg_pool1d(x_normalized.unsqueeze(1), m).squeeze(1)
    batch_size = x_downsampled.shape[0]
    rpm_batch = x_downsampled.unsqueeze(2) - x_downsampled.unsqueeze(1)
    rpm_flat = rpm_batch.view(batch_size, -1)
    rpm_normalized = minmax_normalize(rpm_flat, dim=1)
    return rpm_normalized.view(batch_size, 1, m, m)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance handling.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
        - p_t: probability of correct class
        - gamma: focusing parameter (higher = more focus on hard examples)
        - alpha_t: class weight (balances importance of different classes)
    """
    EPS = 1e-7 # Numerical stability constant
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # Store alpha as buffer if tensor, else as attribute
        if alpha is not None and isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha
    
    def _get_focal_weights(self, p_t, targets):
        """Compute focal modulation weights."""
        focal_weight = (1.0 - p_t) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(p_t.device).gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        return focal_weight
    
    def _apply_label_smoothing(self, log_p, log_p_t):
        """Apply label smoothing to the loss."""
        if self.label_smoothing > 0:
            smooth_loss = -log_p.mean(dim=1)
            return (1.0 - self.label_smoothing) * (-log_p_t) + self.label_smoothing * smooth_loss
        return -log_p_t
    
    def _reduce(self, loss):
        """Apply reduction to the loss."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
        
    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        p = torch.clamp(torch.exp(log_p), self.EPS, 1.0 - self.EPS)
        
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = self._get_focal_weights(p_t, targets)
        ce_loss = self._apply_label_smoothing(log_p, log_p_t) # Compute cross-entropy with optional label smoothing
        
        return self._reduce(focal_weight * ce_loss)
    
    @staticmethod
    def compute_alpha_weights(samples_per_class, method='effective', beta=0.9999):
        """
        Compute class weights for alpha parameter.
        
        Args:
            samples_per_class: List of sample counts per class
            method: 'effective' (class-balanced) or 'inverse' (inverse frequency)
            beta: Beta parameter for effective number method
        
        Returns:
            Normalized weight tensor
        """
        samples = torch.tensor(samples_per_class, dtype=torch.float32)
        n_classes = len(samples_per_class)
        
        if method == 'effective':
            # Class-balanced weighting using effective number of samples
            effective_num = 1.0 - torch.pow(beta, samples)
            effective_num = torch.clamp(effective_num, min=1e-7)
            weights = (1.0 - beta) / effective_num
        else:  # inverse frequency
            weights = 1.0 / torch.clamp(samples, min=1.0)
        
        # Normalize weights
        return weights / weights.sum() * n_classes


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for deeper network training.
    
    Architecture:
        Input -> Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU -> Output
    
    The skip connection allows gradients to flow directly through the network,
    enabling training of very deep networks without vanishing gradients.
    """
    def __init__(self, channels, kernel_size=3, stride=1, momentum=0.1, padding=1): 
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(channels, momentum=momentum)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding) 
        self.bn2 = nn.BatchNorm2d(channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class FeatureExtractionModule(nn.Module):
    def __init__(self, config):
        super(FeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(config.IN_CHANNELS, config.OUT_FEATURES, 
                               config.KERNEL_SIZE, stride=config.CONV_STRIDE, padding=config.PADDING)
        self.bn1 = nn.BatchNorm2d(config.OUT_FEATURES, momentum=config.MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(config.POOL_SIZE, config.POOL_STRIDE)
        self.layers = nn.ModuleList([
            ResidualBlock(config.OUT_FEATURES, config.KERNEL_SIZE, config.CONV_STRIDE, 
                         config.MOMENTUM, config.PADDING) 
            for _ in range(config.NUM_RESIDUAL_LAYERS)
        ])
        self.conv2 = nn.Conv2d(config.OUT_FEATURES, config.OUT_FEATURES, 
                               config.KERNEL_SIZE, stride=config.CONV_STRIDE, padding=config.PADDING)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.fc1 = nn.Linear(config.OUT_FEATURES, config.FEATURE_DIM)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(config.FEATURE_DIM, config.NUM_CLASSES)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        features = self.fc1(x)
        features = self.relu(features)
        x = self.dropout2(features)
        logits = self.fc2(x)
        return features, logits


# ============================================================================
# EXPLAINABILITY MODULES
# ============================================================================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanation.
    
    Grad-CAM generates heatmaps showing which regions of the input (RPM) are
    important for the model's prediction. It uses gradients flowing into the
    final convolutional layer to weight the activation maps.
    
    Process:
    1. Forward pass: save activations of target layer
    2. Backward pass: compute gradients w.r.t. target class
    3. Weight activations by gradients (global average pooled)
    4. Create heatmap: weighted sum of activation maps
    
    Mathematical formulation:
        CAM = ReLU(Σ α_k * A_k)
        where α_k = (1/Z) Σ Σ ∂y^c / ∂A_k
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
    """    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output): # Hook to save forward pass activations.
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output): # Hook to save backward pass gradients
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        _, output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.einsum('c,chw->hw', weights, activations)
        cam = F.relu(cam)
        cam = minmax_normalize(cam).cpu().numpy()
        return cam, output


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretation.
    
    SHAP uses game theory to assign each input feature an importance value
    for a particular prediction. It provides a unified framework for
    interpreting model outputs.
    
    For deep learning, DeepExplainer uses a computationally efficient
    approximation based on DeepLIFT. --- Reference:
        Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
    """    
    def __init__(self, model, background_data, device='cpu'):
        self.model = model
        self.device = device
        self.background_data = background_data.to(device)
        def model_wrapper(x):
            _, logits = self.model(x)
            return logits
        self.model_wrapper = model_wrapper
        self.explainer = None
        
    def create_explainer(self):
        """Initialize SHAP DeepExplainer with background data."""        
        self.model.eval()
        self.explainer = shap.DeepExplainer(self.model_wrapper, self.background_data)
        return self.explainer
    
    def explain(self, input_data, n_samples=50):
        if self.explainer is None:
            self.create_explainer()
        input_data = input_data.to(self.device)
        shap_values = self.explainer.shap_values(input_data, nsamples=n_samples)
        return shap_values


class SHAPCompatibleModel(nn.Module):
    """
    Simplified model wrapper for SHAP compatibility.
    
    SHAP's DeepExplainer sometimes has issues with complex models.
    This wrapper provides a simplified forward pass while maintaining the same computation graph.
    """    
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.maxpool = original_model.maxpool
        self.layers = original_model.layers
        self.conv2 = original_model.conv2
        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in self.layers:
            identity = x
            out = F.relu(layer.bn1(layer.conv1(x)))
            out = layer.bn2(layer.conv2(out))
            x = F.relu(out + identity)
        x = self.conv2(x)
        x = x.mean(dim=[2, 3])
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
class ECGAugmentation:
    """ECG-appropriate augmentation for RPM representations.
    
    Avoided using flipping because RPMs encode temporal distance relationships, and flipping would destroy signal semantics.
    """
    def __init__(self, noise_std=0.04, scale_range=(0.92, 1.08), 
                 p=0.5, noise_p=0.5, scale_p=0.5):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.p = p
        self.noise_p = noise_p
        self.scale_p = scale_p
    
    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        augmented = x.clone()
        if torch.rand(1).item() < self.noise_p:
            augmented = augmented + torch.randn_like(x) * self.noise_std
        if torch.rand(1).item() < self.scale_p:
            scale = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(*self.scale_range)
            augmented = augmented * scale
        return augmented


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, augmentation=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        if augmentation is not None:
            inputs = augmentation(inputs)
        optimizer.zero_grad()
        _, outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    return running_loss / total, 100 * correct / total


def _run_inference(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_outputs = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_outputs.append(outputs.cpu())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels, torch.cat(all_outputs)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total


def compute_metrics(all_labels, all_preds, class_names):
    """Compute comprehensive classification metrics."""
    unique_labels = sorted(set(all_labels) | set(all_preds))
    label_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    metrics = {
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'per_class_f1': dict(zip(
            label_names,
            f1_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
        )),
        'per_class_precision': dict(zip(
            label_names,
            precision_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
        )),
        'per_class_recall': dict(zip(
            label_names,
            recall_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
        )),
    }
    return metrics, unique_labels, label_names


def plot_confusion_matrices(all_labels, all_preds, label_names, unique_labels):
    """Plot raw and normalized confusion matrices side by side."""
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized by true class; shows recall per class
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model with comprehensive metrics for imbalanced classification."""
    all_preds, all_labels, _ = _run_inference(model, dataloader, device)
    metrics, unique_labels, label_names = compute_metrics(all_labels, all_preds, class_names)
    
    # Print macro metrics
    print_banner("MACRO METRICS (Key for Imbalanced Data)")
    for key in ['macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'weighted_f1']:
        print(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}")
    
    # Print per-class metrics table
    print_banner("PER-CLASS METRICS", char="-", width=50)
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 52)
    
    class_supports = {}
    for name in label_names:
        idx = class_names.index(name)
        support = sum(1 for l in all_labels if l == idx)
        class_supports[name] = support
        print(f"{name:<10} {metrics['per_class_precision'][name]:>10.4f} "
              f"{metrics['per_class_recall'][name]:>10.4f} "
              f"{metrics['per_class_f1'][name]:>10.4f} {support:>10}")
    
    # Highlight minority class
    min_class = min(class_supports, key=class_supports.get)
    print(f"\n Minority class '{min_class}': F1={metrics['per_class_f1'][min_class]:.4f}")
    

    print_banner("CLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds, labels=unique_labels,
                                target_names=label_names, zero_division=0))
    
    plot_confusion_matrices(all_labels, all_preds, label_names, unique_labels)
    
    return all_preds, all_labels, metrics


# ============================================================================
# HELPER FUNCTIONS FOR MAIN
# ============================================================================
def load_and_prepare_data():
    dataset_path = download_ptbxl_dataset()
    df, base_path, scp_df = load_ptbxl_metadata(dataset_path)
    if scp_df is None:
        raise ValueError("SCP statements file not found!")
    df_labeled = prepare_ptbxl_labels(df, scp_df)
    if len(df_labeled) == 0:
        raise ValueError("No labeled data found!")
    signals, _, y_labels = load_ecg_signals(df_labeled, base_path, lead_idx=1)
    if not signals:
        raise ValueError("No ECG signals loaded!")
    x_normalized = preprocess_signals(signals, config.SIGNAL_LENGTH)
    x = create_rpm_representations(x_normalized, config.RPM_SIZE)
    y = torch.tensor(y_labels[:len(x)], dtype=torch.long)
    return x, y


def compute_class_weights(y):
    """Compute class counts and weights from labels."""
    class_counts = torch.bincount(y, minlength=config.NUM_CLASSES).float()
    samples_per_class = torch.clamp(class_counts, min=1.0).tolist()
    
    # Use FocalLoss static method to compute weights
    method = 'effective' if config.USE_CLASS_BALANCED_LOSS else 'inverse'
    class_weights = FocalLoss.compute_alpha_weights(
        samples_per_class, method=method, beta=config.CB_BETA
    ).to(config.DEVICE)
    
    return class_weights, samples_per_class


def create_dataloaders(x, y):
    x_np, y_np = x.numpy(), y.numpy()
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_np, y_np, test_size=config.TEST_SPLIT, stratify=y_np, random_state=42)
    val_ratio = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=42)
    
    print_banner("TRAINING DATA DISTRIBUTION", char="-", width=50)
    print(f"Original training distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    for cls, cnt in class_counts.items():
        print(f"  Class {config.CLASS_NAMES[cls]}: {cnt} samples")
    
    if config.OVERSAMPLE_RATIO and config.OVERSAMPLE_RATIO > 0:
        majority_count = max(counts)
        target_count = int(majority_count * config.OVERSAMPLE_RATIO)
        sampling_strategy = {}
        for cls, cnt in class_counts.items():
            if cnt < target_count:
                sampling_strategy[cls] = target_count
        
        if sampling_strategy:
            print(f"\nApplying SMOTE (target = {config.OVERSAMPLE_RATIO:.0%} of majority = {target_count}):")
            for cls, target in sampling_strategy.items():
                print(f"  Class {config.CLASS_NAMES[cls]}: {class_counts[cls]} -> {target}")
            original_shape = x_train.shape[1:]
            x_train_flat = x_train.reshape(x_train.shape[0], -1)
            smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
            x_train_resampled, y_train_resampled = smote.fit_resample(x_train_flat, y_train)
            x_train_resampled = x_train_resampled.reshape(-1, *original_shape)
            x_train_resampled, y_train_resampled = shuffle(
                x_train_resampled, y_train_resampled, random_state=42)
            print(f"\nFinal training distribution:")
            unique_r, counts_r = np.unique(y_train_resampled, return_counts=True)
            for cls, cnt in zip(unique_r, counts_r):
                print(f"  Class {config.CLASS_NAMES[cls]}: {cnt} samples")
            print(f"\nTotal: {len(y_train)} -> {len(y_train_resampled)} samples")
        else:
            print("No classes need oversampling (all above target).")
            x_train_resampled, y_train_resampled = x_train, y_train
    else:
        print("SMOTE disabled (OVERSAMPLE_RATIO = None or 0)")
        x_train_resampled, y_train_resampled = x_train, y_train
    
    loaders = {
        'train': DataLoader(
            TensorDataset(torch.from_numpy(x_train_resampled).float(), 
                         torch.from_numpy(y_train_resampled)),
            batch_size=config.BATCH_SIZE, shuffle=True),
        'val': DataLoader(
            TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
            batch_size=config.BATCH_SIZE, shuffle=False),
        'test': DataLoader(
            TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
            batch_size=config.BATCH_SIZE, shuffle=False),
    }
    return loaders


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train', linewidth=2)
    ax1.plot(val_losses, label='Val', linewidth=2)
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(train_accs, label='Train', linewidth=2)
    ax2.plot(val_accs, label='Val', linewidth=2)
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_gradcam(model, test_loader, device):
    grad_cam = GradCAM(model, model.conv2)
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[0:1].to(device)
    cam, _ = grad_cam.generate_cam(sample_input)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sample_input[0, 0].cpu(), cmap='gray')
    axes[0].set_title('Original RPM')
    axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    axes[2].imshow(sample_input[0, 0].cpu(), cmap='gray', alpha=0.7)
    axes[2].imshow(cam, cmap='jet', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def compute_shap_values(model, x, device):
    try:
        shap_model = SHAPCompatibleModel(model).to(device)
        shap_model.eval()
        background = x[:50].to(device)
        test_samples = x[50:55].to(device)
        explainer = shap.DeepExplainer(shap_model, background)
        shap_values = explainer.shap_values(test_samples, check_additivity=False)
        if shap_values is not None:
            if isinstance(shap_values, list):
                shap_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                shap_img = shap_combined[0, 0]
            else:
                shap_img = np.abs(shap_values[0, 0])
            while shap_img.ndim > 2:
                shap_img = np.mean(shap_img, axis=-1)
            plt.figure(figsize=(10, 8))
            plt.imshow(shap_img, cmap='hot')
            plt.title('SHAP Feature Importance (RPM)')
            plt.colorbar(label='|SHAP value|')
            plt.savefig('shap_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"SHAP computation failed: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print_banner("ECG CLASSIFICATION WITH GRAD-CAM AND SHAP")
    
    x, y = load_and_prepare_data()
    class_weights, samples_per_class = compute_class_weights(y)
    loaders = create_dataloaders(x, y)
    
    # Display class distribution info
    print_banner("CLASS DISTRIBUTION ANALYSIS", char="-", width=50)
    total_samples = sum(samples_per_class)
    for i, (name, count) in enumerate(zip(config.CLASS_NAMES, samples_per_class)):
        pct = 100 * count / total_samples
        print(f"  {name}: {int(count):5d} samples ({pct:5.1f}%)")
    
    imbalance_ratio = max(samples_per_class) / min(samples_per_class)
    print(f"\n  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    model = FeatureExtractionModule(config).to(config.DEVICE)
    augmentation = ECGAugmentation()
    
    if config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config.FOCAL_GAMMA,
            label_smoothing=config.FOCAL_LABEL_SMOOTHING
        )
        weight_method = "effective number" if config.USE_CLASS_BALANCED_LOSS else "inverse frequency"
        print(f"\n✓ Using Focal Loss:")
        print(f"  - Gamma (focusing): {config.FOCAL_GAMMA}")
        print(f"  - Alpha weights: {weight_method}")
        print(f"  - Label smoothing: {config.FOCAL_LABEL_SMOOTHING}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\n✓ Using Cross Entropy Loss with class weights")
    
    # AdamW's reliable regularization prevents the model from memorizing normal patterns and  synthetic SMOTE artifacts
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY) 
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, config.DEVICE, augmentation)
        val_loss, val_acc = validate(model, loaders['val'], criterion, config.DEVICE)
        
        # Compute macro F1 for model selection; data was imbalanced
        all_preds, all_labels, _ = _run_inference(model, loaders['val'], config.DEVICE)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(epoch + 1)
        
        is_best = val_f1 > best_val_f1
        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} │ "
              f"Train: {train_loss:.4f} ({train_acc:5.2f}%) │ "
              f"Val: {val_loss:.4f} ({val_acc:5.2f}%) │ "
              f"F1: {val_f1:.4f}" + (" ✓ BEST" if is_best else ""))
        
        if is_best:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = validate(model, loaders['test'], criterion, config.DEVICE)
    print(f"\nTest Results: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
    _, _, metrics = evaluate_model(model, loaders['test'], config.DEVICE, config.CLASS_NAMES)
    visualize_gradcam(model, loaders['test'], config.DEVICE)
    compute_shap_values(model, x, config.DEVICE)
    
    print_banner("TRAINING COMPLETE - SUMMARY")
    print(f"  Best Validation F1:       {best_val_f1:.4f}")
    print(f"  Test Accuracy:            {test_acc:.2f}%")
    print(f"\n  === Key Metrics for Imbalanced Data ===")
    print(f"  Macro F1-Score:           {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy:        {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro Recall:             {metrics['macro_recall']:.4f}")
    print(f"\n  Per-Class F1 Scores:")
    for class_name, f1 in metrics['per_class_f1'].items():
        print(f"    {class_name}: {f1:.4f}")


if __name__ == "__main__":
    main()
