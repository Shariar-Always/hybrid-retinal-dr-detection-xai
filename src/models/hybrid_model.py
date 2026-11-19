import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM).
    Focuses on 'what' pathological features (e.g., exudates, microaneurysms,
    hemorrhages, neovascularization) are most informative across channels.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_planes = max(in_planes // ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM).
    Focuses on 'where' retinal lesions are localized across the fundus field.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class HybridDRPredictor(nn.Module):
    """
    Hybrid CNN + CBAM Attention Model for Automated Diabetic Retinopathy (DR)
    Detection, 5-Stage ICDR Severity Assessment, and 256-D Retinal Vector
    Embedding Extraction (for Vector Database Case Retrieval & RAG).

    Classes (ICDR Scale):
        0: No DR
        1: Mild NPDR
        2: Moderate NPDR
        3: Severe NPDR
        4: Proliferative DR (PDR)

    Supported Hybrid Backbones: 'efficientnet_b0', 'densenet121', 'resnet50'.
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
    ):
        super(HybridDRPredictor, self).__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.embedding_dim = 256

        if backbone == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            densenet = models.densenet121(weights=weights)
            self.features = densenet.features
            self.num_features = 1024

        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.num_features = 2048

        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            effnet = models.efficientnet_b0(weights=weights)
            self.features = effnet.features
            self.num_features = 1280

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                "Choose from ['densenet121', 'resnet50', 'efficientnet_b0']."
            )

        # Sequential CBAM Dual Attention Blocks (Channel + Spatial)
        self.channel_attention = ChannelAttention(self.num_features)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 5-Stage ICDR Severity Classification Head (Penultimate 256-D layer serves as Vector DB Embedding)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def _forward_cbam_pooled(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        ca_weights = self.channel_attention(features)
        feat_ca = features * ca_weights
        sa_weights = self.spatial_attention(feat_ca)
        feat_out = feat_ca * sa_weights
        pooled = self.avgpool(feat_out)
        return torch.flatten(pooled, 1)

    def extract_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extracts the 256-dimensional CBAM-refined retinal feature embedding vector
        for indexing and querying in the Vector Database (FAISS / Cosine Index).
        """
        flat = self._forward_cbam_pooled(x)
        emb = self.classifier[0](flat)
        emb = self.classifier[1](emb)
        emb = self.classifier[2](emb)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward_with_embedding(self, x: torch.Tensor):
        """
        Single-pass forward returning both (logits, normalized_256d_embedding).
        """
        flat = self._forward_cbam_pooled(x)
        emb_raw = self.classifier[2](self.classifier[1](self.classifier[0](flat)))
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.classifier[4](self.classifier[3](emb_raw))
        return logits, emb_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._forward_cbam_pooled(x)
        return self.classifier(flat)


if __name__ == "__main__":
    backbones = ["efficientnet_b0", "densenet121", "resnet50"]
    dummy_input = torch.randn(2, 3, 224, 224)

    print("================ Hybrid CNN + CBAM + Vector Embedding ================")
    for bb in backbones:
        model = HybridDRPredictor(backbone=bb, num_classes=5, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits, emb = model.forward_with_embedding(dummy_input)
        print(
            f"[{bb} + CBAM] Logits: {tuple(logits.shape)} | "
            f"Vector DB Embedding: {tuple(emb.shape)}"
        )
