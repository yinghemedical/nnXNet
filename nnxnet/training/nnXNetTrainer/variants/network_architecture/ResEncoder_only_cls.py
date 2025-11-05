from typing import Tuple, Union, List, Type
import torch
import torch.nn as nn
import os
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD

# ----------------- 自定义的 CrossAttentionPooling 以支持多查询和拼接 -----------------
class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        
        # 可学习的查询向量，形状为 [query_num, embed_dim] 
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        
        # Cross Attention 层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 分类器：将 [query_num * D] 映射到 [num_classes]
        # 注意这里是 query_num * embed_dim
        self.classifier = nn.Linear(query_num * embed_dim, num_classes) 
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 图像特征 [B, D, H, W, L] 或 [B, D, H*W*L]
        
        Returns:
            分类logits [B, num_classes]
        """
        batch_size = x.shape[0]
        
        # 处理输入特征
        if x.dim() == 5:  # [B, D, H, W, L]
            x = x.flatten(2)  # [B, D, H*W*L]
        
        # 调整维度: [B, D, L] -> [L, B, D] (seq_len, batch, embed_dim)
        x = x.permute(2, 0, 1)  # [H*W*L, B, D]
        
        # 扩展查询向量: [query_num, embed_dim] -> [query_num, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)  # [query_num, B, D]
        
        # Cross Attention
        attended, attention_weights = self.cross_attention(
            query=query, 
            key=x,        
            value=x,      
        )
        
        # attended 形状: [query_num, B, D]
        attended = self.norm(attended)
        attended = self.dropout(attended)
        
        # 调整维度: [query_num, B, D] -> [B, query_num, D]
        attended_permuted = attended.permute(1, 0, 2)
        
        # 展平查询维度和特征维度: [B, query_num, D] -> [B, query_num * D]
        attended_flatten = attended_permuted.flatten(1) 
        
        # 应用分类器：将 [B, query_num * D] 映射到 [B, num_classes]
        logits = self.classifier(attended_flatten)  # [B, num_classes]
        
        return logits


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        query_num,
        num_classes,
        dropout=0.0,
        use_cross_attention=True,
        num_heads=4
    ):
        """
        Args:
            embed_dim (int): 嵌入维度
            query_num (int): 查询数量
            num_classes (int): 类别数量
            dropout (float): dropout率
            use_cross_attention (bool): 是否使用cross attention pooling
            num_heads (int): attention头数
        """
        super(ClassificationHead, self).__init__()
        
        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim=embed_dim,
                query_num=query_num,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # 全局平均池化
                nn.Flatten(1),            # 展平为 [B, D]
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
    
    def forward(self, x):
        return self.pooling(x)
# --------------------------------------------------------------------------------

class ResEncoder_only_cls(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cls_head_num_classes_list: List[int] = [1, 4], 
        cls_drop_out_list: List[float] = [0.0, 0.0],  
        cls_query_num_list: List[int] = [2, 4],        
        use_cross_attention: bool = True,           
        num_heads: int = 4,                    
        n_stages: int = 6,
        features_per_stage: List[int] = [16, 32, 64, 128, 256, 320],
        kernel_sizes: List[Union[Tuple[int, int, int], int]] = [(3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        strides: List[Union[Tuple[int, int, int], int]] = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        dropout_rate: float = 0.0,
        norm_op: nn.Module = nn.InstanceNorm3d,
        norm_op_kwargs: dict = {"eps": 1e-05, "affine": True},
        conv_op: nn.Module = nn.Conv3d,
        conv_bias: bool = True,
        nonlin: nn.Module = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        dropout_op: Union[None, Type[nn.Module]] = None,
        dropout_op_kwargs: dict = None,
        n_blocks_per_stage: List[int] = [2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder: List[int] = [2, 2, 2, 2, 2],
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.in_channels = in_channels
        num_stages = len(features_per_stage)
        self.num_stages = num_stages

        kernel_sizes = kernel_sizes or [(3, 3, 3) for _ in range(num_stages)]
        strides = strides or [(2, 2, 2) for _ in range(num_stages)]
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-05, "affine": True}
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}
        n_blocks_per_stage = n_blocks_per_stage or [2 for _ in range(num_stages)]
        n_conv_per_stage_decoder = n_conv_per_stage_decoder or [2 for _ in range(num_stages - 1)]
        
        # 检查分类头参数列表长度是否一致
        if not (len(cls_head_num_classes_list) == len(cls_drop_out_list) == len(cls_query_num_list)):
            raise ValueError("cls_head_num_classes_list, cls_drop_out_list, and cls_query_num_list must have the same length.")

        # Convolutional Encoder Blocks (U-Net style)
        self.conv_encoder_blocks = self.build_encoder_block(n_blocks_per_stage, features_per_stage, kernel_sizes, strides, in_channels, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, BasicBlockD)
        
        # 分类头列表
        self.cls_head_list = nn.ModuleList()
        embed_dim = features_per_stage[-1] # 使用最后一个 stage 的特征维度
        for i in range(len(cls_head_num_classes_list)):
            cls_head_num_classes = cls_head_num_classes_list[i]
            cls_drop_out = cls_drop_out_list[i]
            cls_query_num = cls_query_num_list[i] 
            
            # 使用 ClassificationHead
            self.cls_head_list.append(ClassificationHead(
                embed_dim=embed_dim, 
                query_num=cls_query_num, 
                num_classes=cls_head_num_classes, 
                dropout=cls_drop_out, 
                use_cross_attention=use_cross_attention, 
                num_heads=num_heads
            ))

    def build_encoder_block(self, n_blocks_per_stage, features_per_stage, kernel_sizes, strides, in_channels, conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block_type):
        blocks = nn.ModuleList()
        blocks.append(
            StackedResidualBlocks(
                n_blocks_per_stage[0], conv_op, in_channels, features_per_stage[0], kernel_sizes[0], strides[0],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block_type
            )
        )
        for i in range(1, len(n_blocks_per_stage)):
            blocks.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[i], conv_op, features_per_stage[i - 1], features_per_stage[i], kernel_sizes[i],
                    strides[i], conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block_type
                )
            )

        return blocks

    def forward(self, input_image: torch.Tensor):

        # 1. 卷积编码器
        conv_enc_outputs = [self.conv_encoder_blocks[0](input_image)]
        for i in range(1, len(self.conv_encoder_blocks)):
            conv_enc_outputs.append(self.conv_encoder_blocks[i](conv_enc_outputs[-1]))

        lres_input = conv_enc_outputs[-1] # shape (batch_size, feature_dim, d, h, w)

        # 2. 分类头
        cls_pred_list = []
        for cls_head in self.cls_head_list:
            cls_pred_list.append(cls_head(lres_input)) 

        return cls_pred_list

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入：批次大小 2，通道 1，体积大小 (32, 32, 16) - 经过 5 次 2 倍下采样后 (1, 1, 0.5) 特征图维度太小，调整输入尺寸
    inputs = torch.randn(size=(2, 1, 64, 64, 32)).to(device) 

    # 实例化网络
    net = ResEncoder_only_cls(
        in_channels=1,
        cls_head_num_classes_list=[1, 4], # 两个分类头，分别输出 1 类和 4 类
        cls_drop_out_list=[0.1, 0.2],     # 不同的 dropout 率
        cls_query_num_list=[2, 8],        # 不同的查询数量
        use_cross_attention=True,
        num_heads=4,
        n_stages=6,
        features_per_stage=[16, 32, 64, 128, 256, 320], # 最后一个 feature dim 是 320
        kernel_sizes=[(3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
        strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        conv_op=nn.Conv3d,
        conv_bias=True,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        n_blocks_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        nonlin_first=True,
    ).to(device)

    with torch.no_grad():
        out = net(inputs)
        print("Test Output Shapes (Classification Logits):")
        # 期望输出形状：[B, 1] 和 [B, 4]
        for item in out:
            print(item.shape)