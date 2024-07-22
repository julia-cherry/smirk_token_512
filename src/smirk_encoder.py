import torch
import torch.nn.functional as F
from torch import nn
import timm
from src.resnet50 import ResNet50


def create_backbone(backbone_name, pretrained=True):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class PoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_small_minimal_100')
        
        self.pose_cam_layers = nn.Sequential(
            nn.Linear(feature_dim, 6)
        )

        self.init_weights()

    def init_weights(self):
        self.pose_cam_layers[-1].weight.data *= 0.001
        self.pose_cam_layers[-1].bias.data *= 0.001

        self.pose_cam_layers[-1].weight.data[3] = 0
        self.pose_cam_layers[-1].bias.data[3] = 7


    def forward(self, img):
        features = self.encoder(img)[-1]  #(bs,576,7,7)  
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)   #(bs,6)

        outputs = {}

        pose_cam = self.pose_cam_layers(features).reshape(img.size(0), -1)
        outputs['pose_params'] = pose_cam[...,:3]  
        outputs['cam'] = pose_cam[...,3:]

        return outputs


class ShapeEncoder(nn.Module):
    def __init__(self, n_shape=300) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')

        self.shape_layers = nn.Sequential(
            nn.Linear(feature_dim, n_shape)
        )

        self.init_weights()


    def init_weights(self):
        self.shape_layers[-1].weight.data *= 0
        self.shape_layers[-1].bias.data *= 0


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        parameters = self.shape_layers(features).reshape(img.size(0), -1)

        return {'shape_params': parameters}


class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw(上下颚) + eyelid（眼睑）
        )

        self.n_exp = n_exp
        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)

        return outputs

class TokenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder = ResNet50(num_classes=1000)
        feature_dim_list = [1024, 512, 1024, 512, 512]
        self.token_layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim_list[i], 512)) for i in range(5)])
        
        self.init_weights()

    def init_weights(self):
        for layer in self.token_layers:
            layer[-1].weight.data *= 0.001
            layer[-1].bias.data *= 0.001

            layer[-1].weight.data[3] = 0
            layer[-1].bias.data[3] = 7


    def forward(self, img):
        # print('-----------')
        # print(self.encoder(img)[-1].shape) #(1024,16,16) 1024 feature[-1] 1
        # print(self.encoder(img)[-2].shape) #(512,32,32) 512 feature[-2] 1 2**(2-2)
        # print(self.encoder(img)[-3].shape) #(256,64,64) 1024 feature[-3] 2 2**(3-2)
        # print(self.encoder(img)[-4].shape) #(128,128,128) 512 feature[-4] 2 2**(4-3)
        # print(self.encoder(img)[-5].shape) #(32,128,128) 512 feature[-5] 4 2**(5-3)
        # print(len(self.encoder(img)))
        token_list = []
        features = self.encoder(img)
        feature = features[-1]  #(bs,1024,16,16)  
        feature = F.adaptive_avg_pool2d(feature, (1, 1)).squeeze(-1).squeeze(-1)   #(bs,1024)
        token = self.token_layers[0](feature).reshape(img.size(0), -1)
        token_list.append(token)
        for i in range(2,6):
            feature = features[-i]
            if i == 2 or i == 3:
                feature = F.adaptive_avg_pool2d(feature, (2**(i-2), 2**(i-2))).flatten(start_dim=1)
            else:
                feature = F.adaptive_avg_pool2d(feature, (2**(i-3), 2**(i-3))).flatten(start_dim=1)
            # print(i)
            # print(feature.shape)
            token = self.token_layers[i-1](feature).reshape(img.size(0), -1)
            token_list.append(token)
        stacked_tensors = torch.stack(token_list, dim=0)
        return stacked_tensors


class SmirkEncoder(nn.Module):
    def __init__(self, n_exp=50, n_shape=300) -> None:
        super().__init__()

        self.pose_encoder = PoseEncoder()

        self.shape_encoder = ShapeEncoder(n_shape=n_shape)

        self.expression_encoder = ExpressionEncoder(n_exp=n_exp) 
        
        self.token_encoder = TokenEncoder()

    def forward(self, img, img_512):
        pose_outputs = self.pose_encoder(img)
        shape_outputs = self.shape_encoder(img)
        expression_outputs = self.expression_encoder(img)
        token_outputs = self.token_encoder(img_512)

        outputs = {}
        outputs.update(pose_outputs)
        outputs.update(shape_outputs)
        outputs.update(expression_outputs)
        outputs['token'] = token_outputs

        return outputs
