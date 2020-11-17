# -*- coding:utf-8 -*-
from Networks.ResNet import ResNet34,ResNet50,ResNet101
from Networks.ResNet_CBAM import ResNet34_CBAM,ResNet50_CBAM,ResNet101_CBAM
from Networks.ResNeXt import ResNeXt34,ResNeXt50,ResNeXt101
from Networks.Res2Net import Res2Net50,Res2Net101_26w_4s
from Networks.SENet import SE_ResNet34,SE_ResNet50,SE_ResNet101
from Networks.EfficientNet_yy import EfficientNetB0
from Networks.EfficientNet_hl import EfficientNet
from Networks.DenseNet import Densenet121,Densenet161,Densenet169,Densenet201
from Networks.AG_HRNet import HRNet30, HRNet64

def Model(type, num_classes):
	if type == 'Resnet34':
		return ResNet34(num_classes=num_classes) # 1
	elif type == 'Resnet50':
		return ResNet50(num_classes=num_classes) # 2
	elif type == 'Resnet101':
		return ResNet101(num_classes=num_classes) # 3
	elif type == 'ResNet34_CBAM':
		return ResNet34_CBAM(num_classes=num_classes)  # 4
	elif type == 'ResNet50_CBAM':
		return ResNet50_CBAM(num_classes=num_classes) # 5
	elif type == 'ResNet101_CBAM':
		return ResNet101_CBAM(num_classes=num_classes) # 6
	elif type == 'Resnext34':
		return ResNeXt34(num_classes=num_classes) # 7
	elif type == 'Resnext50':
		return ResNeXt50(num_classes=num_classes) # 8
	elif type == 'Resnext101':
		return ResNeXt101(num_classes=num_classes) # 9
	elif type == 'Res2net50':
		return Res2Net50(num_classes=num_classes) # 10
	elif type == 'Res2net101':
		return Res2Net101_26w_4s(num_classes=num_classes) # 11
	elif type == 'SEnet34':
		return SE_ResNet34(num_classes=num_classes) # 12
	elif type == 'SEnet50':
		return SE_ResNet50(num_classes=num_classes) # 13
	elif type == 'SEnet101':
		return SE_ResNet101(num_classes=num_classes) # 14
	elif type == 'EfficientnetPre':
		return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes) # 15
	elif type == 'EfficientnetB0':
		return EfficientNetB0(num_classes=num_classes) # 16
	elif type == 'Densenet121':
		return Densenet121(num_classes=num_classes) # 16
	elif type == 'Densenet161':
		return Densenet161(num_classes=num_classes) # 16
	elif type == 'Densenet169':
		return Densenet169(num_classes=num_classes) # 16
	elif type == 'Densenet201':
		return Densenet201(num_classes=num_classes) # 16
	elif type == 'HRNet30':
		return HRNet30(num_classes=num_classes) # 16
	elif type == 'HRNet64':
		return HRNet64(num_classes=num_classes) # 16
	else:
		return None