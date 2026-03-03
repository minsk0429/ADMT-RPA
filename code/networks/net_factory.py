from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT
from networks.vnet import VNet

def net_factory(net_type="unet", in_chns=1, class_num=3, ema=False):
    # .cuda()를 제거하여 CPU/GPU 선택을 호출측에서 결정하도록 함
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num)
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True)
    else:
        net = None
    
    # EMA 모델인 경우 파라미터를 detach
    if ema and net is not None:
        for param in net.parameters():
            param.detach_()
    
    return net
