from torch import nn
from typing import Optional, Dict, Union, Tuple
from abc import ABCMeta
import inspect
from torch.nn.modules.batchnorm import _BatchNorm, _InstanceNorm

# ---------BUILD CONVOLUTION LAYER------------
def build_conv_layer(cfg, *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    
    CONV_LAYERS = {
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'Conv3d': nn.Conv3d,
        'Conv': nn.Conv2d
    }
    
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


# ---------BUILD NORMALIZATION LAYER------------
# Infer abbreviation for normalization layers
def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


# Build normalization layer
def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    NORM_LAYERS = {
        'BN': nn.BatchNorm2d,
        'BN1d': nn.BatchNorm1d,
        'BN3d': nn.BatchNorm3d,
        'BN2d': nn.BatchNorm2d,
        'SyncBN': nn.SyncBatchNorm,
        'GN': nn.GroupNorm,
        'LN': nn.LayerNorm,
        'IN': nn.InstanceNorm2d,
        'IN1d': nn.InstanceNorm1d,
        'IN3d': nn.InstanceNorm3d,
    }

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer



# ---------BUILD INITIALIZATION METHODS------------
# Kaiming initialization is a method for initializing the weights of neural networks
# to help with training deep networks. It is particularly useful for layers
# that use ReLU or similar activation functions. The method is designed to keep the
# variance of the outputs of each layer approximately equal to the variance of its inputs,
# which helps to prevent vanishing or exploding gradients during training.
# It is named after Kaiming He, one of the researchers who proposed it.
def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# Xavier initialization is a method for initializing the weights of neural networks
# to help with training deep networks. It is designed to keep the variance of the outputs
# of each layer approximately equal to the variance of its inputs, which helps to prevent
# vanishing or exploding gradients during training. It is named after Xavier Glorot,
# one of the researchers who proposed it. Xavier initialization can be done using either
# a uniform or normal distribution, and it is particularly useful for layers that use
# sigmoid or tanh activation functions. The gain parameter can be adjusted to scale the
# initialization based on the activation function used.
# The gain is typically set to 1 for tanh and sigmoid activations, and can be adjusted
# for other activation functions.
# The bias parameter is used to initialize the bias term of the layer, if it exists.
# If the bias is not specified, it defaults to 0.
# The distribution parameter specifies whether to use a uniform or normal distribution
# for the weight initialization. If not specified, it defaults to 'normal'.
# Xavier initialization is often used in deep learning frameworks to improve the convergence
# and performance of neural networks, especially in the early stages of training.
# It is particularly effective for deep networks with many layers, as it helps to maintain
# a stable distribution of activations throughout the network.
# It is also known as Glorot initialization, named after one of the researchers who proposed
# it in their paper "Understanding the difficulty of training deep feedforward neural networks"
# (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# Constant initialization sets the weights of a neural network module to a constant value.
# This is often used to initialize the weights of layers in a neural network to a specific value
# before training begins. The constant value can be specified as a parameter, and the bias
# can also be initialized to a different constant value if desired.
# This method is useful for initializing layers that do not require complex weight distributions,
# such as fully connected layers or convolutional layers where a specific constant value is desired.

def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ---------BUILD ACTIVATION LAYERS------------
# Activation layers are used in neural networks to introduce non-linearity

def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    
    act_type = cfg.pop('type')
    if act_type == 'ReLU':
        return nn.ReLU(**cfg)
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(**cfg)
    elif act_type == 'PReLU':
        return nn.PReLU(**cfg)
    elif act_type == 'ELU':
        return nn.ELU(**cfg)
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    elif act_type == 'Softmax':
        return nn.Softmax(**cfg)
    else:
        raise KeyError(f'Unrecognized activation type {act_type}')


# ---------BUILD PADDING LAYER------------
def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    
    PADDING_LAYER = {'zeros': nn.ZeroPad2d,
                     'reflect': nn.ReflectionPad2d,
                     'replicate': nn.ReplicationPad2d
                    }

    cfg_ = cfg.copy()
    pad_type = cfg_.pop('type')
    if pad_type not in PADDING_LAYER:
        raise KeyError(f'Unrecognized padding type {pad_type}')
    return PADDING_LAYER[pad_type](*args, **kwargs, **cfg_)