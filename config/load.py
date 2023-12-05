import yaml, os, torchvision.transforms as transforms
from types import SimpleNamespace
from datetime import datetime

def dict2object(d):
    if isinstance(d, dict): return SimpleNamespace(**{k: dict2object(v) for k, v in d.items()})
    elif isinstance(d, list): return [dict2object(x) for x in d]
    else: return d

def replace_fstring_in_dict(d): # Not used much... sadly
    for key, value in d.items():
        if isinstance(value, dict):
            replace_fstring_in_dict(value)
        elif isinstance(value, str):
            d[key] = value.format_map(d)

def merge_dicts(parent_dict, child_dict):
    ''' Merge two dicts, child overwrites parent '''
    merged_dict = {**parent_dict, **child_dict}
    for key, value in parent_dict.items():
        if isinstance(value, dict) and key in child_dict:
            merged_dict[key] = {**value, **child_dict[key]}
    return merged_dict

def args_from_yaml(fname_base='./base.yaml', fname_aux:str=None) -> SimpleNamespace:
    
    conf = merge_dicts(
        yaml.safe_load(open(fname_base, 'r')), 
        yaml.safe_load(open(fname_aux, 'r'))
    ) if fname_aux is not None else yaml.safe_load(open(fname_base, 'r'))

    conf['exp']['date_time'] = datetime.now().strftime("%m%d-%H:%M:%S")
    args = dict2object(conf)
    args.exp.exp_name = fname_aux.split('/')[-1].split('.')[0] if fname_aux is not None else fname_base.split('/')[-1].split('.')[0]

    # Format map by replacing `base_dir`, `exp_name`, `date_time` in config fstrings
    args.exp.log_dir= args.exp.log_dir.format_map(vars(args.exp))
    if args.ctrnet.pretrained_keypoint_seg_model_path is not None:
        args.ctrnet.pretrained_keypoint_seg_model_path = args.ctrnet.pretrained_keypoint_seg_model_path.format_map(vars(args.exp))
    args.ctrnet.urdf_file = args.ctrnet.urdf_file.format_map(vars(args.exp))
    args.ctrnet.meshobj_dir = args.ctrnet.meshobj_dir.format_map(vars(args.exp))
    args.ctrnet.checkpoint_path = args.ctrnet.checkpoint_path.format_map(vars(args.exp))
    args.ctrnet.lr = float(args.ctrnet.lr)
    args.ctrnet.kp_loss_weight = float(args.ctrnet.kp_loss_weight)

    # TODO: use args.ctrnet.kp_loss_weight

    # Set transformation for dataloader
    args.r2d2.trans_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.r2d2.mean, std=args.r2d2.std),
    ]) if args.r2d2.normalize else transforms.ToTensor()

    return args