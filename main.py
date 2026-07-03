import argparse
import importlib
import random

from models.snn import SAS
from models.utils import *

# NOTE(linux-port): main.py only ever runs ONE dataset (see --datasets dispatch below), but it
# historically imported all 13 loaders/models/trainers eagerly — so a missing OPTIONAL dep in any
# UNUSED dataset crashed every run (faiss for the speech-regression trainers, lmdb for the TU*
# loaders). We import each module defensively: a module whose deps are absent resolves to None and
# only matters if that dataset is actually selected. Subset datasets (ISRUC / MentalArithmetic /
# SEED-VIG) import cleanly. schoffelen2019 / gwilliams2022 stay excluded — they have a trainer but
# NO loader/model/process on disk (non-reproducible stubs, not among the paper's 13 datasets).
_DATASET_SUFFIXES = ['isruc', 'broderick2019', 'brennan2019', 'mumtaz2016', 'mental', 'shumi',
                     'tuab', 'tuev', 'bcic2020', 'seedvig', 'seedv', 'faced', 'physio']
for _pkg, _prefix in [('data_loader', 'data_'), ('models', 'model_'), ('trainers', 'trainer_')]:
    for _suf in _DATASET_SUFFIXES:
        _name = f'{_prefix}{_suf}'
        try:
            globals()[_name] = importlib.import_module(f'{_pkg}.{_name}')
        except ImportError as _err:
            globals()[_name] = None
            print(f"[linux-port] skip optional module {_pkg}.{_name} ({_err}); "
                  f"only affects --datasets that need it")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default="./data")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--early_stop_epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--multi_lr', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--datasets', type=str, default='ISRUC',
                        choices=['brennan2019', 'broderick2019',
                                 'SEED-VIG',
                                 'ISRUC', 'TUEV', 'BCIC2020', 'SEED-V', 'FACED', 'PhysioNet-MI',
                                 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'SHU-MI'])
    parser.add_argument('--model', type=str, default='cbramod', choices=['simplecnn', 'cbramod', 'labram'])
    parser.add_argument('--n_negatives', type=int, default=None)
    parser.add_argument('--n_subjects', type=int, default=1)
    parser.add_argument('--n_channels', type=int, default=None)
    parser.add_argument('--n_slice', type=int, default=1)
    parser.add_argument('--sr', type=int, default=None)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--C', type=float, default=0.2)

    parser.add_argument('--ckpt_snn', type=str, default=None)
    parser.add_argument('--ckpt_ann', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./ckpt")
    parser.add_argument('--load_lbm', action='store_true', default=False)
    parser.add_argument('--foundation_dir', type=str, default="./ckpt/cbramod-base.pth")
    parser.add_argument('--frozen_ann', action='store_true', default=False)
    parser.add_argument('--frozen_snn', action='store_true', default=False)
    parser.add_argument('--frozen_lbm', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()

    # check if evaluate
    if args.eval:
        args.frozen_ann = True
        args.frozen_snn = True
        args.max_epoch = 0

    # clear cache
    # for file_path in glob.glob(os.path.join(rf"{args.base_dir}\cache", "*")):
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

    # setup seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # obtain dataloader, model, trainer
    if args.datasets == 'ISRUC':
        args.n_classes = 5
        args.n_subjects = 100
        args.n_channels = 6
        args.sr = 200
        args.fps = 1
        data_loaders = data_isruc.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_isruc.Model(args)
        snn_model = SAS(args)
        trainer = trainer_isruc
    elif args.datasets == 'FACED':
        args.n_classes = 9
        args.n_channels = 32
        args.sr = 200
        args.fps = 1
        data_loaders = data_faced.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_faced.Model(args)
        snn_model = SAS(args)
        trainer = trainer_faced
    elif args.datasets == 'PhysioNet-MI':
        args.n_classes = 4
        args.n_channels = 64
        args.sr = 200
        args.fps = 5
        data_loaders = data_physio.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_physio.Model(args)
        snn_model = SAS(args)
        trainer = trainer_physio
    elif args.datasets == 'Mumtaz2016':
        args.n_subjects = 64
        args.n_channels = 19
        args.sr = 200
        args.fps = 10
        data_loaders = data_mumtaz2016.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_mumtaz2016.Model(args)
        snn_model = SAS(args)
        trainer = trainer_mumtaz2016
    elif args.datasets == 'MentalArithmetic':
        args.n_subjects = 36
        args.n_channels = 20
        args.sr = 200
        args.fps = 10
        data_loaders = data_mental.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_mental.Model(args)
        snn_model = SAS(args)
        trainer = trainer_mental
    elif args.datasets == 'SHU-MI':
        args.n_channels = 32
        args.sr = 200
        args.fps = 5
        data_loaders = data_shumi.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_shumi.Model(args)
        snn_model = SAS(args)
        trainer = trainer_shumi
    elif args.datasets == 'SEED-VIG':
        args.n_subjects = 1
        args.n_channels = 17
        args.sr = 200
        args.fps = 3
        data_loaders = data_seedvig.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_seedvig.Model(args)
        snn_model = SAS(args)
        trainer = trainer_seedvig
    elif args.datasets == 'SEED-V':
        args.n_classes = 5
        args.n_subjects = 16
        args.n_channels = 62
        args.sr = 200
        args.fps = 10
        data_loaders = data_seedv.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_seedv.Model(args)
        snn_model = SAS(args)
        trainer = trainer_seedv
    elif args.datasets == 'TUAB':
        args.n_subjects = 1
        args.n_channels = 16
        args.sr = 200
        args.fps = 2
        data_loaders = data_tuab.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_tuab.Model(args)
        snn_model = SAS(args)
        trainer = trainer_tuab
    elif args.datasets == 'TUEV':
        args.n_classes = 6
        args.n_subjects = 1
        args.n_channels = 16
        args.sr = 200
        args.fps = 4
        data_loaders = data_tuev.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_tuev.Model(args)
        snn_model = SAS(args)
        trainer = trainer_tuev
    elif args.datasets == 'BCIC2020':
        args.n_classes = 5
        args.n_subjects = 15
        args.n_channels = 16
        args.sr = 200
        args.fps = 10
        data_loaders = data_bcic2020.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_bcic2020.Model(args)
        snn_model = SAS(args)
        trainer = trainer_bcic2020
    elif args.datasets == 'broderick2019':
        args.n_negatives = 100
        args.n_subjects = 19
        args.n_channels = 128
        args.sr = 120
        args.fps = 5
        data_loaders = data_broderick2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_broderick2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_broderick2019
    elif args.datasets == 'brennan2019':
        args.n_negatives = 200
        args.n_subjects = 32
        args.n_channels = 60
        args.sr = 120
        args.fps = 3
        data_loaders = data_brennan2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_brennan2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_brennan2019
    # NOTE(linux-port): schoffelen2019 / gwilliams2022 dispatch removed — see import note above.

    # optimizer and scheduler
    backbone_params = []
    other_params = []
    for name, param in eeg_model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    for frozen, params in [
        (args.frozen_ann, eeg_model.parameters()),
        (args.frozen_snn, snn_model.parameters()),
        (args.frozen_lbm, backbone_params),
    ]:
        if frozen:
            for p in params: p.requires_grad = False

    if args.multi_lr:
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': args.lr, 'name': 'ann'},
            {'params': snn_model.parameters(), 'lr': args.lr, 'name': 'snn'},
            {'params': backbone_params, 'lr': args.lr / 5, 'name': 'lbm'}
        ], betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {'params': eeg_model.parameters(), 'lr': args.lr, 'name': 'ann'},
            {'params': snn_model.parameters(), 'lr': args.lr, 'name': 'snn'},
        ], betas=(0.9, 0.999), weight_decay=args.weight_decay)

    scheduler_ann = GroupCosineAnnealingLR(optimizer, group_index=0, T_max=args.max_epoch * len(data_loaders['train']), eta_min=1e-6, verbose=False, name='ann')
    scheduler_snn = GroupCosineAnnealingLR(optimizer, group_index=1, T_max=args.max_epoch * len(data_loaders['train']), eta_min=1e-6, verbose=False, name='snn')
    schedulers = [scheduler_ann, scheduler_snn]

    print(f"The ann contains {sum(p.numel() for p in other_params)} parameters.")
    print(f"The snn contains {sum(p.numel() for p in snn_model.parameters())} parameters.")
    print(f"The backbone contains {sum(p.numel() for p in backbone_params)} parameters.")
    
    # train
    trainer = trainer.Trainer(data_loaders, eeg_model, snn_model, optimizer, schedulers, args)
    trainer.train()

