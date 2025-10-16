import logging
import os
import wandb
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from utils.methods.weight_methods import WeightMethods



def main(path, lr, bs, device):

    weight_method = WeightMethods(
        args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=args.method_params_lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)



    # some extra statistics we save during training
    loss_list = []

    for epoch in 10:
            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                    calc_loss(train_pred[2], train_normal, "normal"),
                )
            )

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            # for record intermediate statistics
            loss_list.append(losses.detach().cpu())
            optimizer.step()

            if "famo" in args.method:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                        (
                            calc_loss(train_pred[0], train_label, "semantic"),
                            calc_loss(train_pred[1], train_depth, "depth"),
                            calc_loss(train_pred[2], train_normal, "normal"),
                        )
                    )
                    weight_method.method.update(new_losses.detach())


        # scheduler
        scheduler.step()



if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish()