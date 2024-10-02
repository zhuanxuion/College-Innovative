from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
import datahandler
from model import createDeepLabv3
from trainer import train_model


@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type the nu=int,
    help="Specifymber of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--inherit",
              default=False,
              type=bool,
              help="Whither or not start from local checkpoint")
def main(data_directory, exp_directory, epochs, batch_size, inherit=False):
    data_directory = Path(data_directory)
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    model = createDeepLabv3(exp_directory=exp_directory, inherit=inherit)
    model.train()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()
