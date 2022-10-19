import wandb
import argparse


if __name__ == "__main__":
    # get command line args
    parser = argparse.ArgumentParser(description='Get args for building train/test splits of IMDB dataset')

    parser.add_argument('--project', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_dir', type=str)
    args = parser.parse_args()

    # initialize a run to log the datasets
    run = wandb.init(
        project=args.project,
        name=f'{args.project}_dataset',
        job_type="upload-raw-dataset"
    )

    # log the raw data
    raw_data_artifact = wandb.Artifact(args.dataset_name, "dataset")
    raw_data_artifact.add_dir(args.dataset_dir)
    run.log_artifact(raw_data_artifact)

    # finish the run
    run.finish()
