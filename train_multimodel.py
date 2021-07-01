import argparse
import os
from subprocess import check_call
import sys

import utils.utils as utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/',
                    help='Directory containing params.json')

def launch_training_job(parent_dir, job_name, resolution, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir="{}_{}".format(job_name, resolution)
    model_dir = os.path.join(parent_dir, model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --model_type {job_name}".format(python=PYTHON, model_dir=model_dir, job_name=job_name)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    models = ['deeplab_GCN']

    for model in models:
        # Launch job (name has to be unique)
        job_name = "{}".format(model)
        resolution = "_lr_{}_b_{}".format(params.learning_rate, params.batch_size)
        launch_training_job(args.parent_dir, job_name, resolution, params)
