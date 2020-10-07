#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import logging
import string
import random
from datetime import datetime

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider

from sacred import Experiment
import seml

ex = Experiment()
seml.setup_logger(ex)

# TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARN')
tf.autograph.set_verbosity(1)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
                db_collection, overwrite=overwrite))

    out_emb_size = None
    int_emb_size = None
    basis_emb_size = None
    num_bilinear = None
    extensive = True


@ex.automain
def run(model_name, emb_size, out_emb_size, int_emb_size, basis_emb_size,
        num_blocks, num_bilinear, num_spherical, num_radial, output_init,
        extensive, num_before_skip, num_after_skip, num_dense_output,
        cutoff, envelope_exponent, dataset, num_train, num_valid,
        data_seed, num_steps, learning_rate, ema_decay,
        decay_steps, warmup_steps, decay_rate, batch_size,
        evaluation_interval, save_interval, restart, targets,
        comment, logdir):

    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
    def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    # Create directories
    # A unique directory name is created for this run based on the input
    if restart is None:
        if model_name == "dimenet":
            directory = (
                    logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id_generator()
                    + "_" + os.path.basename(dataset)
                    + "_e" + str(emb_size)
                    + "_bi" + str(num_bilinear)
                    + "_sbf" + str(num_spherical)
                    + "_rbf" + str(num_radial)
                    + "_b" + str(num_blocks)
                    + "_nbs" + str(num_before_skip)
                    + "_nas" + str(num_after_skip)
                    + "_no" + str(num_dense_output)
                    + "_cut" + str(cutoff)
                    + "_env" + str(envelope_exponent)
                    + f"_lr{learning_rate:.2e}"
                    + f"_dec{decay_steps:.2e}"
                    + "_" + '-'.join(targets)
                    + "_" + comment
            )
        elif model_name == "dimenet++":
            directory = (
                    logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id_generator()
                    + "_" + os.path.basename(dataset)
                    + "_e" + str(emb_size)
                    + "_oe" + str(out_emb_size)
                    + "_ie" + str(int_emb_size)
                    + "_be" + str(basis_emb_size)
                    + "_sbf" + str(num_spherical)
                    + "_rbf" + str(num_radial)
                    + "_b" + str(num_blocks)
                    + "_nbs" + str(num_before_skip)
                    + "_nas" + str(num_after_skip)
                    + "_no" + str(num_dense_output)
                    + "_cut" + str(cutoff)
                    + "_env" + str(envelope_exponent)
                    + f"_lr{learning_rate:.2e}"
                    + f"_dec{decay_steps:.2e}"
                    + "_" + '-'.join(targets)
                    + "_" + comment
            )
        else:
            raise ValueError(f"Unknown model name: '{model_name}'")
    else:
        directory = restart
    logging.info(f"Directory: {directory}")

    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_dir = os.path.join(directory, 'best')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    best_loss_file = os.path.join(best_dir, 'best_loss.npz')
    best_ckpt_file = os.path.join(best_dir, 'ckpt')
    step_ckpt_folder = log_dir

    # Initialize summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)

    train = {}
    validation = {}

    # Initialize metrics
    train['metrics'] = Metrics('train', targets, ex)
    validation['metrics'] = Metrics('val', targets, ex)

    with summary_writer.as_default():
        logging.info("Load dataset")
        data_container = DataContainer(dataset, cutoff=cutoff, target_keys=targets)

        # Initialize DataProvider (splits dataset into 3 sets based on data_seed and provides tf.datasets)
        data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                                     seed=data_seed, randomized=True)

        # Initialize datasets
        train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
        train['dataset_iter'] = iter(train['dataset'])
        validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)
        validation['dataset_iter'] = iter(validation['dataset'])

        logging.info("Initialize model")
        if model_name == "dimenet":
            model = DimeNet(
                    emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
                    num_spherical=num_spherical, num_radial=num_radial,
                    cutoff=cutoff, envelope_exponent=envelope_exponent,
                    num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                    num_dense_output=num_dense_output, num_targets=len(targets),
                    activation=swish, output_init=output_init)
        elif model_name == "dimenet++":
            model = DimeNetPP(
                    emb_size=emb_size, out_emb_size=out_emb_size,
                    int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
                    num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
                    cutoff=cutoff, envelope_exponent=envelope_exponent,
                    num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                    num_dense_output=num_dense_output, num_targets=len(targets),
                    activation=swish, extensive=extensive, output_init=output_init)
        else:
            raise ValueError(f"Unknown model name: '{model_name}'")

        logging.info("Prepare training")
        # Save/load best recorded loss (only the best model is saved)
        if os.path.isfile(best_loss_file):
            loss_file = np.load(best_loss_file)
            metrics_best = {k: v.item() for k, v in loss_file.items()}
        else:
            metrics_best = {k: np.inf for k in validation['metrics'].keys()}
            metrics_best['step'] = 0
            np.savez(best_loss_file, **metrics_best)

        # Initialize trainer
        trainer = Trainer(model, learning_rate, warmup_steps,
                          decay_steps, decay_rate,
                          ema_decay=ema_decay, max_grad_norm=1000)

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

        # Restore latest checkpoint
        ckpt_restored = tf.train.latest_checkpoint(log_dir)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored)

        if ex is not None:
            ex.current_run.info = {'directory': directory}

        # Training loop
        logging.info("Start training")
        steps_per_epoch = int(np.ceil(num_train / batch_size))

        if ckpt_restored is not None:
            step_init = ckpt.step.numpy()
        else:
            step_init = 1
        for step in range(step_init, num_steps + 1):
            # Update step number
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)

            # Perform training step
            trainer.train_on_batch(train['dataset_iter'], train['metrics'])

            # Save progress
            if (step % save_interval == 0):
                manager.save()

            # Check performance on the validation set
            if (step % evaluation_interval == 0):

                # Save backup variables and load averaged variables
                trainer.save_variable_backups()
                trainer.load_averaged_variables()

                # Compute averages
                for i in range(int(np.ceil(num_valid / batch_size))):
                    trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])

                # Update and save best result
                if validation['metrics'].mean_mae < metrics_best['mean_mae_val']:
                    metrics_best['step'] = step
                    metrics_best.update(validation['metrics'].result())

                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(best_ckpt_file)

                for key, val in metrics_best.items():
                    if key != 'step':
                        tf.summary.scalar(key + '_best', val)

                epoch = step // steps_per_epoch
                logging.info(
                    f"{step}/{num_steps} (epoch {epoch+1}): "
                    f"Loss: train={train['metrics'].loss:.6f}, val={validation['metrics'].loss:.6f}; "
                    f"logMAE: train={train['metrics'].mean_log_mae:.6f}, "
                    f"val={validation['metrics'].mean_log_mae:.6f}")

                train['metrics'].write()
                validation['metrics'].write()

                train['metrics'].reset_states()
                validation['metrics'].reset_states()

                # Restore backup variables
                trainer.restore_variable_backups()

    return {key + '_best': val for key, val in metrics_best.items()}
