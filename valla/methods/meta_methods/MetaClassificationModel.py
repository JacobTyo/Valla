from __future__ import absolute_import, division, print_function

import sklearn.metrics
from simpletransformers.classification import ClassificationModel
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
import logging
import math
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import sklearn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from simpletransformers.classification.classification_utils import (
    flatten_results,
)
from transformers.optimization import AdamW
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


logger = logging.getLogger(__name__)


class MetaClassificationModel(ClassificationModel):
    def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=None,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        **kwargs,
    ):

        """
        Initializes a MetaClassificationModel model. See simpletransformers.classification.ClassificationModel
        for information on the parent class.

        This class does NOT support sliding window as of now.

        Args:

        """

        super(MetaClassificationModel, self).__init__(model_type=model_type, model_name=model_name,
                                                      tokenizer_type=tokenizer_type, tokenizer_name=tokenizer_name,
                                                      num_labels=num_labels, weight=weight, args=args,
                                                      use_cuda=use_cuda, cuda_device=cuda_device,
                                                      onnx_execution_provider=onnx_execution_provider, **kwargs)

        assert not self.args.sliding_window, 'The MetaClassificationModel does not currently support sliding window ' \
                                             'funcationality.  Only a problem if working with longer texts.'
        assert self.args.n_gpu <= 1, 'this does not currently scale to multiple GPUs'
        assert self.args.gradient_accumulation_steps <= 1, 'this does not support gradient accumulation'

        # update args to include meta-parameters: meta_lr, meta_bsz, inner_iters, ways, shots
        self.meta_type = self.args.meta_type
        if self.meta_type == 'ANIL':
            self.model.classifier = l2l.algorithms.MAML(self.model.classifier, lr=self.args.inner_lr)
        elif self.meta_type == 'MAML':
            self.model.bert = l2l.algorithms.MAML(self.model.bert, lr=self.args.inner_lr)
            self.model.classifier = l2l.algorithms.MAML(self.model.classifier, lr=self.args.inner_lr)
        else:
            raise ValueError('unsupported meta type')

    def meta_train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        meta_test_train_df=None,
        meta_test_test_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and meta_test_train_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )

        self._move_model_to_device()

        meta_train_examples = (
            train_df["text"].astype(str).tolist(),
            train_df["labels"].tolist(),
        )

        # TODO: this hangs when not using cache?!?
        meta_train_dataset = self.load_and_cache_examples(
            meta_train_examples, verbose=verbose
        )

        # now make the meta-datasets
        # TODO: verify that this works correctly, not completely convinced that it will
        meta_train_dataset = l2l.data.MetaDataset(meta_train_dataset)
        # now make a task dataset
        meta_transforms = [
            FusedNWaysKShots(meta_train_dataset, n=self.args.ways, k=self.args.shots),
            # probably need to add a k-shot here, at least for training
            LoadData(meta_train_dataset),
            RemapLabels(meta_train_dataset),
            ConsecutiveLabels(meta_train_dataset),
        ]
        meta_train_dataset = l2l.data.TaskDataset(meta_train_dataset,
                                                  task_transforms=meta_transforms,
                                                  num_tasks=-1)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.meta_train(
            meta_train_dataset,
            self.args.meta_batch_size,
            self.args.num_outer_steps,
            self.args.num_inner_updates,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            verbose=verbose,
            meta_test_train_df=meta_test_train_df,
            meta_test_test_df=meta_test_test_df,
            **kwargs,
        )

        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.save_model(model=self.model)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_type, output_dir
                )
            )

        return global_step, training_details

    def fast_adapt(self,
                   adapt_batch,
                   eval_batch,
                   features,
                   dropout,
                   learner,
                   adaptation_steps,
                   no_evaluate=False
                ):

        # TODO: seems like there should be a zero_grad here somewhere - I don't think adapt changes it, so
        #  with multiple steps, is the learner adapting to an average gradient? Seems this should not be the case
        #  but not sure. Also, if we zero_grad, that probably affects the outer update?

        # batch is a list of dictionaries -- Separate data into adaptation/evaluation sets

        for step in range(adaptation_steps):
            loss, *_ = self._calculate_fast_adapt_loss(
                features,
                dropout,
                learner,
                adapt_batch,
                loss_fct=self.loss_fct,
                num_labels=self.num_labels,
                args=self.args,
            )

            learner.adapt(loss)
            if self.meta_type == 'MAML':
                features.adapt(loss)

        loss, acc = None, None

        if not no_evaluate:
            # now evaluate for outer loss - evaluation_data
            loss, logits = self._calculate_fast_adapt_loss(
                features,
                dropout,
                learner,
                eval_batch,
                loss_fct=self.loss_fct,
                num_labels=self.num_labels,
                args=self.args,
            )
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc = sum(preds == eval_batch['labels'].cpu().numpy())/eval_batch['labels'].size(0)

        return loss, acc, learner

    def meta_train(
        self,
        meta_train_task_dataset,
        meta_batch_size,
        num_outer_steps,
        num_inner_updates,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        meta_test_train_df=None,
        meta_test_test_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        features = model.bert
        dropout = model.dropout
        classifier = model.classifier
        args = self.args

        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

        t_total = (
                num_outer_steps
                // args.gradient_accumulation_steps
                * args.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            logger.warning('DataParallel has not been tested, performance may be incorrect.')
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                        num_outer_steps // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                        num_outer_steps // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                multi_label, **kwargs
            )

        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(
                    project=args.wandb_project,
                    config={**asdict(args)},
                    **args.wandb_kwargs,
                )
                wandb.run._label(repo="simpletransformers")
                self.wandb_run_id = wandb.run.id
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        # epochs
        for _ in train_iterator:
            model.train()
            model.zero_grad()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )

            # not sure if this is how it will work. . .
            meta_batch_iterator = tqdm(
                # meta_train_task_dataset,
                range(num_outer_steps),
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            # outer loop
            for step, _ in enumerate(meta_batch_iterator):

                model.zero_grad()

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                current_loss = 0

                for meta_bn in range(meta_batch_size):

                    meta_train_batch = self._get_inputs_dict(meta_train_task_dataset.sample())
                    meta_test_batch = self._get_inputs_dict(meta_train_task_dataset.sample())

                    learner = classifier.clone()

                    # fast-adapt houses the inner looop
                    if self.args.fp16:
                        with amp.autocast():
                            val_error, val_acc, _ = self.fast_adapt(meta_train_batch,
                                                                    meta_test_batch,
                                                                    features,
                                                                    dropout,
                                                                    learner,
                                                                    num_inner_updates)
                    else:
                        val_error, val_acc, _ = self.fast_adapt(meta_train_batch,
                                                                meta_test_batch,
                                                                features,
                                                                dropout,
                                                                learner,
                                                                num_inner_updates)

                    current_loss += val_error.item()

                    if self.args.fp16:
                        scaler.scale(val_error).backward()
                    else:
                        val_error.backward()

                current_loss = current_loss / meta_batch_size

                if show_running_loss:
                    meta_batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                # average the gradient across meta batches
                for p in model.parameters():
                    p.grad.data.mul_(1.0 / meta_batch_size)

                # do we want to enforce a gradent norm? Unclear - set as hyperparameter
                if self.args.fp16:
                    scaler.unscale_(optimizer)
                if args.optimizer == "AdamW":
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                if self.args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar(
                        "lr", scheduler.get_last_lr()[0], global_step
                    )
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss
                    if args.wandb_project or self.is_sweeping:
                        wandb.log(
                            {
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            }
                        )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir_current = os.path.join(
                        output_dir, "checkpoint-{}".format(global_step)
                    )

                    self.save_model(
                        output_dir_current, optimizer, scheduler, model=model
                    )

                if args.evaluate_during_training and (
                    args.evaluate_during_training_steps > 0
                    and global_step % args.evaluate_during_training_steps == 0
                ):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = self.meta_eval_model(
                        meta_test_train_df,
                        meta_test_test_df,
                        num_inner_updates,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )

                    output_dir_current = os.path.join(
                        output_dir, "checkpoint-{}".format(global_step)
                    )

                    if args.save_eval_checkpoints:
                        self.save_model(
                            output_dir_current,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )

                    training_progress_scores["global_step"].append(global_step)
                    training_progress_scores["train_loss"].append(current_loss)

                    for key in results:
                        training_progress_scores[key].append(results[key])

                    if meta_test_test_df is not None:
                        test_results = self.meta_eval_model(
                            meta_test_train_df,
                            meta_test_test_df,
                            num_inner_updates,
                            verbose=verbose
                            and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            **kwargs,
                        )
                        for key in test_results:
                            training_progress_scores["test_" + key].append(
                                test_results[key]
                            )

                    report = pd.DataFrame(training_progress_scores)
                    report.to_csv(
                        os.path.join(
                            args.output_dir, "training_progress_scores.csv"
                        ),
                        index=False,
                    )

                    if args.wandb_project or self.is_sweeping:
                        wandb.log(self._get_last_metrics(training_progress_scores))

                    for key, value in flatten_results(
                        self._get_last_metrics(training_progress_scores)
                    ).items():
                        try:
                            tb_writer.add_scalar(key, value, global_step)
                        except (NotImplementedError, AssertionError):
                            if verbose:
                                logger.warning(
                                    f"can't log value of type: {type(value)} to tensorboar"
                                )
                    tb_writer.flush()

                    if not best_eval_metric:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                    if best_eval_metric and args.early_stopping_metric_minimize:
                        if (
                            best_eval_metric - results[args.early_stopping_metric]
                            > args.early_stopping_delta
                        ):
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping:
                                if (
                                    early_stopping_counter
                                    < args.early_stopping_patience
                                ):
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(
                                            f" No improvement in {args.early_stopping_metric}"
                                        )
                                        logger.info(
                                            f" Current step: {early_stopping_counter}"
                                        )
                                        logger.info(
                                            f" Early stopping patience: {args.early_stopping_patience}"
                                        )
                                else:
                                    if verbose:
                                        logger.info(
                                            f" Patience of {args.early_stopping_patience} steps reached"
                                        )
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    else:
                        if (
                            results[args.early_stopping_metric] - best_eval_metric
                            > args.early_stopping_delta
                        ):
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping:
                                if (
                                    early_stopping_counter
                                    < args.early_stopping_patience
                                ):
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(
                                            f" No improvement in {args.early_stopping_metric}"
                                        )
                                        logger.info(
                                            f" Current step: {early_stopping_counter}"
                                        )
                                        logger.info(
                                            f" Early stopping patience: {args.early_stopping_patience}"
                                        )
                                else:
                                    if verbose:
                                        logger.info(
                                            f" Patience of {args.early_stopping_patience} steps reached"
                                        )
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    model.train()

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.meta_eval_model(
                    meta_test_train_df,
                    meta_test_test_df,
                    num_inner_updates,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                if meta_test_test_df is not None:
                    test_results = self.meta_eval_model(
                        meta_test_train_df,
                        meta_test_test_df,
                        num_inner_updates,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )
                    for key in test_results:
                        training_progress_scores["test_" + key].append(
                            test_results[key]
                        )

                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                for key, value in flatten_results(
                    self._get_last_metrics(training_progress_scores)
                ).items():
                    try:
                        tb_writer.add_scalar(key, value, global_step)
                    except (NotImplementedError, AssertionError):
                        if verbose:
                            logger.warning(
                                f"can't log value of type: {type(value)} to tensorboar"
                            )
                tb_writer.flush()

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                        best_eval_metric - results[args.early_stopping_metric]
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def meta_eval_model(
        self,
        meta_test_train_df,
        meta_test_test_df,
        num_inner_updates,
        multi_label=False,
        output_dir=None,
        verbose=True,
        silent=False,
        wandb_log=True,
        update_epochs=20,
        update_only_classifier=False,
        **kwargs,
    ):
        """
        This adapts a model on a meta-test-train set, and then evaluates it on the meta-test-test set. 
        This simply adapts the classification layer of the model, in order to fine-tune the entire network, 
        just load the saved model into the normal BertAA script.  
        
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"
        self._move_model_to_device()

        meta_train_examples = (
            meta_test_train_df["text"].astype(str).tolist(),
            meta_test_train_df["labels"].tolist(),
        )

        meta_test_examples = (
            meta_test_test_df["text"].astype(str).tolist(),
            meta_test_test_df["labels"].tolist(),
        )

        meta_train_dataset = self.load_and_cache_examples(
            meta_train_examples, verbose=verbose
        )

        meta_test_dataset = self.load_and_cache_examples(
            meta_test_examples, verbose=verbose, evaluate=True, silent=silent
        )

        # instead of the MetaDatasets, just use normal data loaders
        meta_train_sampler = RandomSampler(meta_train_dataset)
        meta_train_dataloader = DataLoader(meta_train_dataset, sampler=meta_train_sampler,
                                           batch_size=self.args.train_batch_size)

        meta_test_sampler = SequentialSampler(meta_test_dataset)
        meta_test_dataloader = DataLoader(meta_test_dataset, sampler=meta_test_sampler,
                                          batch_size=self.args.eval_batch_size)

        self.model.train()
        # now run adapt on the meta_train_sampler
        features = self.model.bert
        dropout = self.model.dropout
        # just want a single learner for this, and use clone and adapt to only update this learner.
        # try a hail marry to prevent memory issues. . .
        i = np.random.randint(1000, 99999999)
        tmp_save_path = f'tmp_saved_model_{i}.torch'
        torch.save(self.model.state_dict(), tmp_save_path)
        learner = self.model.classifier

        # try full model for now
        if update_only_classifier:
            optimizer = AdamW(self.model.classifier.parameters(), lr=self.args.inner_lr, eps=self.args.adam_epsilon)
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.args.inner_lr, eps=self.args.adam_epsilon)

        best_macro_accuracy = 0
        best_accuracy = 0
        best_step = 0

        # for now just say update for one epoch and track eval loss the whole way?
        for epoch in range(20):
            for batch_num, batch in enumerate(tqdm(meta_train_dataloader, desc='meta evaluation')):
                batch = self._get_inputs_dict(batch)
                self.model.zero_grad()
                features.zero_grad()
                dropout.zero_grad()
                learner.zero_grad()
                # make sure learner is being changed

                # testing this as a work-around, using .clone() and fast_adapt causes a memory leak
                for step in range(self.args.num_inner_updates):
                    loss, *_ = self._calculate_fast_adapt_loss(
                        features,
                        dropout,
                        learner,
                        batch,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )

                    wandb.log({'meta_test_train_curve': loss.item()})

                    loss.backward()
                    optimizer.step()

                # now eval full batch
                if (epoch == 0 and batch_num in [0, 5, 50, 500, 5000]) or (batch_num == len(meta_train_dataloader)-1):
                    with torch.no_grad():
                        # pretty sure self.model.eval would do the trick but just incase
                        self.model.eval()
                        features.eval()
                        dropout.eval()
                        learner.eval()
                        preds = []
                        all_labels = []
                        for eval_batch_num, eval_batch in enumerate(meta_test_dataloader):
                            eval_batch = self._get_inputs_dict(eval_batch)
                            labels = eval_batch['labels']

                            tmp = {k: v for k, v in eval_batch.items() if k != 'labels'}
                            tmp = features(**tmp)
                            tmp = dropout(tmp[1])
                            logits = learner(tmp)

                            preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())
                            all_labels.extend(labels.cpu().numpy().tolist())

                        _acc = sklearn.metrics.accuracy_score(preds, all_labels)
                        mac_acc = sklearn.metrics.balanced_accuracy_score(preds, all_labels)
                        log_num = (epoch)*len(meta_train_dataloader)+batch_num
                        # results = {f'{log_num}_accuracy': _acc,
                        #            f'{log_num}_macro_accuracy': mac_acc}

                        if best_macro_accuracy < mac_acc:
                            best_macro_accuracy = mac_acc
                            best_accuracy = _acc
                            best_step = log_num

                        # print(results)

                        # if wandb_log:
                        # wandb.log(results)

                        self.model.train()
                        features.train()
                        dropout.train()
                        learner.train()

        results = {'macro_accuracy': best_macro_accuracy,
                   'accuracy': best_accuracy,
                   'best_step': best_step}

        wandb.log(results)

        # restore the classifier parameters
        self.model.load_state_dict(torch.load(tmp_save_path))
        # remove temporary file
        os.remove(tmp_save_path)
        return results

    #def _calculate_fast_adapt_loss(self, model, inputs, loss_fct, num_labels, args):
    def _calculate_fast_adapt_loss(self, features, dropout, classifier, inputs, loss_fct, num_labels, args):
        labels = inputs['labels']

        tmp = {k: v for k, v in inputs.items() if k != 'labels'}
        tmp = features(**tmp)
        tmp = dropout(tmp[1])
        logits = classifier(tmp)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)