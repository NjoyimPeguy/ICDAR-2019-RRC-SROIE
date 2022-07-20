import os
import sys

sys.path.append(os.getcwd())

import gc
import time
import torch
import warnings
import numpy as np

from options import TrainArgs
from tabulate import tabulate
from torch.utils.data import random_split

# importing useful function/methods
from functional.saving import Checkpointer
from functional.metric import AverageMeter
from functional.time import get_process_time
from functional.data.dataloader import Dataloader
from functional.visualizer import VisdomVisualizer
from functional.event_tracker import logger as Logger

from keyword_information_extraction.configs import configs
from keyword_information_extraction.data.dataset import SROIE2019Dataset, TrainBatchCollator
from keyword_information_extraction.model.charlm import CharacterLevelCNNHighwayBiLSTM as CharLM
from keyword_information_extraction.utils.misc import multilabel_confusion_matrix, check_denominator_consistency

# Put warnings to silence if any.
warnings.filterwarnings("ignore")

train_args = TrainArgs(description="Keyword Information Extraction: training")
parser = train_args.get_parser()
args, _ = parser.parse_known_args()


def main():
    # One can comment the line below.
    # It is important to note it is useful when some nasty errors like the NaN loss show up.
    torch.autograd.set_detect_anomaly(True)

    # A boolean to check whether the user is able to use cuda or not.
    use_cuda = torch.cuda.is_available() and args.use_cuda

    output_dir = os.path.normpath(configs.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # The declaration and initialization of the logger.
    logger = Logger(name="CharLM: training phase", output_dir=output_dir, log_filename="training-log")

    # The declaration and initialization of the checkpointer.
    checkpointer = Checkpointer(logger=logger, output_dir=output_dir)

    # The declaration of the training dataloader arguments.
    training_dataloader_args = dict(configs.DATALOADER.TRAINING)

    # Adding the collate_fn.
    training_dataloader_args["collate_fn"] = TrainBatchCollator(class_labels_padding_value=configs.LOSS.IGNORE_INDEX)

    # The declaration and tensor type of the CPU/GPU device.
    if not use_cuda:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Enabling cuDNN.
        torch.backends.cudnn.enabled = True

    training_dataloader_args["generator"] = torch.Generator(device=device)

    labels_classes = dict(configs.DATASET.LABELS_CLASSES)

    labels_classes = dict(sorted(labels_classes.items(), key=lambda item: item[1]))

    train_dataset = SROIE2019Dataset(directory_dict=dict(configs.DATASET.TRAIN), labels_classes=labels_classes)

    vocabulary = train_dataset.vocabulary

    text_max_length = train_dataset.text_max_length

    class_weights = torch.as_tensor(train_dataset.class_weights.tolist(), device=device, dtype=torch.float32)

    train_criterion = torch.nn.CrossEntropyLoss(weight=class_weights,
                                                ignore_index=configs.LOSS.IGNORE_INDEX,
                                                reduction="mean").to(device, non_blocking=True)

    training_dataloader_args["pin_memory"] = training_dataloader_args["pin_memory"] and use_cuda
    train_loader = Dataloader(dataset=train_dataset, is_train=True, **training_dataloader_args)

    logger.info("Declaration and initialization of the model and optimizer...")
    model_args = dict(configs.MODEL.ARGS)
    model = CharLM(n_classes=configs.DATASET.NUM_CLASSES,
                   max_seq_length=text_max_length,
                   char_vocab_size=len(vocabulary),
                   **model_args).to(device, non_blocking=True)

    logger.info("Declaration and initialization of the optimizer...")
    optimizer_args = dict(configs.SOLVER.ADAM.ARGS)
    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_args)

    logger.info("Declaration and initialization of the learning rate scheduler...")
    lr_scheduler_args = dict(configs.SOLVER.SCHEDULER.ARGS)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, **lr_scheduler_args)

    start_epoch = 1
    max_epochs = configs.SOLVER.MAX_EPOCHS
    checkpoint_data = {}
    last_elapsed_time = 0.0
    plotter = total_loss_plotter_window = None

    if args.resume is not None:

        logger.info("Resuming the training...")

        checkpoint_data = checkpointer.load(file_to_load=args.resume, map_location=device)

        start_epoch = checkpoint_data.get("epoch", 1)

        last_elapsed_time = checkpoint_data.get("elapsed_time", 0.0)

        total_loss_plotter_window = checkpoint_data.get("total_loss_plot_win")

        optimizer_state_dict = checkpoint_data.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        # Loading the learning rate scheduler state dict..
        lr_scheduler_state_dict = checkpoint_data.get("lr_scheduler_state_dict")
        if lr_scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        model_state_dict = checkpoint_data.get("model_state_dict")
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)

        # Loading the loss/criterion...
        train_loss = checkpoint_data.get("train_criterion")
        if train_loss is not None:
            train_criterion = loss

    if args.use_visdom:
        logger.info("Initialising the visualiser 'Visdom'")
        plotter = VisdomVisualizer(port=configs.VISDOM.PORT, env_name=configs.VISDOM.ENV_NAME)

    if plotter is not None and total_loss_plotter_window is None:
        logger.info("Plot creations...")
        title_name = "Character Level CNN + Highway + BiLSTM: training"
        legend_names = ["Training loss"]
        total_loss_plotter_window = plotter.createPlot(xLabel="Epochs", yLabel="Loss",
                                                       legend_names=legend_names, title_name=title_name)

    logger.info("About to start the training...")

    # Force the garbage collector to run.
    gc.collect()

    if use_cuda:
        # Before starting the training, all the unoccupied cached memory are released.
        torch.cuda.empty_cache()

        # waits for all tasks in the GPU to complete.
        torch.cuda.current_stream(device).synchronize()

    # Starting time.
    start_time = time.time()

    for current_epoch in range(start_epoch, max_epochs + 1):

        training_loss = train(model=model, optimizer=optimizer, train_loader=train_loader,
                              criterion=train_criterion, device=device, logger=logger,
                              current_epoch=current_epoch, max_epochs=max_epochs,
                              entities_names=list(labels_classes.keys()))

        # Update the learning rate.
        lr_scheduler.step(current_epoch)

        if use_cuda:
            # waits for all tasks in the GPU to complete
            torch.cuda.current_stream(device).synchronize()

        elapsed_time = last_elapsed_time + (time.time() - start_time)

        remaining_time, estimated_finish_time = get_process_time(start_time=start_time,
                                                                 elapsed_time=elapsed_time,
                                                                 current_epoch=current_epoch,
                                                                 max_epochs=max_epochs)

        logger.info("Elapsed time: {et} seconds || "
                    "Remaining time: {lt} seconds || "
                    "ETA: {eta}\n".format(
            et=elapsed_time,
            lt=remaining_time,
            eta=estimated_finish_time
        ))

        # Updating important data.
        checkpoint_data.update({
            "epoch": current_epoch + 1,
            "elapsed_time": elapsed_time,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict()
        })

        # Updating the plot if it was previously set (i.e., the plotter is not None)
        if plotter is not None and \
                current_epoch > 0 and \
                args.plot_steps is not None and \
                args.plot_steps > 0 and \
                current_epoch % args.plot_steps == 0:
            total_loss_data_y = [training_loss]
            plotter.update_plot(window=total_loss_plotter_window, data_x=current_epoch, data_y=total_loss_data_y)
            checkpoint_data.update({"total_loss_plot_win": total_loss_plotter_window})

        if current_epoch > 0 and \
                args.save_steps is not None and \
                args.save_steps > 0 and \
                current_epoch % args.save_steps == 0 and \
                current_epoch != max_epochs:

            logger.info("Saving a checkpoint at epoch {0}...\n".format(current_epoch))

            checkpointer.save(name="CHARLM_CHECKPOINT_EPOCH_{0}".format(current_epoch), data=checkpoint_data)

            # Saving plots...
            if plotter is not None:
                plotter.save()

    logger.info("Training has just finished. Saving the final checkpoint...\n")
    checkpointer.save(name="CHARLM_FINAL_CHECKPOINT", data=checkpoint_data)

    if use_cuda:
        # Just in case, we release all the unoccupied cached memory after training.
        torch.cuda.empty_cache()


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          device: torch.device,
          current_epoch: int,
          max_epochs: int,
          logger: Logger,
          entities_names: list):
    true_classes = []

    predicted_classes = []

    training_losses = AverageMeter(fmt=":.6f")

    model = model.train()

    for current_iteration, batch_samples in enumerate(train_loader):

        text_features, text_class_labels = batch_samples

        text_features = text_features.to(device, non_blocking=True)

        text_class_labels = text_class_labels.to(device, non_blocking=True)

        # Clearing out the model's gradients before doing backprop.
        # 'set_to_none=True' here can modestly improve performance
        optimizer.zero_grad(set_to_none=True)

        outputs = model(text_features)

        inputs = outputs.contiguous().view(-1, configs.DATASET.NUM_CLASSES)

        targets = text_class_labels.contiguous().view(-1)

        loss = criterion(inputs, targets)

        loss.backward()

        optimizer.step()

        # Calling ".item()" operation requires synchronization. For further info, check this out:
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html And this as well:
        # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
        # Just in case of in-place operations, one can add .clone() to avoid nasty modifications...
        training_losses.update(loss.detach().clone(), n=text_features.size(0))

        with torch.no_grad():

            _, predictions = torch.max(torch.softmax(outputs, dim=2), dim=2)

            predicted_classes.extend(predictions.contiguous().view(-1).tolist())

            true_classes.extend(text_class_labels.contiguous().view(-1).tolist())

        if current_iteration > 0 and args.log_steps is not None and \
                args.log_steps > 0 and current_iteration % args.log_steps == 0:
            logger.info("Epoch: {curr_epoch}/{maximum_epochs} || "
                        "Iteration: {iter} || "
                        "Learning rate: {lr} || "
                        "Loss: {loss}\n".format(
                curr_epoch=current_epoch,
                maximum_epochs=max_epochs,
                iter=current_iteration,
                lr=optimizer.param_groups[0]["lr"],
                loss=training_losses
            ))

    n_classes = configs.DATASET.NUM_CLASSES
    MCM = multilabel_confusion_matrix(inputs=predicted_classes, targets=true_classes, n_classes=n_classes)

    TPc = MCM[:, 1, 1]

    FPc = MCM[:, 0, 1]

    FNc = MCM[:, 1, 0]

    precision_denominator = check_denominator_consistency(TPc + FPc)
    precision = np.expand_dims(TPc / precision_denominator, axis=0).T

    recall_denominator = check_denominator_consistency(TPc + FNc)
    recall = np.expand_dims(TPc / recall_denominator, axis=0).T

    f1_denominator = check_denominator_consistency(precision + recall)
    f1_Score = 2.0 * ((precision * recall) / f1_denominator)

    data_list = np.concatenate([recall, precision, f1_Score], axis=1)

    grid_data_list = [] + configs.TABULATE.DATA_LIST
    for i, data in enumerate(data_list):
        rec = data[1]
        prec = data[0]
        hmean = data[2]
        grid_data_list.append([entities_names[i], rec, prec, hmean])

    training_loss = training_losses.global_avg

    logger.info("Training loss = {0}\n".format(training_loss))

    logger.info("Results:\n" + tabulate(grid_data_list, headers="firstrow", tablefmt="grid"))

    return training_loss


if __name__ == '__main__':

    # Guarding against bad arguments.

    if args.save_steps is not None and args.save_steps <= 0:
        raise ValueError("{0} is an invalid value for the argument: --save-steps".format(args.save_steps))
    elif args.log_steps is not None and args.log_steps <= 0:
        raise ValueError("{0} is an invalid value for the argument: --log-steps".format(args.log_steps))
    elif args.plot_steps is not None and args.plot_steps <= 0:
        raise ValueError("{0} is an invalid value for the argument: --plot-steps".format(args.plot_steps))
    elif args.resume is not None and not os.path.isfile(args.resume):
        raise ValueError("The path to the checkpoint data path_to_file is wrong!")

    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! "
                         "You may want to check it with 'nvidia-smi' or "
                         "'Task Manager' for Windows users.")

    if args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")

    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()

    main()
