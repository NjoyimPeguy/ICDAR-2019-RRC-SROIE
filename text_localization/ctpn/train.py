import os
import sys

sys.path.append(os.getcwd())

import gc
import time
import torch
import warnings

from options import TrainArgs
from functional.saving import Checkpointer
from functional.metric import AverageMeter
from functional.event_tracker import Logger
from functional.time import get_process_time
from functional.data.dataloader import Dataloader
from functional.visualizer import VisdomVisualizer

from text_localization.ctpn.model import CTPN
from text_localization.ctpn.configs import configs
from text_localization.ctpn.losses import MultiBoxLoss
from text_localization.ctpn.data.dataset import SROIE2019Dataset

# Put warnings to silence.
warnings.filterwarnings("ignore")

train_args = TrainArgs(description="Text localization: training")
parser = train_args.get_parser()
args, _ = parser.parse_known_args()


def run():
    # It is important to note it is useful when some nasty errors like the NaN loss show up.
    torch.autograd.set_detect_anomaly(True)

    # A boolean to check whether the user is able to use cuda or not.
    use_cuda = torch.cuda.is_available() and args.use_cuda

    # A boolean to check whether the user is able to use amp or not.
    use_amp = args.use_amp and use_cuda

    # If False, autocast and GradScaler’s calls become no-ops.
    # This allows switching between default precision and mixed precision without if/else statements.
    # A different init scale is used here. Check this issue for further info:
    # https://github.com/pytorch/pytorch/issues/40497
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=2.0 ** 13)

    # Creation of the output directory where all events will be stored.
    output_dir = os.path.normpath(configs.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # The declaration and initialization of the logger.
    logger = Logger(name="CTPN: training phase", output_dir=output_dir, log_filename="training-log")

    logger.info("Declaration and initialization of the checkpointer...")
    checkpointer = Checkpointer(logger=logger, output_dir=output_dir)

    # The declaration of the training dataloader arguments.
    training_dataloader_args = dict(configs.DATALOADER.ARGS)

    # The declaration and tensor type of the CPU/GPU device.
    if not use_cuda:
        device = torch.device("cpu")  # By default, the device is CPU.
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # Enabling cuDNN.
        torch.backends.cudnn.enabled = True

        if args.resume is None:
            # Causing cuDNN to benchmark multiple convolution algorithms and select the fastest.
            torch.backends.cudnn.benchmark = True

    # Adding the generator with the right device.
    training_dataloader_args["generator"] = torch.Generator(device=device)

    if training_dataloader_args["num_workers"] > 0:
        torch.multiprocessing.set_start_method("spawn")

    logger.info("Declaration and initialization of the model...")
    model_args = dict(configs.MODEL.ARGS)
    model = CTPN(**model_args).to(device, non_blocking=True)

    logger.info("Declaration and initialization of the multi-box loss...")
    multiBoxLoss = MultiBoxLoss(configs)

    logger.info("Declaration and initialization of the optimizer...")
    optimizer_args = dict(configs.SOLVER.ADAM.ARGS)
    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_args)

    start_epoch, max_epochs = 1, configs.SOLVER.MAX_EPOCHS

    plotter = regr_loss_plotter_window = cls_loss_plotter_window = total_loss_plotter_window = None

    last_elapsed_time = 0.0

    checkpoint_data = {}

    if args.resume is not None:

        logger.info("Resuming the training...")

        checkpoint_data = checkpointer.load(file_to_load=args.resume, map_location=torch.device(device))

        start_epoch = checkpoint_data.get("start_epoch", 1)

        last_elapsed_time = checkpoint_data.get("elapsed_time", 0.0)

        # Loading the plot windows.
        regr_loss_plotter_window = checkpoint_data.get("regr_loss_plot_win")
        cls_loss_plotter_window = checkpoint_data.get("cls_loss_plot_win")
        total_loss_plotter_window = checkpoint_data.get("total_loss_plot_win")

        # Loading the model amp state.
        amp_state = checkpoint_data.get("amp_state")
        if amp_state is not None:
            grad_scaler.load_state_dict(amp_state)

        # Loading the optimizer state dict.
        optimizer_state_dict = checkpoint_data.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        # Loading the model state dict.
        model_state_dict = checkpoint_data.get("model_state_dict")
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)

    if args.use_visdom:
        logger.info("Declaration and initialization of the visualiser 'Visdom'...")
        plotter = VisdomVisualizer(port=configs.VISDOM.PORT, env_name=configs.VISDOM.ENV_NAME)

    # Creating a plotter window in visdom environment.
    if plotter is not None and \
            regr_loss_plotter_window is None and \
            cls_loss_plotter_window is None and \
            total_loss_plotter_window is None:
        title_name = "CTPN: training"
        legend_names = ["Training loss"]

        logger.info("Plot creations...")
        regr_loss_plotter_window = plotter.createPlot(xLabel="Epochs", yLabel="Localisation Loss",
                                                      legend_names=legend_names, title_name=title_name)
        cls_loss_plotter_window = plotter.createPlot(xLabel="Epochs", yLabel="Confidence Loss",
                                                     legend_names=legend_names, title_name=title_name)
        total_loss_plotter_window = plotter.createPlot(xLabel="Epochs", yLabel="Loss",
                                                       legend_names=legend_names, title_name=title_name)

    logger.info("Loading the dataset...")

    train_dataset = SROIE2019Dataset(configs, is_train=True)
    
    training_dataloader_args["pin_memory"] = training_dataloader_args["pin_memory"] and use_cuda

    train_loader = Dataloader(dataset=train_dataset, is_train=True, **training_dataloader_args)

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

    # Setting the model in training mode...
    model = model.train()

    for current_epoch in range(start_epoch, max_epochs + 1):

        regression_losses = AverageMeter(fmt=":.6f")
        classification_losses = AverageMeter(fmt=":.6f")
        total_losses = AverageMeter(fmt=":.6f")

        for current_iteration, batch_samples in enumerate(train_loader):

            images, targets = batch_samples

            batch_size = images.size(0)

            # Putting into the right device...
            images = images.to(device, non_blocking=True)

            # targets:0 = boxes, targets:1 = labels
            ground_truths = (targets[0].to(device, non_blocking=True), targets[1].to(device, non_blocking=True))

            # Clearing out the gradient before doing backprop.
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):

                predictions = model(images)

                regression_loss, classification_loss = multiBoxLoss(predictions, ground_truths)

                loss = regression_loss + classification_loss

            # Exits autocast before backward().
            # Backward passes under autocast are not recommended.
            # Calls backward() on scaled loss to create scaled gradients.
            grad_scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimiser's assigned kwargs.
            # If these gradients do not contain Infs or NaNs, optimiser.step() is then called,
            # otherwise, optimiser.step() is skipped.
            grad_scaler.step(optimizer)

            # grad_scaler.update() should only be called once,
            # after all optimizers used this iteration have been stepped:
            grad_scaler.update()

            # Reduce loss over all GPUs for logging purposes.
            # Calling ".item()" operation requires synchronization. For further info, check this out:
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html And this as well:
            # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
            # Just in case of in-place operations, one can add .clone() to avoid nasty modifications...
            regression_losses.update(value=regression_loss.detach().clone(), n=batch_size)
            classification_losses.update(value=classification_loss.detach().clone(), n=batch_size)
            total_losses.update(value=loss.detach().clone(), n=batch_size)

            if current_iteration > 0 and args.log_steps is not None and \
                    args.log_steps > 0 and current_iteration % args.log_steps == 0:
                logger.info("Epoch: {curr_epoch}/{maximum_epochs} || "
                            "Iteration: {iter} || "
                            "Learning rate: {lr}\n"
                            "Regression loss: {regr_loss} || "
                            "Classification loss: {cls_loss} || "
                            "Total loss: {total_loss}\n\n".format(
                    curr_epoch=current_epoch,
                    maximum_epochs=max_epochs,
                    iter=current_iteration,
                    lr=optimizer.param_groups[0]["lr"],
                    regr_loss=regression_losses,
                    cls_loss=classification_losses,
                    total_loss=total_losses
                ))

        if use_cuda:
            # waits for all tasks in the GPU to complete
            torch.cuda.current_stream(device).synchronize()

        elapsed_time = last_elapsed_time + (time.time() - start_time)

        remaining_time, estimated_finish_time = get_process_time(start_time=start_time,
                                                                 elapsed_time=elapsed_time,
                                                                 current_epoch=current_epoch,
                                                                 max_epochs=max_epochs)

        logger.info("Regression training loss: {regr_loss: .6f} || "
                    "Classification training loss: {cls_loss: .6f} || "
                    "Total training loss: {total_loss: .6f}\n"
                    "Elapsed time: {et} seconds || "
                    "Remaining time: {lt} seconds || "
                    "ETA: {eta}\n".format(
            regr_loss=regression_losses.global_avg,
            cls_loss=classification_losses.global_avg,
            total_loss=total_losses.global_avg,
            et=elapsed_time,
            lt=remaining_time,
            eta=estimated_finish_time
        ))

        # Updating important data.
        checkpoint_data.update({
            "elapsed_time": elapsed_time,
            "start_epoch": current_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimizer.state_dict(),
        })

        if use_amp:
            checkpoint_data.update({"amp_state": grad_scaler.state_dict()})

        # Updating Visdom plots if any
        if plotter is not None and \
                current_epoch > 0 and \
                args.plot_steps is not None and \
                args.plot_steps > 0 and \
                current_epoch % args.plot_steps == 0:
            regr_loss_data_y = [regression_losses.global_avg]
            cls_loss_data_y = [classification_losses.global_avg]
            total_loss_data_y = [total_losses.global_avg]
            plotter.update_plot(window=regr_loss_plotter_window, data_x=current_epoch,
                                data_y=regr_loss_data_y)
            plotter.update_plot(window=cls_loss_plotter_window, data_x=current_epoch,
                                data_y=cls_loss_data_y)
            plotter.update_plot(window=total_loss_plotter_window, data_x=current_epoch,
                                data_y=total_loss_data_y)

            # Updating plots...
            checkpoint_data.update({"regr_loss_plot_win": regr_loss_plotter_window})
            checkpoint_data.update({"cls_loss_plot_win": cls_loss_plotter_window})
            checkpoint_data.update({"total_loss_plot_win": total_loss_plotter_window})

        # Saving useful data...
        if current_epoch > 0 and \
                args.save_steps is not None and \
                args.save_steps > 0 and \
                current_epoch % args.save_steps == 0 and \
                current_epoch != max_epochs:

            logger.info("Saving a checkpoint at iteration {0}...".format(current_epoch))

            checkpointer.save(
                name="CTPN_CHECKPOINT_EPOCH_{0}".format(current_epoch),
                data=checkpoint_data
            )

            if plotter is not None:
                plotter.save()

    logger.info("Training has just finished. Saving the final checkpoint...")
    checkpointer.save(name="CTPN_FINAL_CHECKPOINT", data=checkpoint_data)

    if use_cuda:
        # Just in case, we release all the unoccupied cached memory after training.
        torch.cuda.empty_cache()


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

    if not args.use_cuda and args.use_amp:
        raise ValueError("The arguments --use-cuda, --use-amp must be used together!")

    configs.freeze()

    run()
