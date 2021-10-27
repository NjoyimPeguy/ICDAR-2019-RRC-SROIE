import os
import sys
import time
import torch
import argparse
import warnings

sys.path.append(os.getcwd())

from text_localization.ctpn.model import CTPN
from text_localization.ctpn.configs import configs
from text_localization.ctpn.model.loss import MultiBoxLoss
from text_localization.ctpn.utils.logger import create_logger
from text_localization.ctpn.utils.checkpointer import Checkpointer
from text_localization.ctpn.data.datasets.sroie2019 import SROIE2019Dataset
from text_localization.ctpn.data.datasets.dataloader import create_dataloader
from text_localization.ctpn.utils.misc import AverageMeter, Visualizer, get_process_time

parser = argparse.ArgumentParser(description="Connectionnist Text Proposal Network: training phase")

parser.add_argument("--config-file", action="store", help="The path to the configs file.")
parser.add_argument("--save-steps", default=10000, type=int, metavar="N",
                    help="After a certain number of iterations, a checkpoint is saved")
parser.add_argument("--plot-steps", default=1000, type=int,
                    help="After a certain number of iterations, the loss is updated and can be seen on the visualizer.")
parser.add_argument("--log-steps", default=10, type=int, help="Print logs every log steps")
parser.add_argument("--resume", action="store",
                    help='Resume training from a given checkpoint.'
                         'If None, then the training will be resumed'
                         'from the latest checkpoint.')
parser.add_argument("--use-cuda", action="store_true", help="enable/disable cuda training")
parser.add_argument("--use-amp", action="store_true",
                    help="enable/disable automatic mixed precision. By default it is disable."
                         "For further info, check those following links:"
                         "https://pytorch.org/docs/stable/amp.html"
                         "https://pytorch.org/docs/stable/notes/amp_examples.html"
                         "https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html")
parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU ID to use. By default, it is the ID 0")
parser.add_argument("--use-visdom", action="store_true",
                    help="A class to visualise data during training, i.e., in real time using Visdom"
                         "and for each iteration.")

args = parser.parse_args()

# Put warnings to silence.
warnings.filterwarnings("ignore")


def main():
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
    
    output_dir = os.path.normpath(configs.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # The declaration and initialization of the logger.
    logger = create_logger(name="CTPN: training phase", output_dir=output_dir, log_filename="training-log")
    
    # The declaration and initialization of the checkpointer.
    checkpointer = Checkpointer(output_dir, logger)
    
    # The declaration of the training dataloader arguments.
    training_dataloader_args = dict(configs.DATALOADER.ARGS)
    
    # The declaration and tensor type of the CPU/GPU device.
    device = torch.device("cpu")  # By default, the device is CPU.
    if not use_cuda:
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        
        # When one does not care about reproducibility,
        # this two lines below will find the best algorithm for your GPU.
        torch.backends.cudnn.enabled = True
    
    training_dataloader_args["generator"] = torch.Generator(device=device)
    
    plotter = None
    if args.use_visdom:
        logger.info("Declaration and initialization the visualiser 'Visdom'")
        plotter = Visualizer(port=configs.VISDOM.PORT, env_name=configs.VISDOM.ENV_NAME)
    
    # Creating the model and its loss...
    modelArgs = dict(configs.MODEL.ARGS)
    model = CTPN(**modelArgs).to(device)
    
    criterion = MultiBoxLoss(configs, neg_pos_ratio=3, lambda_reg=4.0)
    
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=configs.SOLVER.LR,
                                  betas=configs.SOLVER.BETAS,
                                  weight_decay=configs.SOLVER.WEIGHT_DECAY,
                                  eps=configs.SOLVER.EPS,
                                  amsgrad=configs.SOLVER.AMSGRAD)
    
    loc_loss_plotter_window = conf_loss_plotter_window = total_loss_plotter_window = None
    
    logger.info("Declaration and initialization of the learning rate scheduler...")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=configs.SOLVER.LR_DECAY_STEPS,
                                                        gamma=configs.SOLVER.GAMMA)
    
    start_iteration = 1
    max_iterations = configs.SOLVER.MAX_ITERATIONS
    
    checkpoint_data = {}
    
    if args.resume is not None:
        
        logger.info("Resuming the training...")
        
        checkpoint_data = checkpointer.load(file_to_load=args.resume, map_location=torch.device(device))
        
        start_iteration = checkpoint_data.get("iteration", 1)
        
        loc_loss_plotter_window = checkpoint_data.get("loc_loss_plot_win")
        conf_loss_plotter_window = checkpoint_data.get("conf_loss_plot_win")
        total_loss_plotter_window = checkpoint_data.get("total_loss_plot_win")
        
        # Loading the generator model amp state
        amp_state = checkpoint_data.get("amp_state")
        if use_amp and amp_state is not None:
            grad_scaler.load_state_dict(amp_state)
        
        # Loading the optimizer state dict...
        optimizer_state_dict = checkpoint_data.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        
        # Loading the learning rate scheduler state dict..
        lr_scheduler_state_dict = checkpoint_data.get("lr_scheduler_state_dict")
        if lr_scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        
        # Loading the model state dict...
        model_state_dict = checkpoint_data.get("model_state_dict")
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
    
    # Creating a plotter window in visdom environment.
    if plotter is not None and \
            loc_loss_plotter_window is None and \
            conf_loss_plotter_window is None and \
            total_loss_plotter_window is None:
        title_name = "Training: CTPN"
        legend_names = ["Training loss"]
        
        logger.info("Plot creations...")
        loc_loss_plotter_window = plotter.createPlot(xLabel="Iterations", yLabel="Localisation Loss",
                                                     legend_names=legend_names, title_name=title_name)
        conf_loss_plotter_window = plotter.createPlot(xLabel="Iterations", yLabel="Confidence Loss",
                                                      legend_names=legend_names, title_name=title_name)
        total_loss_plotter_window = plotter.createPlot(xLabel="Iterations", yLabel="Loss",
                                                       legend_names=legend_names, title_name=title_name)
    
    logger.info("Loading the dataset...")
    
    train_dataset = SROIE2019Dataset(configs)
    
    train_loader = create_dataloader(dataset=train_dataset,
                                     is_train=True,
                                     start_iteration=start_iteration,
                                     max_iterations=max_iterations,
                                     **training_dataloader_args)
    
    logger.info("About to start the training...")
    
    training_start_time = time.time()
    
    accumulated_iteration = 0
    
    # Setting the model in training mode...
    model = model.train()
    
    regression_losses = AverageMeter(fmt=":.6f")
    classification_losses = AverageMeter(fmt=":.6f")
    total_losses = AverageMeter(fmt=":.6f")
    
    # Before starting the training, all the unoccupied cached memory are released...
    if use_cuda:
        torch.cuda.empty_cache()
    
    for current_iteration, batch_samples in enumerate(train_loader, start=start_iteration):
        
        images, targets = batch_samples
        
        # Putting into the right device...
        images = images.to(device)
        
        # targets:0 = boxes, targets:1 = labels
        ground_truths = (targets[0].to(device), targets[1].to(device))
        
        # Clearing out the gradient before doing backprop.
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            
            predictions = model(images)
            
            regr_loss, cls_loss = criterion(predictions, ground_truths)
            
            loss = regr_loss + cls_loss
        
        grad_scaler.scale(loss).backward()
        
        grad_scaler.step(optimizer)
        
        grad_scaler.update()
        
        lr_scheduler.step(current_iteration)
        checkpoint_data.update({"lr_scheduler_state_dict": lr_scheduler.state_dict()})
        
        # Calling ".item()" operation requires synchronization. For further info, check this out:
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html And this as well:
        # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
        # Just in case of in-place operations, one can add .clone() to avoid nasty modifications...
        regression_losses.update(regr_loss.detach().clone())
        classification_losses.update(cls_loss.detach().clone())
        total_losses.update(loss.detach().clone())
        
        # Updating important data
        checkpoint_data.update({
            "iteration": current_iteration + 1,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimizer.state_dict()
        })
        
        # Updating Visdom plots if any
        if plotter is not None and current_iteration > 0 and \
                args.plot_steps > 0 and \
                current_iteration % args.plot_steps == 0:
            loc_loss_data_y = [regression_losses.global_avg]
            conf_loss_data_y = [classification_losses.global_avg]
            total_loss_data_y = [total_losses.global_avg]
            plotter.update_plot(window=loc_loss_plotter_window, data_x=current_iteration,
                                data_y=loc_loss_data_y)
            plotter.update_plot(window=conf_loss_plotter_window, data_x=current_iteration,
                                data_y=conf_loss_data_y)
            plotter.update_plot(window=total_loss_plotter_window, data_x=current_iteration,
                                data_y=total_loss_data_y)
            
            # Updating plots...
            checkpoint_data.update({"loc_loss_plot_win": loc_loss_plotter_window})
            checkpoint_data.update({"conf_loss_plot_win": conf_loss_plotter_window})
            checkpoint_data.update({"total_loss_plot_win": total_loss_plotter_window})
        
        accumulated_iteration += 1
        elapsed_time, left_time, estimated_finish_time = get_process_time(start_time=training_start_time,
                                                                          current_iteration=accumulated_iteration,
                                                                          max_iterations=max_iterations)
        
        if current_iteration > 0 and args.log_steps > 0 and current_iteration % args.log_steps == 0:
            logger.info("Iter: {curr_iteration}/{maximum_iterations} || "
                        "Learning_rate: {lr:.8f}\n"
                        "Localisation loss: {loss_l} || "
                        "Confidence loss: {loss_c} || "
                        "Total loss: {total_loss}\n"
                        "Elapsed time: {et} seconds || "
                        "Remaining time: {lt} seconds || "
                        "ETA: {eta}\n\n".format(
                curr_iteration=current_iteration,
                maximum_iterations=max_iterations,
                lr=optimizer.param_groups[0]["lr"],
                loss_l=regression_losses,
                loss_c=classification_losses,
                total_loss=total_losses,
                et=elapsed_time,
                lt=left_time,
                eta=estimated_finish_time
            ))
        
        # Saving useful data...
        if current_iteration > 0 and args.save_steps > 0 and current_iteration % args.save_steps == 0:
            logger.info("Saving a checkpoint at iteration {0}...".format(current_iteration))
            
            if use_amp:
                checkpoint_data.update({"amp_state": grad_scaler.state_dict()})
            
            nb_iters = "{0}".format(
                str((current_iteration // 1000)) + "k" if current_iteration % 1000 == 0 else current_iteration
            )
            
            checkpointer.save(
                name="CHECKPOINT{0}".format(nb_iters),
                data=checkpoint_data
            )
            
            if plotter is not None:
                plotter.save()
    
    logger.info("Training has just finished. Saving the final checkpoint...")
    checkpointer.save(name="CTPN_FINAL_CHECKPOINT", data=checkpoint_data)
    
    # Just in case, we release all the unoccupied cached memory after training...
    if use_cuda:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    
    # Guarding against bad arguments.
    
    if args.save_steps < 0:
        raise ValueError("{0} is an invalid value for the argument: --save-steps".format(args.save_steps))
    elif args.log_steps < 0:
        raise ValueError("{0} is an invalid value for the argument: --log-steps".format(args.log_steps))
    elif args.resume is not None and not os.path.isfile(args.resume):
        raise ValueError("The path to the checkpoint data file is wrong!")
    
    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! You may want to check with 'nvidia-smi'")
    elif args.use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    elif args.use_amp:
        raise ValueError("The arguments --use-cuda, --use-amp and --gpu-device must be used together!")
    
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()
    
    main()
