import os
import sys
import time
import torch
import argparse
import warnings
import torch.nn.functional as F

sys.path.append(os.getcwd())

# importing useful function/methods from task 1
from text_localization.ctpn.utils.logger import create_logger
from text_localization.ctpn.utils.checkpointer import Checkpointer
from text_localization.ctpn.data.datasets.dataloader import create_dataloader
from text_localization.ctpn.utils.misc import AverageMeter, Visualizer, get_process_time

from keyword_information_extraction.configs import configs
from keyword_information_extraction.model.charlm import CharacterLevelCNNHighwayBiLSTM
from keyword_information_extraction.data.datasets.sroie2019 import SROIE2019Dataset, TrainBatchCollator
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser(description="Character-Level for Text Classification")

parser.add_argument("--config-file", action="store", help="The path to the configs file.")
parser.add_argument("--save-steps", default=0, type=int, metavar="N",
                    help="After a certain number of epochs, a checkpoint is saved")
parser.add_argument("--log-steps", default=10, type=int, help="Print logs every log steps")
parser.add_argument("--resume", action="store",
                    help="Resume training from a path to the given checkpoint.")
parser.add_argument("--use-cuda", action="store_true",
                    help="Enable or disable the CUDA training. By default it is disable.")
parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU ID to use. By default, it is the ID 0")
parser.add_argument("--use-visdom", action="store_true",
                    help="Enable or disable visualization during training with Visdom which is "
                         "a class to visualise data in real time and for each iteration. By default it is disable.")

args = parser.parse_args()

# Put warnings to silence if any.
warnings.filterwarnings("ignore")


def main():
    # A boolean to check whether the user is able to use cuda or not.
    use_cuda = torch.cuda.is_available() and args.use_cuda
    
    # One can comment the line below.
    # It is important to note it is useful when some nasty errors like the NaN loss show up.
    torch.autograd.set_detect_anomaly(True)
    
    output_dir = os.path.normpath(configs.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # The declaration and initialization of the logger.
    logger = create_logger(name="Training phase", output_dir=output_dir, log_filename="training-log")
    
    # The declaration and initialization of the checkpointer.
    checkpointer = Checkpointer(output_dir, logger)
    
    # The declaration of the dataloader arguments.
    dataloader_args = dict(configs.DATALOADER.TRAINING)
    
    # Adding the collate_fn.
    dataloader_args["collate_fn"] = TrainBatchCollator(class_labels_padding_value=configs.MODEL.LOSS.IGNORE_INDEX)
    
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
    
    dataloader_args["generator"] = torch.Generator(device=device)
    
    plotter = None
    if args.use_visdom:
        logger.info("Initialising the visualiser 'Visdom'")
        plotter = Visualizer(port=configs.VISDOM.PORT, env_name=configs.VISDOM.ENV_NAME)
    
    logger.info("Performing the train-validation dataset split...")
    class_dict = dict(configs.DATASET.CLASS_NAMES)
    train_dataset = SROIE2019Dataset(new_dir=dict(configs.DATASET.TRAIN), class_dict=class_dict,
                                     validation_args={"val_folder": dict(configs.DATASET.VAL), "ratio": 0.1})
    train_loader = create_dataloader(dataset=train_dataset, is_train=True, **dataloader_args)
    train_criterion = torch.nn.CrossEntropyLoss(ignore_index=configs.MODEL.LOSS.IGNORE_INDEX, reduction="none")
    vocabulary = train_dataset.vocabulary
    text_max_length = train_dataset.text_max_length
    
    logger.info("Declaration and initialization of the validation dataset")
    validation_dataset = SROIE2019Dataset(new_dir=dict(configs.DATASET.VAL), class_dict=class_dict)
    validation_loader = create_dataloader(dataset=validation_dataset, is_train=False, **dataloader_args)
    validation_criterion = torch.nn.CrossEntropyLoss(ignore_index=configs.MODEL.LOSS.IGNORE_INDEX, reduction="mean")
    
    logger.info("Declaration and initialization of the model and optimizer...")
    model_params = dict(configs.MODEL.PARAMS)
    vocab_size = len(vocabulary)
    model = CharacterLevelCNNHighwayBiLSTM(n_classes=configs.DATASET.NUM_CLASSES,
                                           max_seq_length=text_max_length,
                                           char_vocab_size=vocab_size, **model_params)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=configs.SOLVER.LR,
                                  weight_decay=configs.SOLVER.WEIGHT_DECAY,
                                  amsgrad=configs.SOLVER.AMSGRAD)
    
    logger.info("Declaration and initialization of the learning rate scheduler...")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=configs.SOLVER.LR_DECAY_STEPS,
                                                        gamma=configs.SOLVER.GAMMA)
    
    start_epoch = 1
    checkpoint_data = {}
    best_validation_f1_score = total_loss_plotter_window = None
    
    if args.resume is not None:
        
        logger.info("Resuming the training...")
        
        checkpoint_data = checkpointer.load(file_to_load=args.resume, map_location=device)
        
        start_epoch = checkpoint_data["epoch"]
        
        total_loss_plotter_window = checkpoint_data.get("total_loss_plot_win")
        best_validation_f1_score = checkpoint_data.get("best_validation_f1_score")
        
        optimizer_state_dict = checkpoint_data.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        
        model_state_dict = checkpoint_data.get("model_state_dict")
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
    
    # Creating a plotter window in visdom environment.
    if plotter is not None and total_loss_plotter_window is None:
        logger.info("Plot creations...")
        title_name = "Training: Character Level CNN + Highway + BiLSTM"
        legend_names = ["Training loss"]
        if validation_loader is not None:
            legend_names.append("Validation loss")
        total_loss_plotter_window = plotter.createPlot(xLabel="Epochs", yLabel="Loss",
                                                       legend_names=legend_names, title_name=title_name)
    
    logger.info("About to start the training...")
    
    if best_validation_f1_score is None:
        best_validation_f1_score = 0.0
    
    accumulated_epoch = 0
    
    training_start_time = time.time()
    
    for current_epoch in range(start_epoch, configs.SOLVER.MAX_EPOCHS + 1):
        
        training_loss = train(model=model, optimizer=optimizer, train_loader=train_loader,
                              criterion=train_criterion, device=device, current_epoch=current_epoch,
                              logger=logger)
        
        logger.info("[TRAINING] Total loss: {total_loss:.6f}\n\n".format(total_loss=training_loss))
        
        logger.info("Validation ongoing...")
        if use_cuda:
            torch.cuda.empty_cache()  # speed up evaluation after k-epoch training has just finished.
        validation_f1_score, validation_loss = validate(model=model, validation_loader=validation_loader,
                                                        criterion=validation_criterion, device=device,
                                                        current_epoch=current_epoch, logger=logger)
        # One can save the best models based on the highest F1-score.
        if validation_f1_score > best_validation_f1_score:
            best_validation_f1_score = validation_f1_score
            checkpoint_data.update({"best_validation_f1_score": best_validation_f1_score})
            checkpointer.save(name="BEST_MODEL_BASED_ON_F1_SCORE", is_best=True, data=checkpoint_data)
        
        # Saving the learning rate scheduler.
        lr_scheduler.step(current_epoch)
        checkpoint_data.update({"lr_scheduler_state_dict": lr_scheduler.state_dict()})
        
        # Updating the plot if it was previously set (i.e., the plotter is not None)
        if plotter is not None:
            total_loss_data_y = [training_loss]
            if validation_loss is not None:
                total_loss_data_y.append(validation_loss)
            plotter.update_plot(window=total_loss_plotter_window, data_x=current_epoch, data_y=total_loss_data_y)
            checkpoint_data.update({"total_loss_plot_win": total_loss_plotter_window})
        
        # Saving useful data.
        checkpoint_data.update({
            "epoch": current_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        })
        
        if current_epoch > 0 and args.save_steps > 0 and current_epoch % args.save_steps == 0:
            
            logger.info("Saving a checkpoint at epoch {0}...".format(current_epoch))
            
            checkpointer.save(name="{0}_{1}_CHECKPOINT_EPOCH_{2}".format(
                configs.MODEL.NAME.upper(),
                configs.DATASET.NAME.upper(),
                current_epoch),
                data=checkpoint_data
            )
            
            # Saving plots...
            if plotter is not None:
                plotter.save()
        
        # Computing time...
        accumulated_epoch += 1
        elapsed_time, left_time, estimated_finish_time = get_process_time(start_time=training_start_time,
                                                                          current_iteration=accumulated_epoch,
                                                                          max_iterations=configs.SOLVER.MAX_EPOCHS)
        logger.info("Elapsed time: {et} seconds || "
                    "Remaining time: {lt} seconds || "
                    "ETA: {eta}\n\n".format(
            et=elapsed_time,
            lt=left_time,
            eta=estimated_finish_time
        ))
    
    logger.info("Training has just finished. Saving the final checkpoint...")
    checkpointer.save(name="{0}_FINAL_CHECKPOINT".format(
        configs.MODEL.NAME.upper()),
        data=checkpoint_data
    )
    
    if use_cuda:
        torch.cuda.empty_cache()


def train(model: torch.nn.Module, optimizer, train_loader,
          criterion: torch.nn.Module, device: torch.device,
          current_epoch: int, logger):
    training_losses = AverageMeter(fmt=":.6f")
    
    # Activating Dropout layer if any...
    model = model.train()
    
    for current_iteration, batch_samples in enumerate(train_loader):
        
        text_features, text_class_labels, class_weights = batch_samples
        
        text_features = text_features.to(device)
        text_class_labels = text_class_labels.to(device)
        
        # Clearing out the model's gradients before doing backprop.
        # 'set_to_none=True' here can modestly improve performance
        optimizer.zero_grad(set_to_none=True)
        
        predictions = model(text_features)
        
        class_weights = class_weights.contiguous().view(-1)
        
        text_class_labels = text_class_labels.contiguous().view(-1)
        
        predictions = predictions.contiguous().view(-1, configs.DATASET.NUM_CLASSES)
        
        loss = (class_weights * criterion(predictions, text_class_labels)).mean()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5.0, norm_type=2.0)
        
        optimizer.step()
        
        # Calling ".item()" operation requires synchronization. For further info, check this out:
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html And this as well:
        # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
        # Just in case of in-place operations, one can add .clone() to avoid nasty modifications...
        training_losses.update(loss.detach().clone(), n=text_features.size(0))
        
        if current_iteration > 0 and args.log_steps > 0 and current_iteration % args.log_steps == 0:
            logger.info("Epoch: {curr_epoch}/{maximum_epochs} || "
                        "Iterations: {iter} || "
                        "Learning_rate: {lr} || "
                        "Loss: {loss}\n\n".format(
                curr_epoch=current_epoch,
                maximum_epochs=configs.SOLVER.MAX_EPOCHS,
                iter=current_iteration,
                lr=optimizer.param_groups[0]["lr"],
                loss=training_losses
            ))
    return training_losses.global_avg


@torch.no_grad()
def validate(model: torch.nn.Module, validation_loader,
             criterion: torch.nn.Module, device: torch.device,
             current_epoch: int, logger):
    validation_losses = AverageMeter(fmt=":.6f")
    
    # Disabling Dropout layer if any...
    model = model.eval()
    
    true_classes = []
    predicted_classes = []
    
    for current_iteration, batch_samples in enumerate(validation_loader):
        
        text_features, text_class_labels, class_weights = batch_samples
        
        # Putting into the right device...
        text_features = text_features.to(device)
        text_class_labels = text_class_labels.to(device)
        
        text_class_labels = text_class_labels.view(-1)
        
        outputs = model(text_features)
        
        loss = criterion(outputs.view(-1, configs.DATASET.NUM_CLASSES), text_class_labels)
        
        validation_losses.update(loss.detach().cpu().numpy(), text_features.size(0))
        idx = torch.where(text_class_labels == configs.MODEL.LOSS.IGNORE_INDEX)[0]
        text_class_labels[idx] = 0
        true_classes.extend(text_class_labels.detach().cpu().numpy())
        
        _, predictions = torch.max(F.softmax(outputs, dim=2), dim=2)
        predictions = predictions.contiguous().view(-1)
        predicted_classes.extend(predictions.detach().cpu().numpy())
        
        if current_iteration > 0 and args.log_steps > 0 and current_iteration % args.log_steps == 0:
            logger.info("Epoch: {curr_epoch}/{maximum_epochs} || "
                        "Iteration: {iter} || "
                        "Loss: {loss}\n\n".format(
                curr_epoch=current_epoch,
                maximum_epochs=configs.SOLVER.MAX_EPOCHS,
                iter=current_iteration,
                loss=validation_losses
            ))
    
    validation_loss = validation_losses.global_avg
    validation_f1_score = f1_score(true_classes, predicted_classes, average="weighted")
    results = {
        "loss": validation_loss,
        "precision": precision_score(true_classes, predicted_classes, average="weighted"),
        "recall": recall_score(true_classes, predicted_classes, average="weighted"),
        "f1": validation_f1_score
    }
    
    report = classification_report(true_classes, predicted_classes)
    logger.info("\n" + report)
    
    logger.info("***** Validation results ******")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return validation_f1_score, validation_loss


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
    elif args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()
    
    main()
