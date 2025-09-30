import argparse
import copy
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from utils import *
from trainer import *
import method
import log_utils
import sys, re, shlex, subprocess

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Machine Unlearning")


    parser.add_argument('--method', type=str, default="boundary_shrink",
                        choices=['random_label', "finetune", "gradient_ascent", 
                                 'boundary_shrink', 'boundary_expand', 
                                 "salun", "l2ul_adv", "l2ul_imp", "bad_teacher",
                                 "fisher", "wood_fisher",
                                 "delete",  "scrub",
                                 "pass", "ablation"
                                 ], help='unlearning method')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', "cifar100", "tiny_imagenet", "vggface"], help='dataset name')
    parser.add_argument('--model_name', type=str, default='resnet18',  choices=['resnet18', "vgg16", "vit-s-16", "swin-t", "vit-b-16"], help='model name')
    parser.add_argument('--exps_dir', type=str, default="~/classification/exps", help='experiments directory')
    ##########################################################################################################

    parser.add_argument('--classes', type=str, default=None,
        help='Comma/space-separated class IDs to forget sequentially, e.g. "0,3,7"')
    
    parser.add_argument('--_single_run', action='store_true', default=False,
        help=argparse.SUPPRESS)  # internal guard to avoid infinite recursion

    parser.add_argument(
        '--forget_id',
        type=int,
        default=None,
        help='Exact class ID to forget (one-vs-all). If set, overrides --forget_class/permutation_map.'
    )

    ############################################# train from scratch #####################################
    parser.add_argument('--train_from_scratch', action='store_true', help='Train model from scratch')
    parser.add_argument('--retrain_from_scratch', action='store_true', help='Retrain model from scratch')
    parser.add_argument('--debug', action='store_true') 
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    ##########################################################################################################


    ##########################################################################################################
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')  
    parser.add_argument('--pretrain_epoch', type=int, default=None, help='train from scratch epoch')
    parser.add_argument('--pretrain_lr',    type=float, default=None , help='learning rate')
    parser.add_argument('--unlearn_epoch',  type=int, default=None, help='unlearning epoch')
    parser.add_argument('--unlearn_rate',   type=float, default=None) 
    parser.add_argument('--finetune_epoch', type=int, default=None)
    parser.add_argument('--finetune_lr',    type=float, default=None)
    ##########################################################################################################


    ##########################################################################################################

    parser.add_argument('--forget_class', type=int, default=1, help='forget class') 

    ##########################################################################################################

    parser.add_argument("--freeze_linear", action="store_true")  
    
    ##########################################################################################################
    parser.add_argument('--extra_exp', type=str, help='optional extra experiment for boundary shrink',
                        choices=['curv', 'weight_assign', None])
    parser.add_argument("--fixed_noise_label", type=str2bool, default=True) 
    parser.add_argument("--approx_different", type=str2bool, default=True)  
    parser.add_argument("--retain_data", type=str2bool, default=False)
    parser.add_argument("--salun_mask", type=str2bool, default=True)    
    ##########################################################################################################

    parser.add_argument("--alpha", type=float, default=0.2)  
    parser.add_argument("--threshold_ratio", type=float, default=0.5)
    parser.add_argument("--adv_lambda", type=float, default=0.1)    
    parser.add_argument("--reg_lambda", type=float, default=0)
    parser.add_argument("--adv_eps", type=float, default=0.4)
    
    ##########################################################################################################


    ##########################################################################################################
    parser.add_argument('--soft_label', type=str, default="inf")
    ##########################################################################################################

    parser.add_argument("--ablation_a", default=None, type=float)
    parser.add_argument("--ablation_t", default=None, type=float)
    parser.add_argument("--wo_dataaug", action="store_true")    
    parser.add_argument("--description", type=str, default="", help="Description for this run")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument('--load_original_model_path', type=str, default=None)
    parser.add_argument('--load_retrain_model_path', type=str, default=None)
    parser.add_argument('--use_pretrained', action='store_true', help='Load ImageNet-1k pretrained weights when available')

    args = parser.parse_args()


    def _parse_classes(s: str):
        return [int(x) for x in re.split(r'[,\s]+', s.strip()) if x != '']

    def _argv_without(opt_name: str):
        # remove --classes <value> from current argv
        out, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a == opt_name:
                skip = True
                continue
            out.append(a)
        return out

    # If user provided --classes and we're not already in a child single run,
    # spawn one child run per class with --forget_id and exit.
    if args.classes is not None and not args._single_run:
        class_ids = _parse_classes(args.classes)
        base = [sys.executable, sys.argv[0], '--_single_run'] + _argv_without('--classes')
        for c in class_ids:
            cmd = base + ['--forget_id', str(c)]
            print(">>", " ".join(shlex.quote(x) for x in cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[warn] class {c} failed with returncode={e.returncode}; continuing")
        sys.exit(0)





    config = OmegaConf.load(f'config/{args.dataset_name}_{args.model_name}.yaml')   
    keys = ["pretrain_epoch", "pretrain_lr", "batch_size", "unlearn_epoch", "unlearn_rate"]
    if args.dataset_name == "vggface":
        keys += ["finetune_epoch", "finetune_lr"]

    for key in keys:
        if getattr(args, key) is None:
            setattr(args, key, config[key])

    if any([getattr(args, key) is None for key in keys]):
        raise ValueError(f"some key are not set")

    print(args)

    model_name = args.model_name

    forget_class = args.forget_class
    num_workers = args.num_workers

    if args.forget_id is not None:
        description = f"{args.dataset_name}_{model_name}_forgetcls{args.forget_id}"
    else:
        description = f"{args.dataset_name}_{model_name}_forget{forget_class}"
        
    if args.freeze_linear:
        description = "freeze_linear_" + description


    method_description = f"{args.method}"

    vice_description = f"{args.description}" if args.description else ""    
    now = datetime.now()
    formatted_time = now.strftime("%m%d-%H:%M:%S")
    vice_description += f"_{formatted_time}"

    seed_torch(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'only support cuda'

    path = Path(args.exps_dir).expanduser()    
    create_dir(path)

    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name, wo_dataaug=args.wo_dataaug)    

    if args.dataset_name == "vggface":
        config_path = 'config/vggface_sample.yaml'
        dir_path = "/mnt/Datasets/vggface2"
        try:
            sample_config = OmegaConf.load(config_path)
        except FileNotFoundError:
            samples = create_dataset(dir_path)
            conf = OmegaConf.create(samples)
            OmegaConf.save(conf, config_path)
            sample_config = OmegaConf.load(config_path)

        pretrain_train_dataset = vggface_dataset(sample_config, transform_train, mode='pretrain', train=True)
        pretrain_test_dataset = vggface_dataset(sample_config, transform_test, mode='pretrain', train=False)
        pretrain_train_loader, pretrain_test_loader = get_dataloader(pretrain_train_dataset, pretrain_test_dataset, args.batch_size, num_workers)

    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, num_workers)


    num_classes = max(train_loader.dataset.targets) + 1
    assert forget_class < num_classes, 'forget class must less than num_classes'    

    # if args.dataset_name == "cifar10" or args.dataset_name == "vggface":
    #     forget_class_index = [forget_class]
    # else:
    #     # forget_class_index =  random.sample(range(0, num_classes), forget_class)
    #     permutation_map = getattr(config, "permutation_map")
    #     forget_class_index = permutation_map[:forget_class]
    
    if args.forget_id is not None:
        forget_class_index = [args.forget_id]      # one-vs-all
        forget_class = 1                           # keep single-class logic elsewhere working
        note_print(f"forget class index: {forget_class_index} (via --forget_id)")
        csv_forget = args.forget_id if args.forget_id is not None else forget_class
    else:
        permutation_map = getattr(config, "permutation_map")
        forget_class_index = permutation_map[:forget_class]
        note_print(f"forget class index: {forget_class_index} (via permutation_map)")    
    
    # permutation_map = getattr(config, "permutation_map")
    # forget_class_index = permutation_map[:forget_class]
    # note_print(f"forget class index: {forget_class_index}")

    num_forget = float("inf")      
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index \
        = get_unlearn_loader(trainset, testset, forget_class_index, args.batch_size, num_forget, num_workers)

    if args.train_from_scratch or args.retrain_from_scratch:
        ckpt_path = path / "test_pretrained_model"   
        create_dir(ckpt_path)
    else:
        ckpt_path = path / "test_pretrained_model"

    ori_model, retrain_model = None, None
    if args.train_from_scratch:  
        print('=' * 100)
        print(' ' * 25 + 'train original model from scratch')
        print('=' * 100)

        if args.dataset_name == "vggface":
            ori_model = train_save_model(pretrain_train_loader,
                                         pretrain_test_loader,
                                         model_name,
                                         args.optim_name,
                                         args.pretrain_lr,
                                         args.pretrain_epoch,
                                         ckpt_path,
                                         f"{args.dataset_name}_{model_name}_pretrain_model",
                                         dataset_name=args.dataset_name)
            
            ori_model.fc = torch.nn.Linear(ori_model.fc.in_features, 10)
            ori_model = ori_model.to("cuda")

            ori_model = finetune_save_model(train_loader,
                                            test_loader,
                                            ori_model,
                                            args.optim_name,
                                            args.finetune_lr,
                                            args.finetune_epoch,
                                            ckpt_path,
                                            f"{args.dataset_name}_{model_name}_original_model",
                                            model_name)
            
        else:
            ori_model = train_save_model(train_loader,
                                         test_loader,
                                         model_name,
                                         args.optim_name,
                                         args.pretrain_lr,
                                         args.pretrain_epoch,
                                         ckpt_path,
                                         f"{args.dataset_name}_{model_name}_original_model",
                                         use_pretrained=args.use_pretrained,
                                         dataset_name=args.dataset_name)
        print('\noriginal model acc:\n', test_each_classes(ori_model, test_loader, num_classes))

        csv_forget = args.forget_id if args.forget_id is not None else forget_class

        if args.forget_id is not None:
            results_csv = ckpt_path / f"{args.dataset_name}_{model_name}_original_model_metrics.csv"

        gather_and_write_metrics_csv(
            csv_path=str(results_csv),
            model=ori_model,
            method="original",
            forget_class=None,
            train_retain_loader=None,
            train_forget_loader=None,
            test_retain_loader=None,
            test_forget_loader=None,
            train_full_loader=train_loader,
            test_full_loader=test_loader,
            mia_result=None,
        )


    if args.retrain_from_scratch:
        print('=' * 100)
        print(' ' * 25 + 'retrain model from scratch')
        print('=' * 100)
        
        forget_label = (args.forget_id if args.forget_id is not None else forget_class)
        forget_tag   = f"cls{forget_label}"
        if args.forget_id is not None:
            save_desc = f"{args.dataset_name}_{model_name}_retrain_forgetcls{args.forget_id}_model"
        else:
            save_desc = f"{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model"
        
              
        
        if args.dataset_name == "vggface":
            retrain_model = train_save_model(pretrain_train_loader,
                                             pretrain_test_loader,
                                             model_name,
                                             args.optim_name,
                                             args.pretrain_lr,
                                             args.pretrain_epoch,
                                             ckpt_path,
                                             f"{args.dataset_name}_{model_name}_pretrain_model",
                                             dataset_name=args.dataset_name)
            retrain_model.fc = torch.nn.Linear(retrain_model.fc.in_features, 10)
            retrain_model = retrain_model.to("cuda")

            retrain_model = finetune_save_model(train_remain_loader,
                                                test_remain_loader,
                                                retrain_model,
                                                args.optim_name,
                                                args.finetune_lr,
                                                args.finetune_epoch,
                                                ckpt_path,
                                                save_desc,
                                                model_name)
        else:
            retrain_model = train_save_model(train_remain_loader,
                                             test_remain_loader,
                                             model_name,
                                             args.optim_name,
                                             args.pretrain_lr, 
                                             args.pretrain_epoch,
                                             ckpt_path,
                                             save_desc,
                                             use_pretrained=args.use_pretrained,
                                             dataset_name=args.dataset_name)
        print('\nretrain model acc:\n', test_each_classes(retrain_model, test_loader, num_classes))
        
        # --- write metrics row for RETRAIN (from-scratch) ---
        results_csv = ckpt_path / f"{args.dataset_name}_{model_name}_unlearned_retrained_forget{forget_class}_model_metrics.csv"
        if args.forget_id is not None:
            gather_and_write_metrics_csv(
                csv_path=str(results_csv),
                model=retrain_model,
                method="retrained",
                forget_class=forget_label,
                train_retain_loader=train_remain_loader,
                train_forget_loader=train_forget_loader,
                test_retain_loader=test_remain_loader,
                test_forget_loader=test_forget_loader,
                train_full_loader=train_loader,
                test_full_loader=test_loader,
                mia_result=None,
            )
        else:
            gather_and_write_metrics_csv(
                csv_path=str(results_csv),
                model=retrain_model,
                method="retrained",
                forget_class=forget_class,
                train_retain_loader=train_remain_loader,
                train_forget_loader=train_forget_loader,
                test_retain_loader=test_remain_loader,
                test_forget_loader=test_forget_loader,
                train_full_loader=train_loader,
                test_full_loader=test_loader,
                mia_result=None,
            )


            
        import argparse
        import gymnasium as gym
        import numpy as np
        from itertools import count
        from collections import namedtuple
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.distributions import Categorical
        
        # Cart Pole
        
        parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                            help='discount factor (default: 0.99)')
        parser.add_argument('--seed', type=int, default=543, metavar='N',
                            help='random seed (default: 543)')
        parser.add_argument('--render', action='store_true',
                            help='render the environment')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='interval between training status logs (default: 10)')
        args = parser.parse_args()
        
        
        env = gym.make('CartPole-v0')
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        
        
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        
        
        class Policy(nn.Module):
            """
            implements both actor and critic in one model
            """
            def __init__(self):
                super(Policy, self).__init__()
                self.affine1 = nn.Linear(4, 128)
        
                # actor's layer
                self.action_head = nn.Linear(128, 2)
        
                # critic's layer
                self.value_head = nn.Linear(128, 1)
        
                # action & reward buffer
                self.saved_actions = []
                self.rewards = []
        
            def forward(self, x):
                """
                forward of both actor and critic
                """
                x = F.relu(self.affine1(x))
        
                # actor: choses action to take from state s_t
                # by returning probability of each action
                action_prob = F.softmax(self.action_head(x), dim=-1)
        
                # critic: evaluates being in the state s_t
                state_values = self.value_head(x)
        
                # return values for both actor and critic as a tupel of 2 values:
                # 1. a list with the probability of each action over the action space
                # 2. the value from state s_t
                return action_prob, state_values
        
        
        model = Policy()
        optimizer = optim.Adam(model.parameters(), lr=3e-2)
        eps = np.finfo(np.float32).eps.item()
        
        
        def select_action(state):
            state = torch.from_numpy(state).float()
            probs, state_value = model(state)
        
            # create a categorical distribution over the list of probabilities of actions
            m = Categorical(probs)
        
            # and sample an action using the distribution
            action = m.sample()
        
            # save to action buffer
            model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        
            # the action to take (left or right)
            return action.item()
        
        
        def finish_episode():
            """
            Training code. Calcultes actor and critic loss and performs backprop.
            """
            R = 0
            saved_actions = model.saved_actions
            policy_losses = [] # list to save actor (policy) loss
            value_losses = [] # list to save critic (value) loss
            returns = [] # list to save the true values
        
            # calculate the true value using rewards returned from the environment
            for r in model.rewards[::-1]:
                # calculate the discounted value
                R = r + args.gamma * R
                returns.insert(0, R)
        
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
        
            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()
        
                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)
        
                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
            # reset gradients
            optimizer.zero_grad()
        
            # sum up all the values of policy_losses and value_losses
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
            # perform backprop
            loss.backward()
            optimizer.step()
        
            # reset rewards and action buffer
            del model.rewards[:]
            del model.saved_actions[:]
        
        
        def main():
            running_reward = 10
        
            # run inifinitely many episodes
            for i_episode in count(1):
        
                # reset environment and episode reward
                state = env.reset()
                ep_reward = 0
        
                # for each episode, only run 9999 steps so that we don't
                # infinite loop while learning
                for t in range(1, 10000):
        
                    # select action from policy
                    action = select_action(state)
        
                    # take the action
                    state, reward, done, _ = env.step(action)
        
                    if args.render:
                        env.render()
        
                    model.rewards.append(reward)
                    ep_reward += reward
                    if done:
                        break
        
                # update cumulative reward
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
                # perform backprop
                finish_episode()
        
                # log results
                if i_episode % args.log_interval == 0:
                    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                          i_episode, ep_reward, running_reward))
        
                # check if we have "solved" the cart pole problem
                if running_reward > env.spec.reward_threshold:
                    print("Solved! Running reward is now {} and "
                          "the last episode runs to {} time steps!".format(running_reward, t))
                    break
        
        
        if __name__ == '__main__':
            main()

      
        
    if args.train_from_scratch or args.retrain_from_scratch:
        note_print('train/retrain from scratch done')
        sys.exit(0)

    print('=' * 100)
    print(' ' * 25 + 'load original model and retrain model')
    print('=' * 100)
    # original model
    if args.load_original_model_path:
        original_model_path = Path(args.load_original_model_path)
    else:
        original_model_path = ckpt_path / f'{args.dataset_name}_{model_name}_original_model.pth'
    note_print(f"load original model from {original_model_path}")

    ori_model = load_model(original_model_path, model_name, num_classes)

    _, Aor = test(ori_model, test_remain_loader)   # (loss, acc)


    if not args.debug:
        # _, acc = test(ori_model, train_loader)
        # note_print(f"original model")
        # print(f"train acc:{acc:.2%}")
        # _, acc = test(ori_model, test_loader)
        # print(f"test acc:{acc:.2%}")

        _, acc = test(ori_model, train_forget_loader)
        print(f"forget train acc:{acc:.2%}")
        # FIXME:
        # print(f"remain train has been blocked")
        _, acc = test(ori_model, train_remain_loader)
        print(f"remain train acc:{acc:.2%}")
        _, acc = test(ori_model, test_forget_loader)
        print(f"forget test acc:{acc:.2%}")
        _, acc = test(ori_model, test_remain_loader)
        print(f"remain test acc:{acc:.2%}")
        # print('\noriginal model acc:\n', test_each_classes(ori_model, test_loader, num_classes))

    # retrain model
    if args.load_retrain_model_path:
        retrain_model_path = Path(args.load_retrain_model_path)
    else:
        forget_label = (args.forget_id if args.forget_id is not None else forget_class)
        forget_tag   = f"cls{forget_label}"
        if args.forget_id is not None:
            retrain_model_path =  ckpt_path / f"{args.dataset_name}_{model_name}_retrain_forgetcls{args.forget_id}_model.pth"
        else:
            retrain_model_path =  ckpt_path / f"{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model.pth"

    note_print(f"load retrain model from {retrain_model_path}")

    retrain_model = load_model(retrain_model_path, model_name, num_classes)

    if not args.debug:
        _, acc = test(retrain_model, train_forget_loader)
        note_print(f"\nretrain model")
        print(f"forget train acc:{acc:.2%}")

        # print(f"remain train has been blocked")
        _, acc = test(retrain_model, train_remain_loader)
        print(f"remain train acc:{acc:.2%}")
        _, acc = test(retrain_model, test_forget_loader)
        print(f"forget test acc:{acc:.2%}")
        _, acc = test(retrain_model, test_remain_loader)
        print(f"remain test acc:{acc:.2%}")
        # print('\nretrain model acc:\n', test_each_classes(retrain_model, test_loader, num_classes))


    create_dir(path / description)
    create_dir(path / description / method_description)
    #create_dir(path / description / method_description / vice_description)
    lr_dir = path / description / method_description / f"lr{args.unlearn_rate}"
    create_dir(lr_dir)

    args_dict = vars(args)
    config = OmegaConf.create(args_dict)
    #OmegaConf.save(config, path / description / method_description / vice_description / "config.yaml")
    OmegaConf.save(config, f"{path}/{description}/{method_description}/lr{args.unlearn_rate}/config.yaml")
    #logger, console_handler = log_utils.setup_logger(path / description / method_description / vice_description, logger_name="train_log")
    logger, console_handler = log_utils.setup_logger(f"{path}/{description}/{method_description}/lr{args.unlearn_rate}/", logger_name="train_log")
    log_utils.enable_console_logging(logger, console_handler, True)
    
    unlearn_model = None
    loader_dict = {"train_forget": train_forget_loader, "train_remain": train_remain_loader,
                   "test_forget": test_forget_loader, "test_remain": test_remain_loader,
                   "test": test_loader, 
                   }
    
    print('*' * 100)
    if args.method:
        note_print(' ' * 25 + f'begin {args.method.replace("_", " ")} unlearning')
    print('*' * 100)

    if args.freeze_linear:
        for name, param in ori_model.named_parameters():
            if "fc" in name:
                print(f"freeze {name}")
                param.requires_grad_(False)

    disable_bn = False
    if forget_class == 1 and args.dataset_name == "tiny_imagenet": 
        disable_bn = True
        note_print("disable bn for tiny imagenet for single class forget")

    experiment_path = path / description / method_description 
    #experiment_path = path / description / method_description / vice_description

    results_csv = experiment_path / f"lr{args.unlearn_rate}/{args.dataset_name}_{model_name}_unlearned_model_{args.method}_metrics_lr{args.unlearn_rate}.csv"

    csv_forget = args.forget_id if args.forget_id is not None else forget_class

    patience = 30

    if args.method == "random_label":
        unlearn_model = method.random_label(ori_model, train_forget_loader, num_classes,
                                                    args.unlearn_epoch, args.unlearn_rate,
                                                    results_csv=results_csv, forget_class= csv_forget,
                                                    fixed_noise_label = args.fixed_noise_label,
                                                    logger = logger, console_handler = console_handler, 
                                                    loader_dict=loader_dict, experiment_path = experiment_path, 
                                                    approx_different = args.approx_different, disable_bn = disable_bn,
                                                    early_stop_patience=patience)
    elif args.method == "finetune":
        unlearn_model = method.finetune(ori_model, train_remain_loader, 
                                                unlearn_epoch= args.unlearn_epoch, unlearn_rate= args.unlearn_rate,
                                                results_csv=results_csv, forget_class= csv_forget,
                                                logger = logger, console_handler = console_handler, 
                                                loader_dict=loader_dict, experiment_path = experiment_path,
                                                early_stop_patience=patience)
  
    elif args.method == "gradient_ascent":
        unlearn_model = method.gradient_ascent(ori_model, train_forget_loader,
                                                unlearn_epoch=args.unlearn_epoch, unlearn_rate=args.unlearn_rate,
                                                results_csv=results_csv, forget_class= csv_forget,
                                                logger=logger, console_handler=console_handler,
                                                loader_dict=loader_dict, experiment_path= experiment_path, disable_bn = disable_bn,
                                                early_stop_patience=patience)
  
    elif args.method == 'boundary_shrink':
        unlearn_model = method.boundary_shrink(ori_model, train_forget_loader,
                                               args.unlearn_epoch, args.unlearn_rate,
                                               results_csv=results_csv, forget_class= csv_forget,
                                               logger = logger, console_handler = console_handler,
                                               loader_dict = loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                               extra_exp=args.extra_exp,
                                               early_stop_patience=patience)

    elif args.method == 'boundary_expand':   
        unlearn_model = method.boundary_expand(ori_model, train_forget_loader,
                                               args.unlearn_epoch, args.unlearn_rate,
                                               num_classes,
                                               results_csv=results_csv, forget_class= csv_forget,
                                               logger = logger, console_handler = console_handler,
                                               loader_dict = loader_dict, experiment_path = experiment_path, disable_bn = disable_bn, 
                                               freeze_linear = args.freeze_linear,
                                               early_stop_patience=patience) 
    elif args.method == "salun":
        unlearn_model = method.salun(ori_model, train_forget_loader, num_classes,
                                            unlearn_epoch=args.unlearn_epoch, unlearn_rate=args.unlearn_rate,
                                            results_csv=results_csv, forget_class= csv_forget,
                                            fixed_noise_label=args.fixed_noise_label, 
                                            logger=logger, console_handler=console_handler,
                                            loader_dict=loader_dict, experiment_path= experiment_path,
                                            threshold_ratio=args.threshold_ratio,
                                            approx_different=args.approx_different,
                                            retain_data=args.retain_data,  disable_bn = disable_bn, 
                                            mask=args.salun_mask, early_stop_patience=patience)


    elif args.method == "scrub":
        unlearn_model = method.scrub(ori_model, train_forget_loader, num_classes,
                                    unlearn_epoch=args.unlearn_epoch, unlearn_rate=args.unlearn_rate,
                                    results_csv=results_csv, forget_class= csv_forget,
                                    logger=logger, console_handler=console_handler,
                                    loader_dict=loader_dict, experiment_path= experiment_path,
                                    alpha=1.0, gamma=1.0, temperature=1.0, extra_min_epochs=0,           
                                    retain_data=args.retain_data,
                                    early_stop_patience=patience,
                                    do_rewind=False,
                                    val_forget_like_loader=None)


    elif args.method == "bad_teacher":
        good_teacher_model = copy.deepcopy(ori_model).to("cuda")
        bad_teacher_model = get_model(model_name, num_classes, use_pretrained=args.use_pretrained).to("cuda")

        filtered_remain_index = random.sample(train_remain_index, int(0.3*len(train_remain_index))) if args.retain_data else []
        
        class UnLearningData(Dataset):
            def __init__(self, dataset, forget_index, remain_index):
                super().__init__()
                self.dataset = dataset
                self.index = forget_index + remain_index
                
                self.len = len(forget_index) + len(remain_index)
                self.forget_index_len = len(forget_index)

            def __len__(self):
                return self.len

            def __getitem__(self, index):
                mapped_index = self.index[index]
                x = self.dataset[mapped_index][0]
                y = 1 if index < self.forget_index_len else 0
                return x, y


        unlearn_dataset = UnLearningData(trainset, train_forget_index, filtered_remain_index)
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers) 
        unlearn_model = method.bad_teacher(ori_model, bad_teacher_model, good_teacher_model, unlearn_loader,
                                            args.unlearn_epoch, args.unlearn_rate,
                                            results_csv=results_csv, forget_class= csv_forget,
                                            logger = logger, console_handler = console_handler, 
                                            loader_dict=loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                            early_stop_patience=patience)


    elif args.method == "l2ul_adv":
        unlearn_model = method.l2ul_adv(ori_model, train_forget_loader, num_classes,
                                            args.unlearn_epoch, args.unlearn_rate,
                                            results_csv=results_csv, forget_class= csv_forget,
                                            logger = logger, console_handler = console_handler, 
                                            loader_dict=loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                            adv_eps = args.adv_eps,
                                            adv_lambda=args.adv_lambda,
                                            early_stop_patience=patience)


    elif args.method == "l2ul_imp":
        unlearn_model = method.l2ul_adv(ori_model, train_forget_loader, num_classes,
                                            args.unlearn_epoch, args.unlearn_rate,
                                            results_csv=results_csv, forget_class= csv_forget,
                                            logger = logger, console_handler = console_handler, 
                                            loader_dict=loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                            adv_eps = args.adv_eps, adv_lambda=args.adv_lambda, 
                                            reg_lambda=args.reg_lambda, early_stop_patience=patience)

        
    elif args.method == "fisher":
        unlearn_model = method.fisher(ori_model, train_forget_loader, train_remain_loader,
                                                    results_csv=results_csv, forget_class= csv_forget,
                                                    alpha=args.alpha, num_classes=num_classes,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path = experiment_path,
                                                    freeze_linear = args.freeze_linear,
                                                    early_stop_patience=patience)


    elif args.method == "wood_fisher":
        train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000
        train_remain_loader_sole = torch.utils.data.DataLoader(dataset=trainset, batch_size=1,    
                                                            sampler=train_remain_sampler,
                                                            num_workers=num_workers)

        unlearn_model = method.wood_fisher(ori_model, train_forget_loader, train_remain_loader, train_remain_loader_sole, 
                                                    results_csv=results_csv, forget_class= csv_forget,
                                                    alpha=args.alpha,
                                                    retain_data=args.retain_data,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path= experiment_path,
                                                    early_stop_patience=patience)

    elif args.method == 'delete':
        unlearn_model = method.delete(ori_model, train_forget_loader,
                                                    args.unlearn_epoch, args.unlearn_rate,
                                                    results_csv=results_csv, forget_class= csv_forget,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path= experiment_path, disable_bn = disable_bn,
                                                    soft_label=args.soft_label,
                                                    early_stop_patience=patience)


    elif args.method == 'ablation': 
        unlearn_model = method.my_method_ablation(ori_model, train_forget_loader,
                                                    args.unlearn_epoch, args.unlearn_rate,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path= experiment_path, 
                                                    soft_label=args.soft_label,
                                                    alpha = args.ablation_a,
                                                    temperature = args.ablation_t)
    elif args.method == 'pass':
        pass
    else:
        raise ValueError('method not found')    

    if unlearn_model:
        # torch.save(unlearn_model.state_dict(), path / description / vice_description / f"ckpt.pth")
        #torch.save(unlearn_model, path / description / method_description / f"ckpt.pth")


        # unlearn method
        note_print(f"\nunlearn model")
        now = time.time()
        _, test_acc         = test(unlearn_model, test_loader)
        _, forget_acc       = test(unlearn_model, test_forget_loader)
        _, remain_acc       = test(unlearn_model, test_remain_loader)
        _, train_forget_acc = test(unlearn_model, train_forget_loader)
        _, train_remain_acc = test(unlearn_model, train_remain_loader)

        logger.info('test acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}, test forget acc:{:.2%}, test remain acc:{:.2%}\n taken time {}'
             .format(test_acc, train_forget_acc, train_remain_acc, forget_acc, remain_acc, time.time()-now)) 
    
    
    
    print('\nretrain model acc:\n', test_each_classes(unlearn_model, test_loader, num_classes))
        
    import evaluation
    test_remain_len    = len(test_remain_index)

    import random
    random.shuffle(train_remain_index)

    train_remain_index = train_remain_index[:test_remain_len]
    logger.info(f"train remain size: {len(train_remain_index)}")
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  
    train_remain_loader = DataLoader(train_remain_loader.dataset, batch_size=args.batch_size, sampler=train_remain_sampler)

    mia_ori = None
    mia_retrain = None
    mia_unlearn = None

    if args.train_from_scratch or args.method == "pass":
        mia_ori = evaluation.SVC_MIA(
            shadow_train=train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=ori_model,
        )
        print(f"original model\n {mia_ori}")
    if args.retrain_from_scratch or args.method == "pass":
        mia_retrain = evaluation.SVC_MIA(
            shadow_train=train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=retrain_model,
        )
        print(f"retrain model\n {mia_retrain}")
    if unlearn_model:
        logger.info("start mia evaluation")
        mia_unlearn = evaluation.SVC_MIA(
            shadow_train=train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=unlearn_model,
        )
        logger.info(f"unlearn model\n {mia_unlearn}")
        
        
    # ---- CSV rows for loaded models & unlearned model ----
    
    csv_forget = args.forget_id if args.forget_id is not None else forget_class

    results_csv_original = ckpt_path / f"{args.dataset_name}_{model_name}_original_model_metrics.csv"

    results_csv_retrain_scratch = ckpt_path / f"{args.dataset_name}_{model_name}_unlearned_retrained_forget{forget_class}_model_metrics.csv"

    base = Path("results")  # or Path(args.exps_dir).expanduser()
    results_csv_unlearn = base / args.method / (
        f"{args.dataset_name}_{model_name}_unlearned_{args.method}_forget{forget_class}_model_metrics_lr{args.unlearn_rate}.csv"
    )
    
    # if 'ori_model' in locals() and ori_model is not None:
    #     gather_and_write_metrics_csv(
    #         csv_path=str(results_csv_original),
    #         model=ori_model,
    #         method="original",
    #         forget_class=None,
    #         train_retain_loader=None,
    #         train_forget_loader=None,
    #         test_retain_loader=None,
    #         test_forget_loader=None,
    #         train_full_loader=train_loader,
    #         test_full_loader=test_loader,
    #         mia_result=mia_ori,
    #     )

    # if 'retrain_model' in locals() and retrain_model is not None:
    #     gather_and_write_metrics_csv(
    #         csv_path=str(results_csv_retrain_scratch),
    #         model=retrain_model,
    #         method="retrained",
    #         forget_class=csv_forget,
    #         train_retain_loader=train_remain_loader,
    #         train_forget_loader=train_forget_loader,
    #         test_retain_loader=test_remain_loader,
    #         test_forget_loader=test_forget_loader,
    #         train_full_loader=train_loader,
    #         test_full_loader=test_loader,
    #         mia_result=mia_retrain,
    #     )

    if unlearn_model:
        gather_and_write_metrics_csv(
            csv_path=str(results_csv_unlearn),
            model=unlearn_model,
            method=args.method,
            forget_class=csv_forget,
            train_retain_loader=train_remain_loader,
            train_forget_loader=train_forget_loader,
            test_retain_loader=test_remain_loader,
            test_forget_loader=test_forget_loader,
            train_full_loader=train_loader,   
            test_full_loader=test_loader,
            mia_result=mia_unlearn,
        )
         
        
    logger.info("")
    exit()
