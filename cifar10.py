from advertorch.attacks import L2PGDAttack
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import robustbench
from robustness.model_utils import make_and_restore_model
from robustbench.data import PREPROCESSINGS, _load_dataset
from torchvision.datasets import CIFAR10
from robustness.datasets import CIFAR


parser = argparse.ArgumentParser()
parser.add_argument("--simplex-dim", type=int, default=1)
parser.add_argument("--model-folder", type=str, required=True)
parser.add_argument("--data-folder", type=str, default=None)
parser.add_argument("--source-model", action='store_true')
parser.add_argument("--force", action='store_true')
parser.add_argument("--eps-iter", type=float, default=0.15)
parser.add_argument("--eps", type=float, default=0.5)
parser.add_argument("--eps-iter-very", type=float, default=0.001)
parser.add_argument("--nb-iter", type=int, default=20)
parser.add_argument("--nb-iter-very", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=64)

args = parser.parse_args()
if args.nb_iter_very is None:
    args.nb_iter_very = 1000*args.simplex_dim
if args.data_folder is None:
    args.data_folder = args.model_folder


SOURCE_PATH = os.path.join(args.model_folder, "robustbench")
TARGET_PATH = os.path.join(
    args.model_folder, "cifar10/resnet18/62d52c66-93d7-4508-bdc1-1c16fe26e500/checkpoint.pt.best")
DATA_PATH = os.path.join(args.data_folder, "cifar-10-batches-py")

adv_path = os.path.join(SOURCE_PATH, "x_adv_"+str(args.nb_iter)+".pt")
very_adv_path = os.path.join(
    SOURCE_PATH, "x_very_adv_"+str(args.simplex_dim)+".pt")
very_adv_path_target = os.path.join(
    SOURCE_PATH, "y_very_adv_"+str(args.simplex_dim)+".pt")


class KLDivLossOnLogits(nn.KLDivLoss):
    def forward(self, input, target):
        output = F.log_softmax(input, dim=-1)
        return super(KLDivLossOnLogits, self).forward(output, target)


def sample_simplex(dim, num_classes, size):
    assert dim <= num_classes
    unif = torch.rand((size, dim)).cuda()
    exps = torch.log(unif)
    norms = torch.sum(exps, dim=-1, keepdim=True)
    simplex = exps/norms
    final = torch.zeros((size, num_classes),
                        dtype=simplex.dtype, device=simplex.device)
    class_choices = torch.stack(
        [torch.randperm(num_classes)[:dim]for _ in range(size)], dim=0)
    for b in range(size):
        final[b, class_choices[b]] = simplex[b]

    return final


def load_model(source_model):
    if source_model:
        model = robustbench.utils.load_model(model_name="Standard", dataset="cifar10", threat_model="L2",
                                             model_dir=SOURCE_PATH)
    else:
        model, _ = make_and_restore_model(
            arch="resnet18", dataset=CIFAR(DATA_PATH), resume_path=TARGET_PATH)
        model = model.model
    model = model.cuda()
    return model


data = CIFAR10(root=DATA_PATH, train=False, download=False,
               transform=PREPROCESSINGS[None])
model = load_model(source_model=args.source_model)
x_test, y_test = _load_dataset(data, args.batch_size)
x_test, y_test = x_test.cuda(), y_test.cuda()
attacker = L2PGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps,
    nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=False, clip_min=-1.0, clip_max=1.0,
    targeted=True)
very_attacker = L2PGDAttack(
    model, loss_fn=KLDivLossOnLogits(reduction="sum"), eps=args.eps,
    nb_iter=args.nb_iter_very, eps_iter=args.eps_iter_very, rand_init=False, clip_min=-1.0, clip_max=1.0,
    targeted=True)

acc = (model(x_test).argmax(dim=-1) == y_test).detach().cpu().numpy().mean()
print("Natural accuracy:", acc)

print("Running standard targeted attack")
y_target = (y_test + 1) % 10
try:
    assert not args.force
    x_adv = torch.load(adv_path).cuda()
except:
    x_adv = attacker.perturb(x_test, y_target)
adv_acc_tgt = (model(x_adv).argmax(dim=-1) ==
               y_target).detach().cpu().numpy().mean()
print("Adversarial accuracy on target:", adv_acc_tgt)
torch.save(x_adv, adv_path)
adv_acc = (model(x_adv).argmax(dim=-1) == y_test).detach().cpu().numpy().mean()
print("Adversarial accuracy on label:", adv_acc)

print("Running very targeted attack")

try:
    assert not args.force
    x_very_adv = torch.load(very_adv_path).cuda()
    y_very_target = torch.load(very_adv_path_target).cuda()

except:
    y_very_target = sample_simplex(args.simplex_dim, 10, args.batch_size)
    x_very_adv = very_attacker.perturb(x_test, y_very_target)

very_acc = (torch.argsort(model(x_very_adv), dim=-1)[:, 10-args.simplex_dim:] ==
            torch.argsort(y_very_target, dim=-1)[:, 10-args.simplex_dim:]).detach().cpu().numpy().mean()
torch.save(x_very_adv, very_adv_path)
torch.save(y_very_target, very_adv_path_target)
nat_very_acc = (torch.argsort(model(x_test), dim=-1)[:, 10-args.simplex_dim:] ==
                torch.argsort(y_very_target, dim=-1)[:, 10-args.simplex_dim:]).detach().cpu().numpy().mean()
print("Adversarial accuracy on sorted classes of target:", very_acc)
print("Natural accuracy on sorted classes of target:", nat_very_acc)
