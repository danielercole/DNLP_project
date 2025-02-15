import torch 
from torch import nn
from typing import List, Optional, Tuple, Union

from cyclegan_tst.models.GeneratorModelDO import GeneratorModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.ClassifierModel import ClassifierModel

import logging


class CycleGANModel(nn.Module):

    def __init__(
        self,
        G_ab: Union[GeneratorModel, None], 
        G_ba: Union[GeneratorModel, None], 
        D_ab: Union[DiscriminatorModel, None],
        D_ba: Union[DiscriminatorModel, None],
        Cls: Union[ClassifierModel, None],
        device = None,
    ):
        """Initialization method for the CycleGANModel

        Args:
            G_ab: Generator model for mapping from A -> B.
            G_ba: Generator model for mapping from B -> A.
            D_ab: Discriminator model for domain B.
            D_ba: Discriminator model for domain A.
            Cls: Style classifier.
        """
        super(CycleGANModel, self).__init__()
        
        if G_ab is None or G_ba is None or D_ab is None or D_ba is None:
            logging.warning("CycleGANModel: Some models are not provided, please invoke 'load_models' to initialize them from a previous checkpoint")

        self.G_ab = G_ab
        self.G_ba = G_ba
        self.D_ab = D_ab
        self.D_ba = D_ba
        self.Cls = Cls

        self.device = device
        logging.info(f"Device: {device}")

        # Move all models to the specified device
        self.G_ab.model.to(self.device)
        self.G_ba.model.to(self.device)
        self.D_ab.model.to(self.device)
        self.D_ba.model.to(self.device)
        if self.Cls is not None:
            self.Cls.model.to(self.device)

    def train(self):
        self.G_ab.train()
        self.G_ba.train()
        self.D_ab.train()
        self.D_ba.train()

    def eval(self):
        self.G_ab.eval()
        self.G_ba.eval()
        self.D_ab.eval()
        self.D_ba.eval()

    def get_optimizer_parameters(self):
        optimization_parameters = list(self.G_ab.model.parameters())
        optimization_parameters += list(self.G_ba.model.parameters())
        optimization_parameters += list(self.D_ab.model.parameters())
        optimization_parameters += list(self.D_ba.model.parameters())
        return optimization_parameters

    def training_cycle(
        self, 
        sentences_a: List[str],
        sentences_b: List[str],
        target_sentences_ab: List[str] = None,
        target_sentences_ba: List[str] = None,
        lambdas: List[float] = None,
        comet_experiment = None,
        loss_logging = None,
        training_step: int = None
    ):
        # ---------- BEGIN: cycle A -> B ----------

        # First half: Generate style-transferred sentences from A -> B.
        # (In inference mode, the new generator builds the prompt internally.)
        out_transferred_ab, transferred_ab = self.G_ab(sentences_a, device=self.device) 
        
        # Evaluate D_ab on the fake (transferred) sentences.
        self.D_ab.eval()  # Loss computed for the generator
        zeros = torch.zeros(len(transferred_ab))
        ones  = torch.ones(len(transferred_ab))
        labels_fake_sentences = torch.column_stack((ones, zeros))  # Label for fake sentences
        _, loss_g_ab = self.D_ab(transferred_ab, labels_fake_sentences, device=self.device)
        
        if lambdas[4] != 0:
            labels_style_b_sentences = torch.ones(len(transferred_ab), dtype=int)
            _, loss_g_ab_cls = self.Cls(transferred_ab, labels_style_b_sentences, device=self.device)
        
        # Second half: Reconstruct A from the transferred B and compute cycle loss.
        out_reconstructed_ba, reconstructed_ba, cycle_loss_aba = self.G_ba(transferred_ab, sentences_a, device=self.device)

        complete_loss_g_ab = lambdas[0] * cycle_loss_aba + lambdas[1] * loss_g_ab
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Cycle Loss A-B-A", lambdas[0] * cycle_loss_aba, step=training_step)
                comet_experiment.log_metric("Loss generator  A-B", lambdas[1] * loss_g_ab, step=training_step)
        loss_logging["Cycle Loss A-B-A"].append(lambdas[0] * cycle_loss_aba.item())
        loss_logging["Loss generator  A-B"].append(lambdas[1] * loss_g_ab.item())

        if lambdas[4] != 0:
            complete_loss_g_ab += lambdas[4] * loss_g_ab_cls
            if comet_experiment is not None:
                with comet_experiment.train():
                    comet_experiment.log_metric("Classifier-guided A-B", lambdas[4] * loss_g_ab_cls, step=training_step)
            loss_logging["Classifier-guided A-B"].append(lambdas[4] * loss_g_ab_cls.item())
        
        complete_loss_g_ab.backward()
        
        # Update Discriminator D_ab.
        self.D_ab.train()
        zeros = torch.zeros(len(transferred_ab))
        ones  = torch.ones(len(transferred_ab))
        labels_fake_sentences = torch.column_stack((zeros, ones))
        _, loss_d_ab_fake = self.D_ab(transferred_ab, labels_fake_sentences, device=self.device)
        zeros = torch.zeros(len(transferred_ab))
        ones  = torch.ones(len(transferred_ab))
        labels_real_sentences = torch.column_stack((ones, zeros))
        _, loss_d_ab_real = self.D_ab(sentences_b, labels_real_sentences, device=self.device)
        complete_loss_d_ab = lambdas[2] * loss_d_ab_fake + lambdas[3] * loss_d_ab_real

        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Loss D(A->B)", complete_loss_d_ab, step=training_step)
        loss_logging["Loss D(A->B)"].append(complete_loss_d_ab.item())
        complete_loss_d_ab.backward()
        # ---------- END: cycle A -> B ----------

        # ---------- BEGIN: cycle B -> A ----------
        # First half: Generate style-transferred sentences from B -> A.
        out_transferred_ba, transferred_ba = self.G_ba(sentences_b, device=self.device)

        # Evaluate D_ba on the fake (transferred) sentences.
        self.D_ba.eval()
        zeros = torch.zeros(len(transferred_ba))
        ones  = torch.ones(len(transferred_ba))
        labels_fake_sentences = torch.column_stack((ones, zeros))
        _, loss_g_ba = self.D_ba(transferred_ba, labels_fake_sentences, device=self.device)
        
        if lambdas[4] != 0:
            labels_style_a_sentences = torch.zeros(len(transferred_ba), dtype=int)
            _, loss_g_ba_cls = self.Cls(transferred_ba, labels_style_a_sentences, device=self.device)
        
        # Second half: Reconstruct B from the transferred A and compute cycle loss.
        out_reconstructed_ab, reconstructed_ab, cycle_loss_bab = self.G_ab(transferred_ba, sentences_b, device=self.device)
        
        complete_loss_g_ba = lambdas[0] * cycle_loss_bab + lambdas[1] * loss_g_ba
        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Cycle Loss B-A-B", lambdas[0] * cycle_loss_bab, step=training_step)
                comet_experiment.log_metric("Loss generator  B-A", lambdas[1] * loss_g_ba, step=training_step)
        loss_logging["Cycle Loss B-A-B"].append(lambdas[0] * cycle_loss_bab.item())
        loss_logging["Loss generator  B-A"].append(lambdas[1] * loss_g_ba.item())

        if lambdas[4] != 0:
            complete_loss_g_ba += lambdas[4] * loss_g_ba_cls
            if comet_experiment is not None:
                with comet_experiment.train():
                    comet_experiment.log_metric("Classifier-guided B-A", lambdas[4] * loss_g_ba_cls, step=training_step)
            loss_logging["Classifier-guided B-A"].append(lambdas[4] * loss_g_ba_cls.item())
        
        complete_loss_g_ba.backward()

        # Update Discriminator D_ba.
        self.D_ba.train()
        zeros = torch.zeros(len(transferred_ba))
        ones  = torch.ones(len(transferred_ba))
        labels_fake_sentences = torch.column_stack((zeros, ones))
        _, loss_d_ba_fake = self.D_ba(transferred_ba, labels_fake_sentences, device=self.device)
        zeros = torch.zeros(len(transferred_ba))
        ones  = torch.ones(len(transferred_ba))
        labels_real_sentences = torch.column_stack((ones, zeros))
        _, loss_d_ba_real = self.D_ba(sentences_a, labels_real_sentences, device=self.device)
        complete_loss_d_ba = lambdas[2] * loss_d_ba_fake + lambdas[3] * loss_d_ba_real

        if comet_experiment is not None:
            with comet_experiment.train():
                comet_experiment.log_metric("Loss D(B->A)", complete_loss_d_ba, step=training_step)
        loss_logging["Loss D(B->A)"].append(complete_loss_d_ba.item())
        complete_loss_d_ba.backward()
        # ---------- END: cycle B -> A ----------

    def save_models(self, base_path: Union[str]):
        self.G_ab.save_model(base_path + "/G_ab/")
        self.G_ba.save_model(base_path + "/G_ba/")
        self.D_ab.save_model(base_path + "/D_ab/")
        self.D_ba.save_model(base_path + "/D_ba/")

    def transfer(self, sentences: List[str], direction: str):
        if direction == "AB":
            transferred_sentences = self.G_ab.transfer(sentences, device=self.device)
        else:
            transferred_sentences = self.G_ba.transfer(sentences, device=self.device)
        return transferred_sentences
