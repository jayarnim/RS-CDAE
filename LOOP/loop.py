from IPython.display import clear_output
from statistics import mean
import torch


class TrainingLoop:
    def __init__(
        self, 
        model, 
        trainer,
    ):
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.trainer = trainer

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        n_epochs: int, 
    ):
        trn_task_loss_list = []
        val_task_loss_list = []
        computing_cost_list = []

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
                n_epochs=n_epochs,
            )
            trn_task_loss, val_task_loss, computing_cost = self.trainer.fit(**kwargs)

            trn_task_loss_list.append(trn_task_loss)
            val_task_loss_list.append(val_task_loss)
            computing_cost_list.extend(computing_cost)

            print(
                f"TRN TASK LOSS: {trn_task_loss:.4f}",
                f"VAL TASK LOSS: {val_task_loss:.4f}",
                sep='\n'
            )

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        history = dict(
            trn=trn_task_loss_list,
            val=val_task_loss_list,
        )

        clear_output(wait=False)

        print(
            "COMPUTING COST FOR LEARNING",
            f"\t(s/epoch): {sum(computing_cost_list)/n_epochs:.4f}",
            f"\t(epoch/s): {n_epochs/sum(computing_cost_list):.4f}",
            f"\t(s/batch): {mean(computing_cost_list):.4f}",
            f"\t(batch/s): {1.0/mean(computing_cost_list):.4f}",
            sep="\n",
        )

        return history