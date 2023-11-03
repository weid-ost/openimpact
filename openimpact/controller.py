from abc import abstractmethod
from typing import Protocol

import lightning as L  # type: ignore
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader  # type: ignore

from openimpact.model.gnn import get_checkpoint, load_model


class Data:
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self._dataset = None
        self._dataloader = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataloader(self):
        return self._dataloader

    def update(self, data):
        self._dataloader = data


class Controller(Protocol):
    @abstractmethod
    def predict(self):
        ...

    @abstractmethod
    def post(self, data):
        ...


class DeployApiController(Controller):
    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        data: Data,
    ):
        super().__init__()

        self.model = model
        self.trainer = trainer
        self.data = data

    def predict(self):
        # self.y_hat = self.trainer.predict(self.model, None)
        print(self.data.dataloader)

    def post(self, data):
        self.data.update(data)


class API:
    def __init__(self, controller: Controller):
        self.controller = controller

    def post(self, data):
        self.controller.post(data)

    def predict(self):
        self.controller.predict()


def main():
    ckpt = get_checkpoint("lightning_logs")

    model = load_model(ckpt)

    logger = CSVLogger("logs", "production_logs")
    trainer = L.Trainer(logger=logger)
    data = Data("/path/to/dir")

    controller: Controller = DeployApiController(model, trainer, data)

    api = API(controller)

    api.predict()
    api.post("Updated DataLoader")
    api.predict()


if __name__ == "__main__":
    main()
