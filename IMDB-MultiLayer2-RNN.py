import os
import torch
import torchtext
import pytorch_lightning

class Model(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self.glove = torchtext.vocab.GloVe(name='6B', dim=100, cache='../../data/vector_cache/')
        self.vocab = torchtext.vocab.vocab(self.glove.stoi)
        self.vocab.insert_token('<unk>', 0)
        self.vocab.set_default_index(0)

        self.Embedding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(self.glove.vectors, freeze=True),
            torch.nn.Dropout(),
            )
        self.RNN = torch.nn.RNN(100, 256, num_layers=2)
        self.Classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 2),
            torch.nn.Softplus(),
            )

    def forward(self, text):
        embedded = self.Embedding(text) #[S, batch_size, E]
        hidden, _ = self.RNN(embedded)
        return self.Classifier(hidden[-1])

    def prepare_data(self):
        return

    def setup(self, stage):
        #python -m spacy download en_core_web_sm
        tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')

        if (stage == 'fit'):
            train_set, valid_set = torchtext.datasets.IMDB(root='../../data/', split=('train', 'test'))
            self.train_data = [(torch.tensor(self.vocab(tokenizer(text.lower()))), label-1) for (label, text) in train_set]
            self.valid_data = [(torch.tensor(self.vocab(tokenizer(text.lower()))), label-1) for (label, text) in valid_set]
        elif (stage == 'test'):
            test_set = torchtext.datasets.IMDB(root='../../data/', split=('test'))
            self.test_data = [(torch.tensor(self.vocab(tokenizer(text.lower()))), label-1) for (label, text) in test_set]

    def _collate_fn(self, _data):
        text_Batch = torch.nn.utils.rnn.pad_sequence([text for (text, _) in _data], padding_value=self.vocab['<pad>'])
        label_Batch = torch.tensor([label for (_, label) in _data])
        return text_Batch, label_Batch

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=100, shuffle=True, collate_fn=self._collate_fn, num_workers=os.cpu_count())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_data, batch_size=100, collate_fn=self._collate_fn, num_workers=os.cpu_count())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=100, collate_fn=self._collate_fn, num_workers=os.cpu_count())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss'
                }
            }

    def configure_callbacks(self):
        return [pytorch_lightning.callbacks.early_stopping.EarlyStopping("train_loss", patience=15, check_on_train_epoch_end=True)]

    def training_step(self, batch, batch_idx):
        text, label = batch
        pred_label = self(text)
        loss = torch.nn.functional.cross_entropy(pred_label, label)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=text.size(1), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch
        pred_label = self(text)
        loss = torch.nn.functional.cross_entropy(pred_label, label)
        acc = torch.mean(pred_label.argmax(-1) == label, dtype=torch.float)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=text.size(1), sync_dist=True)

    def test_step(self, batch, batch_idx):
        text, label = batch
        pred_label = self(text)
        acc = torch.mean(pred_label.argmax(-1) == label, dtype=torch.float)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=text.size(1), sync_dist=True)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            print()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    model = Model()

    trainer = pytorch_lightning.Trainer(
        max_epochs=150,
        accelerator='auto', strategy='ddp',
        default_root_dir=os.path.splitext(__file__)[0]
        )
    trainer.fit(model)
    trainer.test(model)
