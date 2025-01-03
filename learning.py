import os
from time import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import seaborn as sns
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_forecasting import (
    DeepAR,
    EncoderNormalizer,
    NormalDistributionLoss,
    TimeSeriesDataSet,
)


class DeepAREnhanced(DeepAR):
    def forward(self, x):
        if isinstance(self.rnn, (nn.RNN, nn.LSTM, nn.GRU)):
            self.rnn.flatten_parameters()
        return super().forward(x)


class MinMaxScalerVectorized(EncoderNormalizer):
    def __call__(self, tensor):
        scale = 1.0 / tensor.ptp(dim=1, keepdim=True)
        tensor.mul_(scale).sub_(tensor.amin(dim=1, keepdim=True))
        return tensor


class ScaledNormalLoss(NormalDistributionLoss):
    def __init__(self, scale_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def loss(self, y_pred, target):
        base_loss = super().loss(y_pred, target)
        return self.scale_factor * base_loss


def plot_comparison(actuals, predicted, time):
    plt.figure(figsize=(18, 9))
    # empty_elements = np.full(len(actuals)-len(predicted), np.nan)
    # predicted = np.concatenate((empty_elements, predicted))
    # print(len(actuals), len(predicted))

    sns.lineplot(x=time, y=actuals, label='Реальні значення',
                 color='gold', linewidth=2)
    sns.lineplot(x=time, y=predicted, label='Передбачені значення',
                 color='purple', linewidth=2, alpha=0.7)

    plt.title('Передбачення для BTCUSDT', fontsize=20)
    plt.xlabel('Дата', fontsize=18)
    plt.ylabel('Ціна', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.savefig('btc.png', dpi=250)
    plt.show()


def initialize_environment():
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    sns.set_theme(style="darkgrid", context="talk")
    sns.color_palette("pastel")
    pl.seed_everything(42)


def load_data(filename, time_interval=7200):
    df = pd.read_csv(filename, parse_dates=["Timestamp"])
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

    df['Next_Close'] = pd.to_numeric(df['Next_Close'], downcast='float')

    df['time_idx'] = (
        (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
        // time_interval
    ).astype("int32")

    df['group_id'] = df.get("group_id", 1).astype("uint8")

    df = df.astype({
        col: 'float32' for col in df.select_dtypes(include='float64').columns
    })

    return df


def main():
    load_start_time = time()
    train_data = load_data('SOLUSDT.csv')[-10000:]
    test_data = load_data('SOLUSDT.csv')
    test_data = test_data[-400:-100]
    print(f"Data loaded in {time() - load_start_time:.2f} seconds")

    dataset_creation_start = time()
    max_encoder_length = 120
    max_prediction_length = 10
    training_cutoff = train_data["time_idx"].max() - max_prediction_length * 30

    training_dataset = TimeSeriesDataSet(
        train_data.query("time_idx <= @training_cutoff"),
        time_idx="time_idx",
        target="Next_Close",
        group_ids=['group_id'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["Next_Close"],
        allow_missing_timesteps=True,
        target_normalizer=MinMaxScalerVectorized(),
    )

    test_dataset = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target="Next_Close",
        group_ids=['group_id'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["Next_Close"],
        allow_missing_timesteps=True,
        target_normalizer=MinMaxScalerVectorized(),
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        train_data,
        min_prediction_idx=training_cutoff + 1
    )

    print(f"Dataset created in {time() - dataset_creation_start:.2f} seconds")

    batch_size = 64
    num_workers = 11

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=False
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=False
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=False
    )

    model = DeepAR.from_dataset(
        training_dataset,
        learning_rate=0.001,
        log_interval=10,
        hidden_size=256,
        rnn_layers=8,
        dropout=0.6,
        loss=ScaledNormalLoss(scale_factor=12),
        optimizer="adamw",
    )

    logger = TensorBoardLogger("logs", name="my_model")
    model.rnn.flatten_parameters()
    trainer = Trainer(
        max_epochs=50,
        gradient_clip_val=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20),
            LearningRateMonitor(),
        ],
        benchmark=True,
        strategy=DeepSpeedStrategy(config="deepspeed_config.json"),
        logger=logger,
    )

    model_training_start = time()
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print(f"Model trained in {time() - model_training_start:.2f} seconds")

    model_predictions_start = time()
    model.eval()
    predictions = model.predict(test_dataloader)[:, 0].detach().cpu().numpy()[:100]

    actuals = np.array(train_data["Next_Close"][-100:])
    print(predictions, actuals)

    print(f"Predictions generated in {time() - model_predictions_start:.2f} seconds")
    plot_comparison(actuals, predictions,
                    pd.to_datetime(test_data['Timestamp'][:100], unit="d"))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    initialize_environment()
    main()
