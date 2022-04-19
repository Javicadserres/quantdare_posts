import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from model.scores import AverageCoverageError, IntervalScorePaper
from model.losses import SmoothPinballLoss

def get_scores(y_pred, target, quantiles):
    """
    Get prediction scores.
    
    Parameters
    ----------
    y_pred: torch.tensor
    target: torch.tensor
    quantiles: torch.tensor
    
    Returns
    -------
    scores: pd.DataFrame
    """
    # final validation loss
    criterion = SmoothPinballLoss(quantiles)
    qs = criterion(y_pred, target)

    # interval score
    iscore = IntervalScorePaper(quantiles)
    interval_score, sharpness = iscore.forward(y_pred, target)

    # average coverage error
    acerror = AverageCoverageError(quantiles)
    ace = acerror.forward(y_pred, target)
    
    scores = pd.Series(
        [qs.item(), interval_score, sharpness, ace],
        index=['QS', 'IS', 'Sharpnees', 'ACE'],
    )
    return scores

def plot_results(
    y_preds_quantilenet, y_preds_quantilenet_att, y_test
):
    """
    Plots predictions
    """
    n_quantiles = len(y_preds_quantilenet_att.columns)
    viridis = cm.get_cmap('viridis', n_quantiles)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 12), sharex=True)

    for i in range(n_quantiles-1):
        ax[0, 0].fill_between(
            y_preds_quantilenet_att.index[:200], 
            y_preds_quantilenet_att[i][:200], 
            y_preds_quantilenet_att[i+1][:200], 
            color=viridis.colors[i]
        )  
    y_test[:200].plot(color='red', ax=ax[0, 0])
    ax[0, 0].set_title('Returns predictions using attention Model')
    ax[0, 0].grid()

    for i in range(n_quantiles-1):
        ax[0, 1].fill_between(
            y_preds_quantilenet.index[:200], 
            y_preds_quantilenet[i][:200], 
            y_preds_quantilenet[i+1][:200], 
            color=viridis.colors[i]
        )
    y_test[:200].plot(color='red', ax=ax[0, 1])
    ax[0, 1].set_title('Returns predictions using LSTM')
    ax[0, 1].grid() 

    y_cum = (1 + y_test).cumprod()
    y_preds_quantilenet_att_cum = (1 + y_preds_quantilenet_att).mul(y_cum, axis=0)
    y_preds_quantilenet_cum = (1 + y_preds_quantilenet).mul(y_cum, axis=0)

    for i in range(n_quantiles-1):
        ax[1, 0].fill_between(
            y_preds_quantilenet_att_cum.index[:200], 
            y_preds_quantilenet_att_cum[i][:200], 
            y_preds_quantilenet_att_cum[i+1][:200], 
            color=viridis.colors[i]
        )  
    y_cum[:200].plot(color='red', ax=ax[1, 0])
    ax[1, 0].set_title('Predictions using attention Model')
    ax[1, 0].grid()

    for i in range(n_quantiles-1):
        ax[1, 1].fill_between(
            y_preds_quantilenet_cum.index[:200], 
            y_preds_quantilenet_cum[i][:200], 
            y_preds_quantilenet_cum[i+1][:200], 
            color=viridis.colors[i]
        )
    y_cum[:200].plot(color='red', ax=ax[1, 1])
    ax[1, 1].set_title('Predictions using LSTM')
    ax[1, 1].grid() 
    
    plt.show()
    

def plot_losses(trainer, trainer_att):
    """
    Plot trainer losses
    """
    loss_ss = pd.Series(trainer.train_losses[1:])
    validations_losses_ss = pd.Series(trainer.val_losses)

    loss_ss_att = pd.Series(trainer_att.train_losses[1:])
    validations_losses_ss_att = pd.Series(trainer_att.val_losses)

    fig, ax = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)

    loss_ss.plot(ax=ax[0])
    loss_ss_att.plot(ax=ax[0])
    validations_losses_ss.plot(ax=ax[1])
    validations_losses_ss_att.plot(ax=ax[1])

    ax[0].set_ylabel('Loss')
    ax[0].set_title('Train Loss per epoch')
    ax[0].legend(['LSTM', 'Attention + LSTM'])
    ax[0].grid()

    ax[1].set_title('Validation Loss per epoch')
    ax[1].legend(['LSTM', 'Attention + LSTM'])
    ax[1].grid()
    
    plt.show()
