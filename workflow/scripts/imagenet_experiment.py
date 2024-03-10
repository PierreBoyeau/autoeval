# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import plotnine as gg
from tqdm import tqdm
from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
from scipy.stats import spearmanr, rankdata


from ppi_py.utils import (
    construct_weight_vector,
    reshape_to_2d,
)


def _calc_lhat_glm(
    grads,
    grads_hat,
    grads_hat_unlabeled,
    inv_hessian,
    coord=None,
    clip=False,
    optim_mode="overall",
):
    grads = reshape_to_2d(grads)
    grads_hat = reshape_to_2d(grads_hat)
    grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    if grads.shape[1] != d:
        raise ValueError(
            "Dimension mismatch between the gradient and the inverse Hessian."
        )

    grads_cent = grads - grads.mean(axis=0)
    grad_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * (
        grads_cent.T @ grad_hat_cent + grad_hat_cent.T @ grads_cent
    )

    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )
    var_grads_hat = var_grads_hat.reshape(d, d)

    vhat = inv_hessian if coord is None else inv_hessian[coord, coord]
    if optim_mode == "overall":
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
        lhat = num / denom
        lhat = lhat.item()
    elif optim_mode == "element":
        num = np.diag(vhat @ cov_grads @ vhat)
        denom = 2 * (1 + (n / N)) * np.diag(vhat @ var_grads_hat @ vhat)
        lhat = num / denom
    else:
        raise ValueError(
            "Invalid value for optim_mode. Must be either 'overall' or 'element'."
        )
    if clip:
        lhat = np.clip(lhat, 0, 1)
    return lhat


def ppi_mean_std(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lambd_optim_mode="overall",
):
    """Computes the prediction-powered confidence interval for a d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Y.shape[1] if len(Y.shape) > 1 else 1

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lhat is None:
        ppi_pointest = ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=1,
            w=w,
            w_unlabeled=w_unlabeled,
        )
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
            optim_mode=lambd_optim_mode,
        )
        return ppi_mean_std(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    imputed_std = (w_unlabeled * (lhat * Yhat_unlabeled)).std(0) / np.sqrt(N)
    rectifier_std = (w * (Y - lhat * Yhat)).std(0) / np.sqrt(n)

    return np.sqrt(imputed_std**2 + rectifier_std**2)



plt.rcParams['svg.fonttype'] = 'none'

# %%
result_files = [
    'results/imagenet/resnet18.npy',
    'results/imagenet/resnet34.npy',
    'results/imagenet/resnet50.npy',
    'results/imagenet/resnet101.npy',
    'results/imagenet/resnet152.npy',
]
gt_file = 'results/imagenet/gt_labels.npy'


# %%
gt_labels = np.load(gt_file)

probs_ = []
y_preds = []
model_names = []
for file in result_files:
    model_name = file.split('/')[-1].split('.')[0]
    data = np.load(file)
    y_pred = np.argmax(data, axis=1)
    
    probs_.append(data[:, None])
    y_preds.append(y_pred[:, None])
    model_names.append(model_name)
probs_ = np.concatenate(probs_, axis=1)
y_preds = np.concatenate(y_preds, axis=1)

print(probs_.shape, y_preds.shape, gt_labels.shape)

# %%
mu_gt = (y_preds == gt_labels[:, None]).mean(axis=0)

print(mu_gt)

mu_gt_df = pd.DataFrame({
    'model': model_names,
    'accuracy': mu_gt
})

(
    gg.ggplot(mu_gt_df, gg.aes(x='model', y='accuracy')) 
    + gg.geom_bar(stat='identity', fill='blue')
    + gg.labs(x="")
)

# %%
z_all = (y_preds == gt_labels[:, None]).astype(float)
z_tilde_all = probs_.max(axis=-1)

# %%
n = 1000
alpha = 0.1
z_alpha = norm.ppf(1.0 - alpha / 2)


def build_ci(mu, sigma, n, alpha=0.1):
    z_alpha = norm.ppf(1.0 - alpha / 2)
    std = np.sqrt(np.diag(sigma))
    vmin = mu - z_alpha * std / np.sqrt(n)
    vmax = mu + z_alpha * std / np.sqrt(n)
    return vmin, vmax


def does_intersect(vmin, vmax, vmin_b, vmax_b):
    return not ((vmax < vmin_b) or (vmax_b < vmin))


def get_ranks(vmins, vmaxs):
    ties = np.array([
        [does_intersect(vmins[i], vmaxs[i], vmins[j], vmaxs[j])
        for i in range(len(vmins))] for j in range(len(vmins))
    ])
    ranks_ = rankdata(vmins)
    for i in range(len(vmins)):
        intersect_with_i = ties[i]
        ranks_intersect = ranks_[intersect_with_i]
        rank_to_use = ranks_intersect.max()
        ranks_[intersect_with_i] = rank_to_use
    return ranks_


def experiment(z, z_tilde_l, z_tilde_u):
    marginal_exp = []
    n = z.shape[0]
    m_models = z.shape[1]
    
    
    mu_classic = z.mean(axis=0)
    cov_classic = z.var(axis=0)
    vmin_classic, vmax_classic = build_ci(
        mu_classic, np.diag(cov_classic), n, alpha=alpha,
    )
    vmin_b, vmax_b = build_ci(
        mu_classic, np.diag(cov_classic), n, alpha=alpha / m_models,
    )
    ranks_classic = get_ranks(vmin_b, vmax_b)
    classic_marg_ = pd.DataFrame(
        {
            'model': model_names,
            'mean': mu_classic,
            'vmin': vmin_classic,
            'vmax': vmax_classic,
            'vmin_b': vmin_b,
            'vmax_b': vmax_b,
            'rank': ranks_classic,
            'rank_gt': rankdata(mu_gt),
            'method': 'classic',
            "ess": n,
            "cov_ratio": 1.0
        }
    )
    marginal_exp.append(classic_marg_)

    vmin, vmax = ppi_mean_ci(
        z, 
        z_tilde_l, 
        z_tilde_u, 
        alpha=alpha, 
        lambd_optim_mode="element",
        lhat=None,
    )
    vmin_b, vmax_b = ppi_mean_ci(
        z,
        z_tilde_l,
        z_tilde_u,
        alpha=alpha / m_models,
        lambd_optim_mode="element",
        lhat=None,
    )
    ranks_ppi = get_ranks(vmin_b, vmax_b)
    mu_ppi = (vmin + vmax) / 2
    std_ppi = ppi_mean_std(
        z, 
        z_tilde_l, 
        z_tilde_u,
        lambd_optim_mode="element",
        lhat=None,
    )
    std_ppi = std_ppi * np.sqrt(n)
    var_ppi = std_ppi ** 2
    cov_ratio = cov_classic / var_ppi
    ess = n * cov_ratio
    ppi_marg_ = pd.DataFrame(
        {
            'model': model_names,
            'mean': mu_ppi,
            'vmin': vmin,
            'vmax': vmax,
            'vmin_b': vmin_b,
            'vmax_b': vmax_b,
            'rank': ranks_ppi,
            'rank_gt': rankdata(mu_gt),
            'method': 'PPI++',
            "ess": ess,
            "cov_ratio": cov_ratio
        }
    )
    marginal_exp.append(ppi_marg_)

    vmin, vmax = ppi_mean_ci(z, z_tilde_l, z_tilde_u, alpha=alpha, lhat=1.0)
    vmin_b, vmax_b = ppi_mean_ci(z, z_tilde_l, z_tilde_u, alpha=alpha / m_models, lhat=1.0)
    ranks_ppi = get_ranks(vmin_b, vmax_b)
    mu_ppi = (vmin + vmax) / 2
    std_ppi = ppi_mean_std(z, z_tilde_l, z_tilde_u, lhat=1.0)
    std_ppi = std_ppi * np.sqrt(n)
    var_ppi = std_ppi ** 2
    cov_ratio = cov_classic / var_ppi
    ess = n * cov_ratio
    ppi_marg_ = pd.DataFrame(
        {
            'model': model_names,
            'mean': mu_ppi,
            'vmin': vmin,
            'vmax': vmax,
            'vmin_b': vmin_b,
            'vmax_b': vmax_b,
            'rank': ranks_ppi,
            'rank_gt': rankdata(mu_gt),
            'method': 'PPI',
            "ess": ess,
            "cov_ratio": cov_ratio
        }
    )
    marginal_exp.append(ppi_marg_)

    marginal_exp = pd.concat(marginal_exp)
    return marginal_exp




# %%
all_res = []
for n in tqdm([50, 100, 200, 300, 400, 500, 1000]):
# for n in tqdm([1000, 2000, 5000]):
    for mc_sample in range(500):
        lab_indices = np.random.choice(np.arange(len(gt_labels)), n, replace=False)
        is_lab = np.isin(np.arange(len(gt_labels)), lab_indices)

        z = z_all[is_lab]
        z_tilde_l = z_tilde_all[is_lab]
        z_tilde_u = z_tilde_all[~is_lab]
        

        exp_ = (
            experiment(z, z_tilde_l, z_tilde_u)
            .assign(
                n=n, 
                mc_sample=mc_sample,
            )
            .merge(mu_gt_df, on='model', how='left')
            .assign(
                gt_in_ci=lambda x: ((x.vmin <= x.accuracy) & (x.vmax >= x.accuracy)),
                ci_width=lambda x: x.vmax - x.vmin,
            )
        )
        all_res.append(exp_) 
all_res = pd.concat(all_res)

        
# %%
CMAP = {
    "PPI": "#E41A1C",
    "PPI++": "#377EB8",
    "classic": "#4DAF4A",
}
all_res["method"].unique()  

THEME = (
    gg.theme(
        legend_position='none',
        axis_text=gg.element_text(size=6),
        axis_title=gg.element_text(size=7),
    )
)
        
# all_marginal_exp = pd.concat(all_marginal_exp)
# %%
model_name = 'resnet50'

width = 0.001
plot_df = (
    all_res
    .query(f'model == "{model_name}"')
    .query("mc_sample == 0")
    .query("n!=1000")
)
delta_x = 10
plot_df.loc[plot_df.loc[:, "method"] == "PPI", "n"] += delta_x
plot_df.loc[plot_df.loc[:, "method"] == "PPI++", "n"] += 2 * delta_x
fig = (
    gg.ggplot(plot_df, gg.aes(x='n', y='mean', color='method', group='model'))
    + gg.geom_point(position=gg.position_dodge(width=width))
    + gg.geom_hline(gg.aes(yintercept="accuracy"), linetype='dashed')
    + gg.geom_errorbar(
        gg.aes(ymin='vmin', ymax='vmax'), 
        width=0.0,
        position=gg.position_dodge(width=1),
        # position="dodge2"
    )
    + gg.scale_color_manual(values=CMAP)
    + gg.labs(x="# of labeled samples", y="accuracy")
    + gg.theme_classic()
    + THEME
    + gg.theme(
        figure_size=(2, 2),
    )
)
fig



# %%
calib = (
    all_res
    .groupby(['n', 'method'])
    .gt_in_ci
    .mean()
    .to_frame('coverage')
    .reset_index()
)

# calib.loc[:, "n"] = calib.n.astype(str)
fig_calib = (
    gg.ggplot(calib, gg.aes(x='n', y='coverage', color='method'))
    + gg.geom_point()
    # + gg.geom_boxplot()
    + gg.geom_hline(yintercept=1 - alpha, linetype='dashed', color='black')
    + gg.labs(x="# of labeled samples", y="coverage")
    + gg.ylim(0.8, 1.0)
    # + gg.scale_y_continuous(breaks=[0.8, 0.9, 1.0])
    + gg.theme_classic()
    + gg.scale_color_manual(values=CMAP)
    + THEME
    + gg.theme(
        figure_size=(2, 2),
    )
    
)
gg.ggsave(fig_calib, 'results/imagenet/imagenet_calib.svg')
# fig_calib.save("results/imagenet/imagenet_calib.svg")
fig_calib
# %%
widths = (
    all_res
    # .groupby(['n', 'model', 'method'])
    .groupby(['n', 'method'])
    .ci_width
    .mean()
    .to_frame("ci_width")
    .reset_index()
)
# calib.loc[:, "n"] = calib.n.astype(str)
fig_widths = (
    gg.ggplot(widths, gg.aes(x='n', y='ci_width', color='method'))
    + gg.geom_point()
    # + gg.geom_boxplot()
    + gg.labs(x="# of labeled samples", y="CI width")
    + gg.theme_classic()
    + THEME
    + gg.theme(
        figure_size=(2, 2),
    )
    + gg.scale_color_manual(values=CMAP)
)
gg.ggsave(fig_widths, 'results/imagenet/imagenet_width.svg')
# fig_widths.save("results/imagenet/imagenet_width.svg")
fig_widths

# %%

ess_plot = (
    all_res
    .loc[lambda x: x.method != "classic"]
    .groupby(['n', 'method'])
    .ess
    .mean()
    .to_frame("ess")
    .reset_index()
)
ess_plot
fig_ess = (
    gg.ggplot(ess_plot, gg.aes(x='n', y='ess', color='method'))
    # + gg.geom_point()
    + gg.geom_point(position=gg.position_dodge(width=25))
    + gg.labs(x="# of labeled samples", y="ESS")
    + gg.theme_classic()
    + THEME
    + gg.geom_abline(slope=1, intercept=0, linetype='dashed', color='black')
    + gg.theme(
        figure_size=(2, 2),
    )
    + gg.scale_x_continuous(limits=(0, 1800))
    + gg.scale_y_continuous(limits=(0, 1800))
    + gg.scale_color_manual(values=CMAP)
)
# gg.ggsave(fig_ess, 'results/imagenet/imagenet_ess.svg')
fig_ess

# %%
mse_plot = (
    all_res
    .assign(mse=lambda x: (x["mean"] - x.accuracy) ** 2)
)
# mse_plot.loc[:, "n"] = mse_plot.n.astype(str)
fig_mse = (
    gg.ggplot(mse_plot, gg.aes(x='n', y='mse', color='method'))
    + gg.geom_point(stat="summary", fun_y=np.mean, position=gg.position_dodge(width=0.9))
    # + gg.geom_boxplot(outlier_alpha=0.0, outlier_shape=None, outlier_size=0.0)
    + gg.labs(x="# of labeled samples", y="MSE")
    + gg.theme_classic()
    + THEME
    + gg.scale_color_manual(values=CMAP)
    + gg.theme(
        figure_size=(2, 2),
    )
    # + gg.ylim(0, 0.01)
    # + gg.facet_wrap("n", scales="free", ncol=3)
)
# gg.ggsave(fig_mse, 'results/imagenet/imagenet_mse.svg')
fig_mse

# %%
all_res
# %%
def get_spearman(x):
    corr = spearmanr(x["rank"], x["rank_gt"])[0]
    return corr if not np.isnan(corr) else 0.0


spear_plot = (
    all_res
    .groupby(['n', 'method', 'mc_sample'])
    .apply(get_spearman)
    .to_frame("spearman")
    .reset_index()
)
spear_plot

# %%
fig = (
    gg.ggplot(spear_plot, gg.aes(x='factor(n)', y="spearman", fill='method'))
    + gg.geom_bar(stat="summary", fun_y=np.mean, position=gg.position_dodge(width=0.9))
    # + gg.geom_line()
    + gg.labs(x="# of labeled samples", y="Rank correlation")
    + gg.theme_classic()
    + THEME
    + gg.scale_fill_manual(values=CMAP)
    + gg.theme(
        figure_size=(2, 2),
    )
)
fig.save("results/imagenet/imagenet_ranks_spearman.svg")
fig
