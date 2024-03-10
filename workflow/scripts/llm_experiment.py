# %%
from datasets import load_dataset
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from numpyro.distributions import Bernoulli
import jaxopt

import plotnine as gg

plt.rcParams['svg.fonttype'] = 'none'


# %%
class BTModel(nn.Module):
    n_classes: int

    def setup(self):
        self.zetas = self.param("zetas", nn.initializers.zeros, (self.n_classes - 1))

    def __call__(self, x, y):
        idx_a = x[..., 0].astype(int)
        idx_b = x[..., 1].astype(int)
        zetas = jnp.concatenate([jnp.zeros(1), self.zetas])
        zeta_i = zetas[idx_a]
        zeta_j = zetas[idx_b]
        logits = zeta_i - zeta_j
        loss = - Bernoulli(logits=logits).log_prob(y)
        return {
            "loss": loss,
        }


def optimize(model, x, y):
    x0 = jnp.ones((32, 2))
    y0 = jnp.ones((32,))
    zetas = model.init(jax.random.PRNGKey(0), x0, y0)
    zetas_ = zetas["params"]["zetas"]
    
    def loss_fn(zetas, x, y):
        zetas_params = {"params": {"zetas": zetas}}
        return model.apply(zetas_params, x, y)["loss"].mean()
    
    solver = jaxopt.BFGS(fun=loss_fn, maxiter=1000)
    zeta_opt = solver.run(zetas_, x, y)
    zetas_opt = zeta_opt[0]
    zetas_opt = {"params": {"zetas": zetas_opt}}
    return zetas_opt



def optimize_ppi(
    model, x_gt, y_gt, x_hat, y_hat, x_unl, y_unl, lambd_=1.0
):
    x0 = jnp.ones((32, 2))
    y0 = jnp.ones((32,))
    zetas = model.init(jax.random.PRNGKey(0), x0, y0)
    zetas_ = zetas["params"]["zetas"]
    
    def loss_fn(zetas):
        zetas_params = {"params": {"zetas": zetas}}
        loss_gt = model.apply(zetas_params, x_gt, y_gt)["loss"].mean()
        loss_hat = model.apply(zetas_params, x_hat, y_hat)["loss"].mean()
        loss_unl = model.apply(zetas_params, x_unl, y_unl)["loss"].mean()
        loss = (lambd_ * loss_unl) - (lambd_ * loss_hat) + loss_gt
        return loss
    
    solver = jaxopt.BFGS(fun=loss_fn, maxiter=1000)
    zeta_opt = solver.run(zetas_)
    zetas_opt = zeta_opt[0]
    zetas_opt = {"params": {"zetas": zetas_opt}}
    return zetas_opt



class PPIBT:
    def __init__(
        self, 
        inputs_gt, 
        inputs_hat, 
        inputs_unl, 
        lambd_mode="overall",
    ):
        self.inputs_gt = inputs_gt
        self.inputs_hat = inputs_hat
        self.inputs_unl = inputs_unl

        self.n = inputs_gt[0].shape[0]
        self.N = inputs_unl[0].shape[0]
        self.r = float(self.n) / self.N
        self.theta = None
        self.sigma = None
        self.lambd_mode = lambd_mode
        
        self.is_hessian_diagonal = False


        x_gt = self.inputs_gt[0]
        x_hat = self.inputs_hat[0]
        x_unl = self.inputs_unl[0]
        all_x = np.vstack([x_gt, x_hat, x_unl])

        self.n_classes = len(np.unique(all_x))
        self.model = BTModel(n_classes=self.n_classes)
        self.model_params = None
        self.lambd_ = None
        self.log = None

    def fit(self, lambd_=None):
        if lambd_ is None:
            lambd_ = self.get_lambda()
            self.lambd_ = lambd_
        print(lambd_)
        self.get_asymptotic_distribution(lambd_)

    def get_pointestimate(self, lambd_, **kwargs):
        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl
        model_params = optimize_ppi(
            self.model,
            lambd_=lambd_,
            x_gt=x_gt,
            y_gt=y_gt,
            x_hat=x_hat,
            y_hat=y_hat,
            x_unl=x_unl,
            y_unl=y_unl,
            **kwargs
        )
        self.model_params = model_params

        zeta = np.array(model_params["params"]["zetas"])
        return zeta

    def grad_fn(self, inputs):
        x, y = inputs

        @jax.jit
        def likelihood(model_params, x, y):
            return self.model.apply(model_params, x, y)["loss"]

        score = jax.jit(jax.jacfwd(likelihood))
        grads = score(self.model_params, x, y)

        grad_mu = grads["params"]["zetas"]
        return np.array(grad_mu)


    def hessian_fn(self, inputs):
        x, y = inputs

        @jax.jit
        def likelihood(model_params, x, y):
            return self.model.apply(model_params, x, y)["loss"]

        score = jax.jacfwd(likelihood)
        hess_fn = jax.jit(jax.jacfwd(score))

        hess_ = hess_fn(self.model_params, x, y)
        mu_mu = np.array(hess_["params"]["zetas"]["params"]["zetas"].mean(0))
        return mu_mu

    def invert_hess(self, hess):
        if self.is_hessian_diagonal:
            diag_ = np.diag(hess)
            if 0.0 in diag_:
                raise ValueError("Hessian is singular")
            inv_ess = 1.0 / diag_
            return np.diag(inv_ess)
        return np.linalg.pinv(hess)
    
    def get_asymptotic_distribution(
        self,
        lambd_=None,
    ):
        if lambd_ is None:
            lambd_ = self.get_lambda()
        self.theta = self.get_pointestimate(lambd_=lambd_)

        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_x_gt = self.grad_fn(self.inputs_gt)

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (lambd_**2) * (grad_f_.T @ grad_f_) / (self.n + self.N)
        rect_ = grad_x_gt - lambd_ * grad_f_hat
        rect_ = rect_ - rect_.mean(axis=0)
        vdelta = (rect_.T @ rect_) / self.n

        v = vdelta + (self.r * vf)

        hess = self.hessian_fn(self.inputs_gt)
        inv_hess = self.invert_hess(hess)

        sigma_ = inv_hess @ v @ inv_hess
        sigma_ = sigma_ / self.n
        self.sigma = sigma_
        print(lambd_)
        return self.theta, self.sigma
    
    def get_lambda(self):
        self.theta = self.get_pointestimate(lambd_=1.0)
        
        hess = self.hessian_fn(self.inputs_gt)
        inv_hess = self.invert_hess(hess)

        grad_f_unl = self.grad_fn(self.inputs_unl)
        grad_f_hat = self.grad_fn(self.inputs_hat)
        grad_f_all = np.vstack([grad_f_hat, grad_f_unl])
        grad_f_gt = self.grad_fn(self.inputs_gt)
        grad_f_hat_ = grad_f_hat - grad_f_hat.mean(0)
        grad_f_gt_ = grad_f_gt - grad_f_gt.mean(0)
        cov1 = (grad_f_hat_.T @ grad_f_gt_) / self.n
        cov2 = (grad_f_gt_.T @ grad_f_hat_) / self.n

        grad_f_ = grad_f_all - grad_f_all.mean(axis=0)
        vf = (grad_f_.T @ grad_f_) / (self.n + self.N)
        num = inv_hess @ (cov1 + cov2) @ inv_hess
        denom = 2 * (1.0 + self.r) * (inv_hess @ vf @ inv_hess)
        return np.trace(num) / np.trace(denom)

# %%
def parse_conversation(conversation):
    conversation_str = ""
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        new_msg = f"{role}: {content}\n"
        conversation_str += new_msg
    return conversation_str


def parse_data(data):
    def parse_observation(obs):
        conversation_a = parse_conversation(obs["conversation_a"])
        conversation_b = parse_conversation(obs["conversation_b"])
        winner = obs["winner"]
        model_a = obs["model_a"]
        model_b = obs["model_b"]
        question = obs["conversation_a"][0]["content"]
        answer_a = parse_conversation(obs["conversation_a"][1:])
        answer_b = parse_conversation(obs["conversation_b"][1:])
        question_id = obs["question_id"]
        comparison_tag = sorted([model_a, model_b])
        comparison = f"{comparison_tag[0]} vs {comparison_tag[1]}"
        model_a_lex = comparison_tag[0]
        model_b_lex = comparison_tag[1]
        
        
        is_lexicographic = model_a < model_b
        
        score = 0.0
        score_lex = 0.0
        if winner == "model_a":
            score = 1.0
            score_lex = 1.0 if is_lexicographic else 0.0
        elif winner == "model_b":
            score = 0.0
            score_lex = 0.0 if is_lexicographic else 1.0
        elif "tie" in winner:
            score = 0.5
            score_lex = 0.5
        else:
            raise ValueError("Invalid winner")
        return {
            "conversation_a": conversation_a,
            "conversation_b": conversation_b,
            "winner": winner,
            "model_a": model_a,
            "model_b": model_b,
            "comparison": comparison,
            "len_a": len(obs["conversation_a"]),
            "len_b": len(obs["conversation_b"]),
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "question_id": question_id,
            "score": score,
            "score_lex": score_lex,
            "model_a_lex": model_a_lex,
            "model_b_lex": model_b_lex
        }
    
    res = []
    for obs in data:
        parsed_obs = parse_observation(obs)
        res.append(parsed_obs)
    return pd.DataFrame(res)
    



dataset = load_dataset("lmsys/mt_bench_human_judgments")
large_data = load_dataset("lmsys/chatbot_arena_conversations")["train"]
gpt_data = dataset["gpt4_pair"]
human_data = dataset["human"]

# %%
gpt_data_ = parse_data(gpt_data)
human_data_ = parse_data(human_data)

gpt_data_ = (
    gpt_data_
    .groupby(
        [
            "question_id",
            "model_a_lex",
            "model_b_lex",
        ]
    )
    .score_lex
    .mean()
    .to_frame()
    .reset_index()
)


# %%
full_data_ = (
    gpt_data_.merge(
        human_data_, 
        on=["question_id", "model_a_lex", "model_b_lex"],
        # how="outer",
        suffixes=("_pred", "_gt")
    )
)

# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()
# all_labels = np.concatenate([full_data_.model_a_lex.values, full_data_.model_b_lex.values])
# label_encoder.fit(all_labels)

# full_data_["idx_a_lex"] = label_encoder.transform(full_data_.model_a_lex)
# full_data_["idx_b_lex"] = label_encoder.transform(full_data_.model_b_lex)


replacer = {
    "gpt-3.5-turbo": 0,
    "gpt-4": 1,
    "llama-13b": 2,
    "alpaca-13b": 3,
    "claude-v1": 4,
    "vicuna-13b-v1.2": 5,
}

full_data_["idx_a_lex"] = full_data_["model_a_lex"].map(replacer)
full_data_["idx_b_lex"] = full_data_["model_b_lex"].map(replacer)

# %%
full_data_[["idx_a_lex", "model_a_lex"]]

# %%
full_data_.groupby(["model_a_lex", "model_b_lex"]).score_lex_gt.mean().sort_values()

# %%
from scipy.stats import spearmanr

spearmanr(full_data_.score_lex_gt, full_data_.score_lex_pred)

# %%

n_obs = full_data_.shape[0]
annotated_sizes = [100, 200, 300, 400, 500]

# for n in annotated_sizes:
#     for mc_sample in range(100):
# n = 25
n = 25
np.random.seed(0)
gt_samples = np.random.choice(n_obs, size=n, replace=False)
is_gt = np.isin(np.arange(n_obs), gt_samples)

data_gt = full_data_.loc[is_gt]
data_pred = full_data_.loc[~is_gt]

x_gt = data_gt.loc[:, ["idx_a_lex", "idx_b_lex"]].values
y_gt = data_gt.loc[:, "score_lex_gt"].values
y_hat = data_gt.loc[:, "score_lex_pred"].values

x_unl = data_pred.loc[:, ["idx_a_lex", "idx_b_lex"]].values
y_unl = data_pred.loc[:, "score_lex_pred"].values

x_imp = full_data_.loc[:, ["idx_a_lex", "idx_b_lex"]].values
y_imp = full_data_.loc[:, "score_lex_pred"].values

model_input = PPIBT(
    inputs_gt=(x_imp, y_imp),
    inputs_hat=(x_imp, y_imp),
    inputs_unl=(x_imp, y_imp),
    lambd_mode="overall",
)
model_input.fit(lambd_=0.0)
theta_input = model_input.theta
sigma_input = model_input.sigma

model_classic = PPIBT(
    inputs_gt=(x_gt, y_gt),
    inputs_hat=(x_gt, y_hat),
    inputs_unl=(x_unl, y_unl),
    lambd_mode="overall",
)
model_classic.fit(lambd_=0.0)
theta_classic = model_classic.theta
sigma_classic = model_classic.sigma


model_lambd = PPIBT(
    inputs_gt=(x_gt, y_gt),
    inputs_hat=(x_gt, y_hat),
    inputs_unl=(x_unl, y_unl),
    lambd_mode="overall",
    # lambd_mode="element",
)
model_lambd.fit(lambd_=None)
theta_lambd = model_lambd.theta
sigma_lambd = model_lambd.sigma


model_ppi = PPIBT(
    inputs_gt=(x_gt, y_gt),
    inputs_hat=(x_gt, y_hat),
    inputs_unl=(x_unl, y_unl),
    lambd_mode="overall",
    # lambd_mode="element",
)
model_ppi.fit(lambd_=1.0)
theta_ppi = model_ppi.theta
sigma_ppi = model_ppi.sigma
  
# %%
x_all = full_data_.loc[:, ["idx_a_lex", "idx_b_lex"]].values
y_all = full_data_.loc[:, "score_lex_gt"].values


model_gt = PPIBT(
    inputs_gt=(x_all, y_all),
    inputs_hat=(x_all, y_all),
    inputs_unl=(x_all, y_all),
    lambd_mode="overall",
)
model_gt.fit(lambd_=0.0)
theta_gt = model_gt.theta

# %%
std_classic = np.sqrt(np.diag(sigma_classic))
std_lambd = np.sqrt(np.diag(sigma_lambd))
std_ppi = np.sqrt(np.diag(sigma_ppi))
std_input = np.sqrt(np.diag(sigma_input))

# %%
alpha = 0.1 / 5
from scipy.stats import norm

z_alpha = norm.ppf(1 - alpha / 2)
vmin_classic = theta_classic - z_alpha * std_classic
vmax_classic = theta_classic + z_alpha * std_lambd

vmin_lambd = theta_lambd - z_alpha * std_lambd
vmax_lambd = theta_lambd + z_alpha * std_lambd

vmin_ppi = theta_ppi - z_alpha * std_ppi
vmax_ppi = theta_ppi + z_alpha * std_ppi

vmin_input = theta_input - z_alpha * std_input
vmax_input = theta_input + z_alpha * std_input

# %%
plot_df = pd.concat(
    [
        pd.DataFrame(
            {
                "vmin": vmin_classic,
                "vmax": vmax_classic,
                "theta": theta_classic,
                "theta_gt": theta_gt,
                "method": "classic",
                "model": list(replacer.keys())[1:]
            }
        ),
        pd.DataFrame(
            {
                "vmin": vmin_lambd,
                "vmax": vmax_lambd,
                "theta": theta_lambd,
                "theta_gt": theta_gt,
                "method": "PPI++",
                "model": list(replacer.keys())[1:]
            }
        ),
        pd.DataFrame(
            {
                "vmin": vmin_ppi,
                "vmax": vmax_ppi,
                "theta": theta_ppi,
                "theta_gt": theta_gt,
                "method": "PPI",
                "model": list(replacer.keys())[1:]
            }
        ),
    ]
)

# %%
plot_input = pd.DataFrame(
    {
        "vmin": vmin_input,
        "vmax": vmax_input,
        "theta": theta_input,
        "theta_gt": theta_gt,
        "method": "Imput.",
        "model": list(replacer.keys())[1:]
    }
)

# %%
CMAP = {
    "PPI": "#E41A1C",
    "PPI++": "#377EB8",
    "classic": "#4DAF4A",
    "PPIcheck": "#984EA3",
}

# %%
model_order = pd.DataFrame(
    {
        "model": list(replacer.keys())[1:],
        "perf": theta_gt
    }
).sort_values("perf").model.values


# %%
fig = (
    gg.ggplot(plot_df, gg.aes(x="model", color="method"))
    + gg.geom_point(
        gg.aes(y="theta"), 
        position=gg.position_dodge(width=0.2),
        size=2
    )
    + gg.coord_flip()
    + gg.theme_classic()
    + gg.theme(
        legend_position='none',
        axis_text=gg.element_text(size=6),
        axis_title=gg.element_text(size=7),
    )
    + gg.geom_errorbar(
        gg.aes(ymin="vmin", ymax="vmax"), position=gg.position_dodge(width=0.2)
    )
    + gg.geom_point(gg.aes(y="theta_gt"), color="black")
    + gg.geom_point(
        plot_input,
        gg.aes(y="theta"),
        color="black",
        shape="x",
        size=3,
    )
    + gg.labs(
        x="",
        y="BT effect size"
    )
    + gg.theme(
        figure_size=(3,2)
    )
    + gg.scale_color_manual(values=CMAP) 
    + gg.scale_x_discrete(limits=model_order)
)
fig.save("results/llm/llm_main.svg")
fig
# %%
