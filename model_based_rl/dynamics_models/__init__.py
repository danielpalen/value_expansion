from model_based_rl.dynamics_models import analytic_model
from model_based_rl.dynamics_models import network_model

available_models = {
    'analytic': analytic_model.DynamicsModel,
    'network': network_model.DynamicsModel,
}
