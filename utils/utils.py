

def get_mean_std(features, eps = 1e-5):
    size = features.size()
    n_batch, channel_size = size[:2]
    features_var = features.view(n_batch, channel_size, -1).var(dim = 2) + eps
    features_std = features_var.sqrt().view(n_batch, channel_size, 1, 1)
    features_mean = features.view(n_batch, channel_size, -1).mean(dim = 2).view(n_batch, channel_size, 1, 1)

    return features_mean, features_std